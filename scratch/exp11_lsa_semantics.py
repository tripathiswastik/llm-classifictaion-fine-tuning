import os
import sys
import time
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor
from metric import compute_log_loss
from calibration import TemperatureCalibrator
from models import build_logistic_model, build_hist_gbdt_model

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

CHAMPION_10K_BENCHMARK = 1.02548


class SymmetricMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor, x_swap: torch.Tensor) -> torch.Tensor:
        logits_orig = self.forward_single(x)
        logits_swap = self.forward_single(x_swap)
        logits_swap_inv = torch.stack([logits_swap[:, 1], logits_swap[:, 0], logits_swap[:, 2]], dim=1)
        return 0.5 * (logits_orig + logits_swap_inv)


class PyTorchGPUPreferenceModel:
    def __init__(self, feature_names: list, hidden_dim: int = 128, lr: float = 2e-3, epochs: int = 25, batch_size: int = 256):
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.diff_indices = [i for i, name in enumerate(feature_names) if name.startswith("diff_")]
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.model: Optional[SymmetricMLP] = None

    def _swap_features_tensor(self, X: torch.Tensor) -> torch.Tensor:
        X_swap = X.clone()
        X_swap[:, self.diff_indices] = -X_swap[:, self.diff_indices]
        return X_swap

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        X_swap_t = self._swap_features_tensor(X_t)
        X_aug = torch.cat([X_t, X_swap_t], dim=0)
        self.mean = X_aug.mean(dim=0, keepdim=True)
        self.mean[:, self.diff_indices] = 0.0
        self.std = X_aug.std(dim=0, keepdim=True) + 1e-7

        X_norm = (X_t - self.mean) / self.std
        X_swap_norm = (X_swap_t - self.mean) / self.std

        dataset = TensorDataset(X_norm, X_swap_norm, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SymmetricMLP(X_train.shape[1], hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_x_swap, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x, batch_x_swap)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean is None or self.std is None:
            raise ValueError("Model must be fitted before predict_proba")
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_norm = (X_t - self.mean) / self.std
        X_swap_norm = self._swap_features_tensor(X_norm)

        with torch.no_grad():
            logits = self.model(X_norm, X_swap_norm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


def extract_lsa_features(df: pd.DataFrame, n_components: int = 16) -> Tuple[np.ndarray, List[str]]:
    """
    Fits TruncatedSVD on the TF-IDF corpus to extract dense latent semantic features.
    Computes difference vectors and cosine similarities preserving strict anti-symmetry.
    """
    corpus = list(df["prompt_text"].values) + list(df["response_a_text"].values) + list(df["response_b_text"].values)
    tfidf = TfidfVectorizer(max_features=10000, sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(corpus)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf_mat)

    p_vecs = svd.transform(tfidf.transform(df["prompt_text"]))
    a_vecs = svd.transform(tfidf.transform(df["response_a_text"]))
    b_vecs = svd.transform(tfidf.transform(df["response_b_text"]))

    # Normalize vectors for cosine similarities
    def l2_norm(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)

    p_norm = l2_norm(p_vecs)
    a_norm = l2_norm(a_vecs)
    b_norm = l2_norm(b_vecs)

    cos_pa = np.sum(p_norm * a_norm, axis=1, keepdims=True)
    cos_pb = np.sum(p_norm * b_norm, axis=1, keepdims=True)
    cos_ab = np.sum(a_norm * b_norm, axis=1, keepdims=True)

    diff_cos_p = (cos_pa - cos_pb).astype(np.float32)
    sym_cos_ab = cos_ab.astype(np.float32)
    sym_mean_cos_p = (0.5 * (cos_pa + cos_pb)).astype(np.float32)

    # Dense SVD component differences and means
    diff_svd = (a_vecs - b_vecs).astype(np.float32)
    sym_svd = (0.5 * (a_vecs + b_vecs)).astype(np.float32)

    feature_names = ["diff_lsa_prompt_cos", "sym_lsa_ab_cos", "sym_lsa_mean_prompt_cos"]
    for i in range(n_components):
        feature_names.append(f"diff_lsa_dim_{i}")
        feature_names.append(f"sym_lsa_dim_{i}")

    lsa_matrix = np.hstack([diff_cos_p, sym_cos_ab, sym_mean_cos_p])
    # Interleave diff and sym dimensions
    dim_cols = []
    for i in range(n_components):
        dim_cols.append(diff_svd[:, i:i+1])
        dim_cols.append(sym_svd[:, i:i+1])
    lsa_matrix = np.hstack([lsa_matrix] + dim_cols)

    return lsa_matrix, feature_names


if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "data", "train.csv")
    print("================================================================")
    print("EXP 11: DENSE LATENT SEMANTIC ANALYSIS (LSA / TRUNCATED SVD)")
    print(f"Champion 10k Benchmark to beat: {CHAMPION_10K_BENCHMARK:.5f}")
    print("================================================================")

    df_train = load_dataset(train_csv, nrows=10000, is_test=False)
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X_base = extractor.fit_transform(df_train)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values

    print(f"Base Features Extracted: {X_base.shape[1]}")

    t_lsa = time.time()
    n_lsa_components = 16
    X_lsa, lsa_names = extract_lsa_features(df_train, n_components=n_lsa_components)
    print(f"Extracted {len(lsa_names)} LSA Semantic Features in {time.time() - t_lsa:.1f}s")

    X_all = np.hstack([X_base, X_lsa])
    all_feature_names = extractor.feature_names_ + lsa_names
    print(f"Total Combined Feature Space: {X_all.shape[1]} features")

    folds = list(get_stratified_group_folds(df_train, n_splits=5))
    oof_pytorch = np.zeros_like(y_onehot, dtype=np.float64)
    oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
    oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_tr, y_tr = X_all[train_idx], y[train_idx]
        X_va, y_va = X_all[val_idx], y_onehot[val_idx]

        # 1. PyTorch GPU Model
        t_gpu = time.time()
        pt_model = PyTorchGPUPreferenceModel(all_feature_names, hidden_dim=128, epochs=25)
        pt_model.fit(X_tr, y_tr)
        p_pt = pt_model.predict_proba(X_va)
        oof_pytorch[val_idx] = p_pt
        loss_pt = compute_log_loss(y_va, p_pt)
        print(f"  PyTorch GPU ({time.time() - t_gpu:.1f}s) Log Loss: {loss_pt:.5f}")

        # 2. Logistic Regression
        lr_model = build_logistic_model(all_feature_names, C=0.5)
        lr_model.fit(X_tr, y_tr)
        p_lr = lr_model.predict_proba(X_va)
        oof_lr[val_idx] = p_lr
        print(f"  Logistic Regression Log Loss: {compute_log_loss(y_va, p_lr):.5f}")

        # 3. HistGBDT
        gbdt_model = build_hist_gbdt_model(all_feature_names)
        gbdt_model.fit(X_tr, y_tr)
        p_gbdt = gbdt_model.predict_proba(X_va)
        oof_gbdt[val_idx] = p_gbdt
        print(f"  HistGBDT Log Loss: {compute_log_loss(y_va, p_gbdt):.5f}")

    print(f"\nTotal 5-Fold Training Time: {time.time() - t0:.1f}s")
    print(f"Raw PyTorch GPU OOF Log Loss: {compute_log_loss(y_onehot, oof_pytorch):.5f}")
    print(f"Raw Logistic Regression OOF Log Loss: {compute_log_loss(y_onehot, oof_lr):.5f}")
    print(f"Raw HistGBDT OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt):.5f}")

    # Standard weights: 0.35 LR + 0.40 GBDT + 0.25 PyTorch
    tri_blend = 0.35 * oof_lr + 0.40 * oof_gbdt + 0.25 * oof_pytorch
    tri_blend /= tri_blend.sum(axis=1, keepdims=True)
    raw_tri_loss = compute_log_loss(y_onehot, tri_blend)
    print(f"\nFixed Tri-Ensemble (0.35/0.40/0.25) Raw Log Loss: {raw_tri_loss:.5f}")

    cal_tri = TemperatureCalibrator().fit(tri_blend, y_onehot)
    cal_tri_loss = compute_log_loss(y_onehot, cal_tri.transform(tri_blend))
    print(f"Optimal Temperature: {cal_tri.temperature:.4f}, Tie Scale: {cal_tri.tie_scale:.4f}")
    print(f"Fixed Tri-Ensemble Calibrated Log Loss: {cal_tri_loss:.5f}")

    # Weight Grid Search
    best_loss = 999.0
    best_weights: Optional[Tuple[float, float, float]] = None
    for w_pt in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for w_lr in [0.25, 0.30, 0.35, 0.40, 0.45]:
            w_gb = 1.0 - w_pt - w_lr
            if w_gb < 0:
                continue
            blend = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * oof_pytorch
            blend /= blend.sum(axis=1, keepdims=True)
            loss = compute_log_loss(y_onehot, blend)
            if loss < best_loss:
                best_loss = loss
                best_weights = (float(w_lr), float(w_gb), float(w_pt))

    assert best_weights is not None, "Failed to optimize weights"
    print(f"\nTuned Tri-Ensemble Weights: LR={best_weights[0]:.2f}, GBDT={best_weights[1]:.2f}, GPU_PT={best_weights[2]:.2f}")
    opt_blend = best_weights[0] * oof_lr + best_weights[1] * oof_gbdt + best_weights[2] * oof_pytorch
    opt_blend /= opt_blend.sum(axis=1, keepdims=True)
    cal_opt = TemperatureCalibrator().fit(opt_blend, y_onehot)
    opt_cal_loss = compute_log_loss(y_onehot, cal_opt.transform(opt_blend))
    print(f"Tuned Tri-Ensemble Calibrated Log Loss: {opt_cal_loss:.5f}")

    delta = opt_cal_loss - CHAMPION_10K_BENCHMARK
    print(f"\nDelta vs Champion ({CHAMPION_10K_BENCHMARK:.5f}): {delta:+.5f}")
    if delta < 0:
        print("GATE STATUS: PASSED (Strict Monotonic Improvement)")
    else:
        print("GATE STATUS: FAILED (No Improvement - Candidate will be REJECTED)")
