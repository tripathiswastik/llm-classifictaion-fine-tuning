import os
import sys
import time
import math
import zlib
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor, compute_pairwise_features, STOPWORDS, RE_WORDS
from metric import compute_log_loss
from calibration import TemperatureCalibrator
from models import build_logistic_model, build_hist_gbdt_model

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Precompiled discourse markers
RE_CONNECTIVES = re.compile(
    r"\b(furthermore|moreover|therefore|however|nevertheless|consequently|specifically|in particular|for instance|for example|in summary|in conclusion)\b",
    re.IGNORECASE,
)

def compute_entropy(words: List[str]) -> float:
    if len(words) <= 1:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def compute_compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    raw_bytes = text.encode("utf-8", errors="replace")
    compressed = zlib.compress(raw_bytes, level=6)
    return len(compressed) / max(len(raw_bytes), 1)

def extract_advanced_pairwise_features(
    prompt: str,
    resp_a: str,
    resp_b: str,
    turns_a: int = 1,
    turns_b: int = 1,
) -> Dict[str, float]:
    # 1. Base 61 hand-crafted features
    base_feats = compute_pairwise_features(prompt, resp_a, resp_b, turns_a, turns_b)

    # 2. Words & Bigrams
    p_words = [w.lower() for w in RE_WORDS.findall(prompt)]
    a_words = [w.lower() for w in RE_WORDS.findall(resp_a)]
    b_words = [w.lower() for w in RE_WORDS.findall(resp_b)]

    # Feature 1: Shannon Entropy of Word Distribution
    ent_a = compute_entropy(a_words)
    ent_b = compute_entropy(b_words)
    diff_entropy = ent_a - ent_b
    sym_mean_entropy = 0.5 * (ent_a + ent_b)

    # Feature 2: Zlib Compression Ratio (Information Density / Verbosity)
    cr_a = compute_compression_ratio(resp_a)
    cr_b = compute_compression_ratio(resp_b)
    diff_cr = cr_a - cr_b
    sym_mean_cr = 0.5 * (cr_a + cr_b)

    # Feature 3: Content Word Bigram Coverage
    p_content = [w for w in p_words if w not in STOPWORDS]
    a_content = [w for w in a_words if w not in STOPWORDS]
    b_content = [w for w in b_words if w not in STOPWORDS]

    p_bigrams = set(zip(p_content[:-1], p_content[1:])) if len(p_content) >= 2 else set()
    a_bigrams = set(zip(a_content[:-1], a_content[1:])) if len(a_content) >= 2 else set()
    b_bigrams = set(zip(b_content[:-1], b_content[1:])) if len(b_content) >= 2 else set()

    if p_bigrams:
        cov_bigram_a = len(p_bigrams & a_bigrams) / len(p_bigrams)
        cov_bigram_b = len(p_bigrams & b_bigrams) / len(p_bigrams)
    else:
        cov_bigram_a = cov_bigram_b = 0.0

    diff_bigram_cov = cov_bigram_a - cov_bigram_b
    sym_mean_bigram_cov = 0.5 * (cov_bigram_a + cov_bigram_b)

    # Feature 4: Discourse Connectives
    conn_a = float(len(RE_CONNECTIVES.findall(resp_a)))
    conn_b = float(len(RE_CONNECTIVES.findall(resp_b)))
    diff_connectives = conn_a - conn_b
    sym_connectives = 0.5 * (conn_a + conn_b)

    # Add all new features
    base_feats.update({
        "diff_entropy": diff_entropy,
        "sym_mean_entropy": sym_mean_entropy,
        "diff_compression_ratio": diff_cr,
        "sym_mean_compression_ratio": sym_mean_cr,
        "diff_bigram_cov": diff_bigram_cov,
        "sym_mean_bigram_cov": sym_mean_bigram_cov,
        "diff_connectives": diff_connectives,
        "sym_connectives": sym_connectives,
    })

    return base_feats


class AdvancedFeatureExtractor:
    def __init__(self, use_tfidf: bool = True, max_tfidf_features: int = 15000):
        self.use_tfidf = use_tfidf
        self.max_tfidf_features = max_tfidf_features
        self.base_feature_names_: List[str] = []
        self.feature_names_: List[str] = []
        self.tfidf_vectorizer = None

    def fit(self, df: pd.DataFrame) -> "AdvancedFeatureExtractor":
        sample = extract_advanced_pairwise_features(
            prompt=df["prompt_text"].iloc[0],
            resp_a=df["response_a_text"].iloc[0],
            resp_b=df["response_b_text"].iloc[0],
        )
        self.base_feature_names_ = sorted(list(sample.keys()))

        if self.use_tfidf:
            from sklearn.feature_extraction.text import TfidfVectorizer
            corpus = (
                list(df["prompt_text"].values)
                + list(df["response_a_text"].values)
                + list(df["response_b_text"].values)
            )
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 2),
            )
            self.tfidf_vectorizer.fit(corpus)
            self.feature_names_ = self.base_feature_names_ + [
                "diff_tfidf_prompt_sim",
                "sym_tfidf_ab_sim",
                "sym_tfidf_mean_prompt",
            ]
        else:
            self.feature_names_ = list(self.base_feature_names_)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        prompts = df["prompt_text"].values
        resps_a = df["response_a_text"].values
        resps_b = df["response_b_text"].values
        turns_a = df["response_a_turns"].values
        turns_b = df["response_b_turns"].values

        base_matrix = np.zeros((n, len(self.base_feature_names_)), dtype=np.float32)
        for i in range(n):
            t_a = len(turns_a[i]) if hasattr(turns_a[i], "__len__") else 1
            t_b = len(turns_b[i]) if hasattr(turns_b[i], "__len__") else 1
            feats = extract_advanced_pairwise_features(prompts[i], resps_a[i], resps_b[i], t_a, t_b)
            for j, k in enumerate(self.base_feature_names_):
                base_matrix[i, j] = feats[k]

        if not self.use_tfidf or self.tfidf_vectorizer is None:
            return base_matrix

        p_tfidf = self.tfidf_vectorizer.transform(prompts)
        a_tfidf = self.tfidf_vectorizer.transform(resps_a)
        b_tfidf = self.tfidf_vectorizer.transform(resps_b)

        sim_pa = p_tfidf.multiply(a_tfidf).sum(axis=1)
        sim_pb = p_tfidf.multiply(b_tfidf).sum(axis=1)
        sim_ab = a_tfidf.multiply(b_tfidf).sum(axis=1)

        sim_pa = np.asarray(sim_pa).flatten()
        sim_pb = np.asarray(sim_pb).flatten()
        sim_ab = np.asarray(sim_ab).flatten()

        diff_sim = (sim_pa - sim_pb).astype(np.float32)[:, None]
        sym_ab = sim_ab.astype(np.float32)[:, None]
        sym_mean = (0.5 * (sim_pa + sim_pb)).astype(np.float32)[:, None]

        return np.hstack([base_matrix, diff_sim, sym_ab, sym_mean])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


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


if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "data", "train.csv")
    print("================================================================")
    print("EXP 10: ADVANCED FEATURES (ENTROPY, COMPRESSION, BIGRAM, CONNECTIVES)")
    print("Champion Benchmark to beat: 1.02579 (10k 5-Fold StratifiedGroupKFold)")
    print("================================================================")

    df_train = load_dataset(train_csv, nrows=10000, is_test=False)
    extractor = AdvancedFeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X = extractor.fit_transform(df_train)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values

    print(f"Total Features Extracted: {len(extractor.feature_names_)} (was 64)")
    print(f"New feature list: {[f for f in extractor.feature_names_ if 'entropy' in f or 'compression' in f or 'bigram' in f or 'connectives' in f]}")

    folds = list(get_stratified_group_folds(df_train, n_splits=5))
    oof_pytorch = np.zeros_like(y_onehot, dtype=np.float64)
    oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
    oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y_onehot[val_idx]

        # 1. PyTorch GPU Model
        t_gpu = time.time()
        pt_model = PyTorchGPUPreferenceModel(extractor.feature_names_, hidden_dim=128, epochs=25)
        pt_model.fit(X_tr, y_tr)
        p_pt = pt_model.predict_proba(X_va)
        oof_pytorch[val_idx] = p_pt
        loss_pt = compute_log_loss(y_va, p_pt)
        print(f"  PyTorch GPU ({time.time() - t_gpu:.1f}s) Log Loss: {loss_pt:.5f}")

        # 2. Logistic Regression
        lr_model = build_logistic_model(extractor.feature_names_, C=0.5)
        lr_model.fit(X_tr, y_tr)
        p_lr = lr_model.predict_proba(X_va)
        oof_lr[val_idx] = p_lr
        print(f"  Logistic Regression Log Loss: {compute_log_loss(y_va, p_lr):.5f}")

        # 3. HistGBDT
        gbdt_model = build_hist_gbdt_model(extractor.feature_names_)
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

    delta = opt_cal_loss - 1.02579
    print(f"\nDelta vs Champion (1.02579): {delta:+.5f}")
    if delta < 0:
        print("GATE STATUS: PASSED (Strict Monotonic Improvement)")
    else:
        print("GATE STATUS: FAILED (No Improvement - Candidate will be REJECTED)")
