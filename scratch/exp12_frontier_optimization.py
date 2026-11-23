import os
import sys
import time
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
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

CHAMPION_10K_BENCHMARK = 1.01751


class DirichletSymmetricCalibrator:
    """
    3-Parameter Decoupled Symmetric Dirichlet Calibrator.
    Decouples pair contest A vs B from tie margin:
      z_AB = log(p_A / p_B) / T_pair
      z_tie = beta_tie * log(p_tie / sqrt(p_A * p_B)) + alpha_tie
    Strictly preserves mathematical anti-symmetry and simplex constraints.
    """
    def __init__(self, t_pair: float = 1.0, beta_tie: float = 1.0, alpha_tie: float = 0.0):
        self.t_pair = float(t_pair)
        self.beta_tie = float(beta_tie)
        self.alpha_tie = float(alpha_tie)

    def fit(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> "DirichletSymmetricCalibrator":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab = log_a - log_b
        z_tie = log_tie - 0.5 * (log_a + log_b)

        def objective(params: np.ndarray) -> float:
            t_p, b_t, a_t = params
            z_ab_scaled = z_ab / max(t_p, 1e-3)
            z_tie_scaled = b_t * z_tie + a_t

            # Stable softmax
            l_a = 0.5 * z_ab_scaled
            l_b = -0.5 * z_ab_scaled
            l_tie = z_tie_scaled

            l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
            exp_a = np.exp(l_a[:, None] - l_max)
            exp_b = np.exp(l_b[:, None] - l_max)
            exp_tie = np.exp(l_tie[:, None] - l_max)

            denom = exp_a + exp_b + exp_tie
            p_cal = np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])
            return compute_log_loss(y_true_onehot, p_cal)

        res = minimize(
            objective,
            x0=[self.t_pair, self.beta_tie, self.alpha_tie],
            bounds=[(0.1, 5.0), (0.1, 5.0), (-5.0, 5.0)],
            method="L-BFGS-B",
        )
        self.t_pair = float(res.x[0])
        self.beta_tie = float(res.x[1])
        self.alpha_tie = float(res.x[2])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab = (log_a - log_b) / max(self.t_pair, 1e-3)
        z_tie = self.beta_tie * (log_tie - 0.5 * (log_a + log_b)) + self.alpha_tie

        l_a = 0.5 * z_ab
        l_b = -0.5 * z_ab
        l_tie = z_tie

        l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
        exp_a = np.exp(l_a[:, None] - l_max)
        exp_b = np.exp(l_b[:, None] - l_max)
        exp_tie = np.exp(l_tie[:, None] - l_max)

        denom = exp_a + exp_b + exp_tie
        return np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])


class SwiGLUBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc_gate = nn.Linear(dim, dim)
        self.fc_val = nn.Linear(dim, dim)
        self.fc_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        gate = self.silu(self.fc_gate(x))
        val = self.fc_val(x)
        h = self.fc_out(gate * val)
        return self.norm(res + h)


class SymmetricSwiGLUMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, num_blocks: int = 2, dropout: float = 0.15):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([SwiGLUBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)

    def forward(self, x: torch.Tensor, x_swap: torch.Tensor) -> torch.Tensor:
        logits_orig = self.forward_single(x)
        logits_swap = self.forward_single(x_swap)
        logits_swap_inv = torch.stack([logits_swap[:, 1], logits_swap[:, 0], logits_swap[:, 2]], dim=1)
        return 0.5 * (logits_orig + logits_swap_inv)


class PyTorchSwiGLUPreferenceModel:
    def __init__(self, feature_names: list, hidden_dim: int = 128, lr: float = 1e-3, epochs: int = 20, batch_size: int = 256):
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.diff_indices = [i for i, name in enumerate(feature_names) if name.startswith("diff_")]
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.model: Optional[SymmetricSwiGLUMLP] = None

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

        self.model = SymmetricSwiGLUMLP(X_train.shape[1], hidden_dim=self.hidden_dim, num_blocks=2).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

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
    print("EXP 12: FRONTIER OPTIMIZATION (SWIGLU RESNET + DIRICHLET CALIBRATOR)")
    print(f"Champion 10k Benchmark to beat: {CHAMPION_10K_BENCHMARK:.5f}")
    print("================================================================")

    df_train = load_dataset(train_csv, nrows=10000, is_test=False)
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000, use_lsa=True, n_lsa_components=16)
    X = extractor.fit_transform(df_train)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values

    print(f"Feature Space Extracted: {X.shape[1]} features (107 total)")

    folds = list(get_stratified_group_folds(df_train, n_splits=5))
    oof_swiglu = np.zeros_like(y_onehot, dtype=np.float64)
    oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
    oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y_onehot[val_idx]

        # 1. SwiGLU ResNet GPU Model
        t_gpu = time.time()
        pt_model = PyTorchSwiGLUPreferenceModel(extractor.feature_names_, hidden_dim=128, epochs=20, lr=1e-3)
        pt_model.fit(X_tr, y_tr)
        p_pt = pt_model.predict_proba(X_va)
        oof_swiglu[val_idx] = p_pt
        loss_pt = compute_log_loss(y_va, p_pt)
        print(f"  PyTorch SwiGLU GPU ({time.time() - t_gpu:.1f}s) Log Loss: {loss_pt:.5f}")

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
    print(f"Raw PyTorch SwiGLU OOF Log Loss: {compute_log_loss(y_onehot, oof_swiglu):.5f}")
    print(f"Raw Logistic Regression OOF Log Loss: {compute_log_loss(y_onehot, oof_lr):.5f}")
    print(f"Raw HistGBDT OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt):.5f}")

    # Test Temperature vs Dirichlet Calibrator on Champion Blend (0.45 LR + 0.40 GBDT + 0.15 GPU)
    blend = 0.45 * oof_lr + 0.40 * oof_gbdt + 0.15 * oof_swiglu
    blend /= blend.sum(axis=1, keepdims=True)
    raw_loss = compute_log_loss(y_onehot, blend)
    print(f"\nTri-Ensemble Raw Log Loss: {raw_loss:.5f}")

    cal_temp = TemperatureCalibrator().fit(blend, y_onehot)
    temp_loss = compute_log_loss(y_onehot, cal_temp.transform(blend))
    print(f"Standard Temperature Calibrated Log Loss: {temp_loss:.5f} (T={cal_temp.temperature:.4f}, Tie={cal_temp.tie_scale:.4f})")

    cal_dirichlet = DirichletSymmetricCalibrator().fit(blend, y_onehot)
    dirichlet_loss = compute_log_loss(y_onehot, cal_dirichlet.transform(blend))
    print(f"Decoupled Dirichlet Calibrated Log Loss: {dirichlet_loss:.5f} (T_pair={cal_dirichlet.t_pair:.4f}, Beta_tie={cal_dirichlet.beta_tie:.4f}, Alpha_tie={cal_dirichlet.alpha_tie:.4f})")

    # Weight Tuning with Dirichlet Calibrator
    best_loss = 999.0
    best_weights: Optional[Tuple[float, float, float]] = None
    for w_pt in [0.05, 0.10, 0.15, 0.20]:
        for w_lr in [0.40, 0.45, 0.50, 0.55]:
            w_gb = 1.0 - w_pt - w_lr
            if w_gb < 0:
                continue
            b = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * oof_swiglu
            b /= b.sum(axis=1, keepdims=True)
            cal = DirichletSymmetricCalibrator().fit(b, y_onehot)
            l = compute_log_loss(y_onehot, cal.transform(b))
            if l < best_loss:
                best_loss = l
                best_weights = (float(w_lr), float(w_gb), float(w_pt))

    assert best_weights is not None
    print(f"\nTuned Weights with Dirichlet Calibrator: LR={best_weights[0]:.2f}, GBDT={best_weights[1]:.2f}, SwiGLU_GPU={best_weights[2]:.2f}")
    print(f"Optimal Dirichlet Calibrated Log Loss: {best_loss:.5f}")

    delta = best_loss - CHAMPION_10K_BENCHMARK
    print(f"\nDelta vs Champion ({CHAMPION_10K_BENCHMARK:.5f}): {delta:+.5f}")
    if delta < 0:
        print("GATE STATUS: PASSED (Strict Monotonic Improvement)")
    else:
        print("GATE STATUS: FAILED (No Improvement)")
