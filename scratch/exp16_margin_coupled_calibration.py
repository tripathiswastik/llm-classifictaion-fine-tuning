import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
for p in (BASE_DIR, SRC_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor
from metric import compute_log_loss
from calibration import TemperatureCalibrator, DirichletSymmetricCalibrator
from models import build_logistic_model, build_hist_gbdt_model

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")

CHAMPION_10K_BENCHMARK = 1.01601


class MarginCoupledDirichletCalibrator:
    """
    4-Parameter Margin-Coupled Symmetric Calibrator.
    Guarantees exact mathematical anti-symmetry and simplex constraints.
    Couples tie logit to contest decisiveness:
      z_AB = (log(p_A) - log(p_B)) / T_pair
      z_tie = beta_tie * (log(p_tie) - 0.5*(log(p_A) + log(p_B))) + alpha_tie - gamma_tie * (|z_AB| / (1 + |z_AB|))
    """
    def __init__(self, t_pair: float = 1.0, beta_tie: float = 1.0, alpha_tie: float = 0.0, gamma_tie: float = 0.0):
        self.t_pair = float(t_pair)
        self.beta_tie = float(beta_tie)
        self.alpha_tie = float(alpha_tie)
        self.gamma_tie = float(gamma_tie)

    def fit(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> "MarginCoupledDirichletCalibrator":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab = log_a - log_b
        z_tie_base = log_tie - 0.5 * (log_a + log_b)

        def objective(params: np.ndarray) -> float:
            t_p, b_t, a_t, g_t = params
            z_ab_scaled = z_ab / max(t_p, 1e-3)
            # Smooth bounded contest decisiveness
            margin = np.abs(z_ab_scaled) / (1.0 + np.abs(z_ab_scaled))
            z_tie_scaled = b_t * z_tie_base + a_t - g_t * margin

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
            x0=[self.t_pair, self.beta_tie, self.alpha_tie, self.gamma_tie],
            bounds=[(0.2, 5.0), (0.2, 5.0), (-3.0, 3.0), (-3.0, 5.0)],
            method="L-BFGS-B",
        )
        self.t_pair = float(res.x[0])
        self.beta_tie = float(res.x[1])
        self.alpha_tie = float(res.x[2])
        self.gamma_tie = float(res.x[3])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab_scaled = (log_a - log_b) / max(self.t_pair, 1e-3)
        margin = np.abs(z_ab_scaled) / (1.0 + np.abs(z_ab_scaled))
        z_tie_scaled = self.beta_tie * (log_tie - 0.5 * (log_a + log_b)) + self.alpha_tie - self.gamma_tie * margin

        l_a = 0.5 * z_ab_scaled
        l_b = -0.5 * z_ab_scaled
        l_tie = z_tie_scaled

        l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
        exp_a = np.exp(l_a[:, None] - l_max)
        exp_b = np.exp(l_b[:, None] - l_max)
        exp_tie = np.exp(l_tie[:, None] - l_max)

        denom = exp_a + exp_b + exp_tie
        return np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])


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
    def __init__(self, feature_names: list, hidden_dim: int = 128, lr: float = 2e-3, epochs: int = 20, batch_size: int = 512):
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
    print("EXP 16: MARGIN-COUPLED DIRICHLET CALIBRATOR ON 158 FEATURES")
    print(f"Champion 10k Benchmark to beat: {CHAMPION_10K_BENCHMARK:.5f}")
    print("================================================================")

    df_train = load_dataset(train_csv, nrows=10000, is_test=False)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values
    folds = list(get_stratified_group_folds(df_train, n_splits=5))

    extractor = FeatureExtractor(
        use_tfidf=True,
        max_tfidf_features=15000,
        use_lsa=True,
        n_lsa_components=24,
        use_char_lsa=True,
        n_char_lsa_components=16,
    )
    t0_feat = time.time()
    X = extractor.fit_transform(df_train)
    print(f"Feature Extraction Time: {time.time() - t0_feat:.1f}s", flush=True)
    print(f"Total Feature Space: {X.shape[1]} features", flush=True)

    oof_pytorch = np.zeros_like(y_onehot, dtype=np.float64)
    oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
    oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y_onehot[val_idx]

        # 1. PyTorch GPU Model
        pt_model = PyTorchGPUPreferenceModel(extractor.feature_names_, hidden_dim=128, epochs=20, batch_size=512)
        pt_model.fit(X_tr, y_tr)
        oof_pytorch[val_idx] = pt_model.predict_proba(X_va)

        # 2. Logistic Regression (C=0.5)
        lr_model = build_logistic_model(extractor.feature_names_, C=0.5)
        lr_model.fit(X_tr, y_tr)
        oof_lr[val_idx] = lr_model.predict_proba(X_va)

        # 3. HistGBDT (max_leaf_nodes=31, min_samples_leaf=50)
        gbdt_model = build_hist_gbdt_model(extractor.feature_names_, max_leaf_nodes=31, min_samples_leaf=50)
        gbdt_model.fit(X_tr, y_tr)
        oof_gbdt[val_idx] = gbdt_model.predict_proba(X_va)

        print(f"  Fold {fold+1}/5 done ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n5-Fold Training Time: {time.time() - t0:.1f}s", flush=True)
    print(f"  PyTorch GPU OOF Log Loss: {compute_log_loss(y_onehot, oof_pytorch):.5f}", flush=True)
    print(f"  Logistic Regression OOF Log Loss: {compute_log_loss(y_onehot, oof_lr):.5f}", flush=True)
    print(f"  HistGBDT OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt):.5f}", flush=True)

    # Save OOF predictions for fast iterative calibration experiments
    cache_path = os.path.join(BASE_DIR, "scratch", "oof_exp16.npz")
    np.savez_compressed(cache_path, lr=oof_lr, gbdt=oof_gbdt, pt=oof_pytorch, y=y_onehot)
    print(f"Cached OOF arrays to {cache_path}", flush=True)

    # Standard Champion Blend (0.35 LR + 0.50 GBDT + 0.15 GPU)
    blend_v7 = 0.35 * oof_lr + 0.50 * oof_gbdt + 0.15 * oof_pytorch
    blend_v7 /= blend_v7.sum(axis=1, keepdims=True)
    raw_v7_loss = compute_log_loss(y_onehot, blend_v7)
    print(f"\nVersion 7 Raw Blend Log Loss: {raw_v7_loss:.5f}", flush=True)

    # 1. 3-Parameter Standard Dirichlet Calibrator (V7 Baseline)
    cal_dir3 = DirichletSymmetricCalibrator().fit(blend_v7, y_onehot)
    loss_dir3 = compute_log_loss(y_onehot, cal_dir3.transform(blend_v7))
    print(f"1. 3-Parameter Dirichlet Log Loss: {loss_dir3:.5f} (T_pair={cal_dir3.t_pair:.4f}, Beta_tie={cal_dir3.beta_tie:.4f}, Alpha_tie={cal_dir3.alpha_tie:.4f})", flush=True)

    # 2. 4-Parameter Margin-Coupled Calibrator
    cal_margin = MarginCoupledDirichletCalibrator().fit(blend_v7, y_onehot)
    loss_margin = compute_log_loss(y_onehot, cal_margin.transform(blend_v7))
    print(f"2. 4-Parameter Margin-Coupled Log Loss: {loss_margin:.5f} (T_pair={cal_margin.t_pair:.4f}, Beta_tie={cal_margin.beta_tie:.4f}, Alpha_tie={cal_margin.alpha_tie:.4f}, Gamma_tie={cal_margin.gamma_tie:.4f})", flush=True)

    # 3. Fine-grained Blend Grid with Margin-Coupled Calibrator
    best_loss = 999.0
    best_w: Optional[Tuple[float, float, float]] = None
    best_cal: Optional[MarginCoupledDirichletCalibrator] = None

    for w_pt in [0.05, 0.10, 0.15, 0.20]:
        for w_lr in [0.25, 0.30, 0.35, 0.40]:
            w_gb = 1.0 - w_pt - w_lr
            if w_gb < 0:
                continue
            b = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * oof_pytorch
            b /= b.sum(axis=1, keepdims=True)
            c = MarginCoupledDirichletCalibrator().fit(b, y_onehot)
            l = compute_log_loss(y_onehot, c.transform(b))
            if l < best_loss:
                best_loss = l
                best_w = (float(w_lr), float(w_gb), float(w_pt))
                best_cal = c

    assert best_w is not None and best_cal is not None
    print(f"\n3. Tuned Blend Margin-Coupled Log Loss: {best_loss:.5f} (Weights: LR={best_w[0]:.2f}, GBDT={best_w[1]:.2f}, GPU={best_w[2]:.2f})", flush=True)
    print(f"   Optimal Calibrator: T_pair={best_cal.t_pair:.4f}, Beta_tie={best_cal.beta_tie:.4f}, Alpha_tie={best_cal.alpha_tie:.4f}, Gamma_tie={best_cal.gamma_tie:.4f}", flush=True)

    delta = best_loss - CHAMPION_10K_BENCHMARK
    print(f"\nDelta vs Champion ({CHAMPION_10K_BENCHMARK:.5f}): {delta:+.5f}", flush=True)
    if delta < 0:
        print("GATE STATUS: PASSED (Strict Monotonic Improvement)", flush=True)
    else:
        print("GATE STATUS: FAILED (No Improvement)", flush=True)
