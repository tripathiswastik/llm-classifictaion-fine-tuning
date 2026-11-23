import os
import sys
import time
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
    Preserves exact anti-symmetry and simplex constraints:
      z_AB = (log(p_A) - log(p_B)) / T_pair
      z_tie = beta_tie * (log(p_tie) - 0.5 * (log(p_A) + log(p_B))) + alpha_tie
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
            bounds=[(0.2, 5.0), (0.2, 5.0), (-3.0, 3.0)],
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
    print("EXP 13: LSA OPTIMIZATION & DECOUPLED DIRICHLET CALIBRATION")
    print(f"Champion 10k Benchmark to beat: {CHAMPION_10K_BENCHMARK:.5f}")
    print("================================================================")

    df_train = load_dataset(train_csv, nrows=10000, is_test=False)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values
    folds = list(get_stratified_group_folds(df_train, n_splits=5))

    for k_lsa in [16, 24]:
        print(f"\n========================================================")
        print(f"Testing LSA components k = {k_lsa}")
        print(f"========================================================")
        extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000, use_lsa=True, n_lsa_components=k_lsa)
        X = extractor.fit_transform(df_train)
        print(f"Feature Space Extracted: {X.shape[1]} features")

        oof_pytorch = np.zeros_like(y_onehot, dtype=np.float64)
        oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
        oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

        t0 = time.time()
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y_onehot[val_idx]

            # 1. PyTorch GPU Model
            pt_model = PyTorchGPUPreferenceModel(extractor.feature_names_, hidden_dim=128, epochs=25)
            pt_model.fit(X_tr, y_tr)
            oof_pytorch[val_idx] = pt_model.predict_proba(X_va)

            # 2. Logistic Regression
            lr_model = build_logistic_model(extractor.feature_names_, C=0.5)
            lr_model.fit(X_tr, y_tr)
            oof_lr[val_idx] = lr_model.predict_proba(X_va)

            # 3. HistGBDT
            gbdt_model = build_hist_gbdt_model(extractor.feature_names_)
            gbdt_model.fit(X_tr, y_tr)
            oof_gbdt[val_idx] = gbdt_model.predict_proba(X_va)

        print(f"5-Fold Training Time: {time.time() - t0:.1f}s")
        print(f"  PyTorch GPU OOF Log Loss: {compute_log_loss(y_onehot, oof_pytorch):.5f}")
        print(f"  Logistic Regression OOF Log Loss: {compute_log_loss(y_onehot, oof_lr):.5f}")
        print(f"  HistGBDT OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt):.5f}")

        # Standard Champion Blend (0.45 LR + 0.40 GBDT + 0.15 GPU)
        blend = 0.45 * oof_lr + 0.40 * oof_gbdt + 0.15 * oof_pytorch
        blend /= blend.sum(axis=1, keepdims=True)
        raw_loss = compute_log_loss(y_onehot, blend)

        # Standard Temperature Calibration
        cal_temp = TemperatureCalibrator().fit(blend, y_onehot)
        temp_loss = compute_log_loss(y_onehot, cal_temp.transform(blend))
        print(f"  Fixed Blend Raw Log Loss: {raw_loss:.5f}")
        print(f"  Temperature Calibrated: {temp_loss:.5f} (T={cal_temp.temperature:.4f}, Tie={cal_temp.tie_scale:.4f})")

        # Decoupled Dirichlet Calibration
        cal_dir = DirichletSymmetricCalibrator().fit(blend, y_onehot)
        dir_loss = compute_log_loss(y_onehot, cal_dir.transform(blend))
        print(f"  Dirichlet Calibrated: {dir_loss:.5f} (T_pair={cal_dir.t_pair:.4f}, Beta_tie={cal_dir.beta_tie:.4f}, Alpha_tie={cal_dir.alpha_tie:.4f})")

        # Tuned weights with Dirichlet Calibrator
        best_loss = 999.0
        best_w = None
        for w_pt in [0.05, 0.10, 0.15, 0.20]:
            for w_lr in [0.40, 0.45, 0.50, 0.55]:
                w_gb = 1.0 - w_pt - w_lr
                if w_gb < 0:
                    continue
                b = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * oof_pytorch
                b /= b.sum(axis=1, keepdims=True)
                c = DirichletSymmetricCalibrator().fit(b, y_onehot)
                l = compute_log_loss(y_onehot, c.transform(b))
                if l < best_loss:
                    best_loss = l
                    best_w = (float(w_lr), float(w_gb), float(w_pt))

        assert best_w is not None
        print(f"  Tuned Blend Dirichlet Log Loss: {best_loss:.5f} (Weights: LR={best_w[0]:.2f}, GBDT={best_w[1]:.2f}, GPU={best_w[2]:.2f})")
        delta = best_loss - CHAMPION_10K_BENCHMARK
        print(f"  Delta vs Champion ({CHAMPION_10K_BENCHMARK:.5f}): {delta:+.5f}")
        if delta < 0:
            print("  GATE STATUS: PASSED (Strict Monotonic Improvement)")
        else:
            print("  GATE STATUS: FAILED (No Improvement)")
