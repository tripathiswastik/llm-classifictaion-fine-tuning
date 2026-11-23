import os
import sys
import time
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

from typing import List, Optional, Tuple, Union

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor
from metric import compute_log_loss
from calibration import TemperatureCalibrator
from models import build_logistic_model, build_hist_gbdt_model

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

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
    test_csv = os.path.join(BASE_DIR, "data", "test.csv")

    print(f"Loading 10,000 battles for GPU benchmarking...")
    df_train = load_dataset(train_csv, nrows=10000, is_test=False)

    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X = extractor.fit_transform(df_train)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values

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

    # Out-of-fold evaluations
    print(f"\nTotal 5-Fold Time: {time.time() - t0:.1f}s")
    print(f"Raw PyTorch GPU OOF Log Loss: {compute_log_loss(y_onehot, oof_pytorch):.5f}")
    print(f"Raw Logistic Regression OOF Log Loss: {compute_log_loss(y_onehot, oof_lr):.5f}")
    print(f"Raw HistGBDT OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt):.5f}")

    # Standard Champion Blend (0.44 LR + 0.56 GBDT)
    champ_blend = 0.44 * oof_lr + 0.56 * oof_gbdt
    champ_blend /= champ_blend.sum(axis=1, keepdims=True)
    cal_champ = TemperatureCalibrator().fit(champ_blend, y_onehot)
    print(f"Champion (LR + GBDT) Calibrated Log Loss: {compute_log_loss(y_onehot, cal_champ.transform(champ_blend)):.5f}")

    # Tri-Ensemble Grid Search: w_lr * LR + w_gbdt * GBDT + w_pt * PyTorch_GPU
    best_loss = 999.0
    best_weights: Optional[Tuple[float, float, float]] = None
    for w_pt in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for w_lr in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
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
    print(f"\nOptimal Tri-Ensemble Weights: LR={best_weights[0]:.2f}, GBDT={best_weights[1]:.2f}, GPU_PT={best_weights[2]:.2f}")
    opt_blend = best_weights[0] * oof_lr + best_weights[1] * oof_gbdt + best_weights[2] * oof_pytorch
    opt_blend /= opt_blend.sum(axis=1, keepdims=True)
    cal_tri = TemperatureCalibrator().fit(opt_blend, y_onehot)
    cal_tri_loss = compute_log_loss(y_onehot, cal_tri.transform(opt_blend))
    print(f"Raw Tri-Ensemble OOF Log Loss: {best_loss:.5f}")
    print(f"Calibrated Tri-Ensemble OOF Log Loss: {cal_tri_loss:.5f}")
