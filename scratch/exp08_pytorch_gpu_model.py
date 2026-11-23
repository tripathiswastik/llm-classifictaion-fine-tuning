import os
import sys
import time
import math
from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure src is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor
from metric import compute_log_loss
from calibration import TemperatureCalibrator

# 1. Device check
assert torch.cuda.is_available(), "CUDA is not available! Must run on NVIDIA GPU."
device = torch.device("cuda:0")
print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
print(f"Allocated memory before start: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")

# 2. PyTorch Symmetric Neural Preference Network
class SymmetricMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.25):
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
        """
        Anti-symmetrized forward pass:
        Averages forward(x) with target-swapped forward(x_swap).
        Guarantees exact anti-symmetry by construction.
        """
        logits_orig = self.forward_single(x)
        logits_swap = self.forward_single(x_swap)

        # Invert target positions for swapped features: swap col 0 and 1
        logits_swap_inv = torch.stack([logits_swap[:, 1], logits_swap[:, 0], logits_swap[:, 2]], dim=1)

        # Enforce exact symmetry
        sym_logits = 0.5 * (logits_orig + logits_swap_inv)
        return sym_logits


class PyTorchGPUPreferenceModel:
    def __init__(
        self,
        feature_names: list,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 256,
        device: torch.device = device,
    ):
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model: Optional[SymmetricMLP] = None
        self.diff_indices = [i for i, name in enumerate(feature_names) if name.startswith("diff_")]

        # Mean and std for scaling on GPU
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def _swap_features_tensor(self, X: torch.Tensor) -> torch.Tensor:
        X_swap = X.clone()
        X_swap[:, self.diff_indices] = -X_swap[:, self.diff_indices]
        return X_swap

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        # Standardize features using symmetric augmentation so diff features have strictly 0 mean
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_x_swap, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x, batch_x_swap)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)
            scheduler.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.mean is None or self.std is None:
            raise ValueError("Model must be fitted before calling predict_proba")
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_norm = (X_t - self.mean) / self.std
        X_swap_norm = self._swap_features_tensor(X_norm)

        with torch.no_grad():
            logits = self.model(X_norm, X_swap_norm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Enforce simplex bounds
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


# 3. Quick Verification on 5,000 battles
if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "data", "train.csv")
    test_csv = os.path.join(BASE_DIR, "data", "test.csv")

    print(f"Loading data from {train_csv}...")
    df_train = load_dataset(train_csv, nrows=5000, is_test=False)
    print(f"Loaded {len(df_train)} rows.")

    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X = extractor.fit_transform(df_train)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values

    print(f"Features: {X.shape} ({len(extractor.feature_names_)} columns)")

    # 3-Fold quick CV on GPU
    folds = list(get_stratified_group_folds(df_train, n_splits=3))
    oof_preds = np.zeros_like(y_onehot, dtype=np.float64)
    model: Optional[PyTorchGPUPreferenceModel] = None

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Training Fold {fold+1}/3 on NVIDIA GPU ({torch.cuda.get_device_name(0)})...")
        model = PyTorchGPUPreferenceModel(
            extractor.feature_names_,
            hidden_dim=128,
            lr=2e-3,
            epochs=20,
            batch_size=256,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[val_idx])
        oof_preds[val_idx] = preds
        loss = compute_log_loss(y_onehot[val_idx], preds)
        print(f"  Fold {fold+1} Log Loss: {loss:.5f}")

    total_time = time.time() - t0
    raw_loss = compute_log_loss(y_onehot, oof_preds)
    print(f"\nGPU Model Raw OOF Log Loss: {raw_loss:.5f} (Elapsed: {total_time:.2f}s)")

    # Calibration
    cal = TemperatureCalibrator()
    cal.fit(oof_preds, y_onehot)
    cal_oof = cal.transform(oof_preds)
    cal_loss = compute_log_loss(y_onehot, cal_oof)
    print(f"GPU Model Calibrated OOF Log Loss: {cal_loss:.5f} (T={cal.temperature:.4f}, TieScale={cal.tie_scale:.4f})")

    # Anti-symmetry verification on GPU
    assert model is not None, "Model was not trained"
    print("\nVerifying Anti-Symmetry on GPU...")
    X_sample = X[:100]
    p_orig = model.predict_proba(X_sample)
    diff_indices = [i for i, n in enumerate(extractor.feature_names_) if n.startswith("diff_")]
    X_swap = X_sample.copy()
    X_swap[:, diff_indices] = -X_swap[:, diff_indices]
    p_swapped = model.predict_proba(X_swap)

    assert np.allclose(p_orig[:, 0], p_swapped[:, 1], atol=1e-5), "Anti-symmetry failed for A <-> B"
    assert np.allclose(p_orig[:, 1], p_swapped[:, 0], atol=1e-5), "Anti-symmetry failed for B <-> A"
    assert np.allclose(p_orig[:, 2], p_swapped[:, 2], atol=1e-5), "Symmetry failed for tie"
    print("Anti-Symmetry Verification: PASSED (Exact equality within 1e-5 tolerance)")

    # Inference on test set
    df_test = load_dataset(test_csv, is_test=True)
    X_test = extractor.transform(df_test)
    test_preds = model.predict_proba(X_test)
    cal_test_preds = cal.transform(test_preds)

    sub = pd.DataFrame({
        "id": df_test["id"].values,
        "winner_model_a": cal_test_preds[:, 0],
        "winner_model_b": cal_test_preds[:, 1],
        "winner_tie": cal_test_preds[:, 2],
    })
    print("\nGPU Model Test Predictions:")
    print(sub)
    print(f"Max VRAM Allocated on GPU: {torch.cuda.max_memory_allocated(0)/(1024**2):.2f} MB")
