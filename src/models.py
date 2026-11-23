from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .features import FeatureExtractor
except ImportError:
    from features import FeatureExtractor  # type: ignore


class SymmetricModelWrapper:
    """
    Enforces anti-symmetric invariance via pair-swap training augmentation
    and test-time symmetrization.
    """

    def __init__(self, base_estimator, feature_names: List[str]):
        self.base_estimator = base_estimator
        self.feature_names = feature_names
        self.scaler = StandardScaler()

    def _swap_features(self, X: np.ndarray) -> np.ndarray:
        """
        Negates anti-symmetric difference features while keeping symmetric features unchanged.
        """
        X_swapped = X.copy()
        for idx, name in enumerate(self.feature_names):
            if name.startswith("diff_"):
                X_swapped[:, idx] = -X_swapped[:, idx]
        return X_swapped

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymmetricModelWrapper":
        """
        Trains with mirrored anti-symmetric pair augmentation.
        Label 0 -> 1, Label 1 -> 0, Label 2 -> 2.
        """
        X_swap = self._swap_features(X)
        y_swap = y.copy()
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        y_swap[mask_0] = 1
        y_swap[mask_1] = 0

        # Concatenate original and swapped samples
        X_aug = np.vstack([X, X_swap])
        y_aug = np.concatenate([y, y_swap])

        # Scale features
        X_scaled = self.scaler.fit_transform(X_aug)

        # Fit underlying estimator
        self.base_estimator.fit(X_scaled, y_aug)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities with test-time symmetrization.
        """
        X_scaled_orig = self.scaler.transform(X)
        p_orig = self.base_estimator.predict_proba(X_scaled_orig)

        X_swap = self._swap_features(X)
        X_scaled_swap = self.scaler.transform(X_swap)
        p_swap = self.base_estimator.predict_proba(X_scaled_swap)

        # Invert predictions from swapped order: swap col 0 and 1
        p_swap_inverted = np.zeros_like(p_swap)
        p_swap_inverted[:, 0] = p_swap[:, 1]
        p_swap_inverted[:, 1] = p_swap[:, 0]
        p_swap_inverted[:, 2] = p_swap[:, 2]

        # Average original and swapped predictions
        p_sym = 0.5 * (p_orig + p_swap_inverted)

        # Normalize simplex
        p_sym = p_sym / p_sym.sum(axis=1, keepdims=True)
        return p_sym


class EnsemblePreferenceModel:
    """
    Symmetric blend of regularized Logistic Regression and HistGBDT.
    """

    def __init__(self, feature_names: List[str], lr_weight: float = 0.44):
        self.feature_names = feature_names
        self.lr_weight = lr_weight
        self.lr_model = build_logistic_model(feature_names, C=0.5)
        self.gbdt_model = build_hist_gbdt_model(feature_names, max_iter=150, learning_rate=0.05)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsemblePreferenceModel":
        self.lr_model.fit(X, y)
        self.gbdt_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_lr = self.lr_model.predict_proba(X)
        p_gbdt = self.gbdt_model.predict_proba(X)
        p_blend = self.lr_weight * p_lr + (1.0 - self.lr_weight) * p_gbdt
        return p_blend / p_blend.sum(axis=1, keepdims=True)


def get_hist_gbdt_loss() -> str:
    """
    Dynamically detects supported loss parameter across scikit-learn versions
    ('log_loss' in sklearn >= 1.1; 'auto' in sklearn <= 1.0).
    """
    try:
        import sklearn
        parts = [int(x) for x in sklearn.__version__.split(".") if x.isdigit()]
        if parts and (parts[0] > 1 or (parts[0] == 1 and len(parts) > 1 and parts[1] >= 1)):
            return "log_loss"
    except Exception:
        pass
    return "auto"


def build_logistic_model(feature_names: List[str], C: float = 1.0) -> SymmetricModelWrapper:
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    return SymmetricModelWrapper(clf, feature_names)


def build_hist_gbdt_model(
    feature_names: List[str],
    max_iter: int = 150,
    learning_rate: float = 0.04,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 45,
) -> SymmetricModelWrapper:
    clf = HistGradientBoostingClassifier(
        loss=get_hist_gbdt_loss(),
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    return SymmetricModelWrapper(clf, feature_names)


def build_ensemble_model(feature_names: List[str], lr_weight: float = 0.44) -> EnsemblePreferenceModel:
    return EnsemblePreferenceModel(feature_names, lr_weight=lr_weight)


if TORCH_AVAILABLE:
    class SymmetricMLP(nn.Module):
        """
        PyTorch Neural Preference Network with LayerNorm, SiLU activations,
        and guaranteed exact anti-symmetric invariance.
        """
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
        """
        Anti-symmetric PyTorch preference model running on NVIDIA GPU with automatic fallback.
        """
        def __init__(
            self,
            feature_names: List[str],
            hidden_dim: int = 128,
            lr: float = 2e-3,
            epochs: int = 25,
            batch_size: int = 256,
            temperature: float = 2.8,
            device: Optional[Union[str, torch.device]] = None,
        ):
            self.feature_names = feature_names
            self.hidden_dim = hidden_dim
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.temperature = float(temperature)

            if device is None:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)

            self.diff_indices = [i for i, name in enumerate(feature_names) if name.startswith("diff_")]
            self.mean: Optional[torch.Tensor] = None
            self.std: Optional[torch.Tensor] = None
            self.model: Optional[SymmetricMLP] = None

        def _swap_features_tensor(self, X: torch.Tensor) -> torch.Tensor:
            X_swap = X.clone()
            X_swap[:, self.diff_indices] = -X_swap[:, self.diff_indices]
            return X_swap

        def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PyTorchGPUPreferenceModel":
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
                raise ValueError("Model must be fitted before calling predict_proba")
            self.model.eval()
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            X_norm = (X_t - self.mean) / self.std
            X_swap_norm = self._swap_features_tensor(X_norm)

            with torch.no_grad():
                logits = self.model(X_norm, X_swap_norm)
                scaled_logits = logits / max(self.temperature, 0.1)
                probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()

            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs


    class TriEnsemblePreferenceModel:
        """
        Three-way blend combining:
        1. Linear Regularized Logistic Regression (0.315)
        2. Non-linear HistGBDT (0.460)
        3. Deep Anti-Symmetric Neural Network on NVIDIA GPU (0.225)
        """
        def __init__(
            self,
            feature_names: List[str],
            lr_weight: float = 0.315,
            gbdt_weight: float = 0.460,
            gpu_pt_weight: float = 0.225,
            device: Optional[Union[str, torch.device]] = None,
        ):
            self.feature_names = feature_names
            self.lr_weight = lr_weight
            self.gbdt_weight = gbdt_weight
            self.gpu_pt_weight = gpu_pt_weight

            self.lr_model = build_logistic_model(feature_names, C=0.5)
            self.gbdt_model = build_hist_gbdt_model(feature_names, max_iter=150, learning_rate=0.04)
            self.pt_model = PyTorchGPUPreferenceModel(feature_names, hidden_dim=128, epochs=25, temperature=2.8, device=device)

        def fit(self, X: np.ndarray, y: np.ndarray) -> "TriEnsemblePreferenceModel":
            self.lr_model.fit(X, y)
            self.gbdt_model.fit(X, y)
            self.pt_model.fit(X, y)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            p_lr = self.lr_model.predict_proba(X)
            p_gbdt = self.gbdt_model.predict_proba(X)
            p_pt = self.pt_model.predict_proba(X)

            p_blend = (
                self.lr_weight * p_lr
                + self.gbdt_weight * p_gbdt
                + self.gpu_pt_weight * p_pt
            )
            return p_blend / p_blend.sum(axis=1, keepdims=True)


    def build_pytorch_gpu_model(feature_names: List[str], device: Optional[str] = None) -> PyTorchGPUPreferenceModel:
        return PyTorchGPUPreferenceModel(feature_names, device=device)


    def build_tri_ensemble_model(
        feature_names: List[str],
        lr_weight: float = 0.315,
        gbdt_weight: float = 0.460,
        gpu_pt_weight: float = 0.225,
        device: Optional[str] = None,
    ) -> TriEnsemblePreferenceModel:
        return TriEnsemblePreferenceModel(
            feature_names,
            lr_weight=lr_weight,
            gbdt_weight=gbdt_weight,
            gpu_pt_weight=gpu_pt_weight,
            device=device,
        )

