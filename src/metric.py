import numpy as np
from typing import Dict, Union


def compute_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Computes multi-class log loss strictly matching Kaggle's evaluation with eps=auto.

    Args:
        y_true: Binary indicator matrix of shape (N, 3), where columns are
                [winner_model_a, winner_model_b, winner_tie].
        y_pred: Predicted probability matrix of shape (N, 3).
        eps: Epsilon clipping threshold (1e-15).

    Returns:
        float: Multi-class log loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Normalize predictions across row simplex
    row_sums = y_pred.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    y_norm = y_pred / row_sums

    # Clip to avoid log(0)
    y_clipped = np.clip(y_norm, eps, 1.0 - eps)

    # Re-normalize after clipping
    y_clipped = y_clipped / y_clipped.sum(axis=1, keepdims=True)

    # Multi-class log loss: -1/N * sum(y_true * log(y_pred))
    loss_matrix = y_true * np.log(y_clipped)
    return float(-np.mean(np.sum(loss_matrix, axis=1)))


def evaluate_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple = ("winner_model_a", "winner_model_b", "winner_tie")
) -> Dict[str, float]:
    """
    Computes overall log loss and per-class loss decomposition.
    """
    overall = compute_log_loss(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = y_pred / np.maximum(y_pred.sum(axis=1, keepdims=True), 1e-15)
    y_clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

    breakdown = {"overall_log_loss": overall}
    for idx, name in enumerate(class_names):
        mask = (y_true[:, idx] == 1)
        if mask.any():
            class_loss = float(-np.mean(np.log(y_clipped[mask, idx])))
            breakdown[f"{name}_loss"] = class_loss

    return breakdown
