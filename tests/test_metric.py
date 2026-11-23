import numpy as np
import pytest
try:
    from src.metric import compute_log_loss, evaluate_breakdown
except (ImportError, AttributeError):
    from metric import compute_log_loss, evaluate_breakdown  # type: ignore


def test_perfect_predictions():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    loss = compute_log_loss(y_true, y_pred)
    assert loss < 1e-10, f"Expected loss close to 0, got {loss}"


def test_uniform_baseline():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    y_pred = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)
    loss = compute_log_loss(y_true, y_pred)
    expected = -np.log(1.0 / 3.0)
    assert np.isclose(loss, expected, atol=1e-5), f"Expected {expected}, got {loss}"


def test_eps_clipping_handles_zeros():
    y_true = np.array([[1, 0, 0]], dtype=np.float64)
    y_pred = np.array([[0.0, 0.5, 0.5]], dtype=np.float64)
    loss = compute_log_loss(y_true, y_pred)
    assert np.isfinite(loss), "Log loss should be finite under eps clipping"
    assert loss > 30.0, "Extreme error should yield high log loss"


def test_evaluate_breakdown():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    y_pred = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)
    breakdown = evaluate_breakdown(y_true, y_pred)
    assert "overall_log_loss" in breakdown
    assert "winner_model_a_loss" in breakdown
    assert "winner_model_b_loss" in breakdown
    assert "winner_tie_loss" in breakdown
