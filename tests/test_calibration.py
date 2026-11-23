import os
import sys
import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration import TemperatureCalibrator, DirichletSymmetricCalibrator
from metric import compute_log_loss


def test_temperature_calibrator_anti_symmetry():
    calibrator = TemperatureCalibrator(temperature=0.85, tie_scale=1.15)

    # Probabilities for battle 1: [0.6, 0.2, 0.2]
    # Inverted battle 1: [0.2, 0.6, 0.2]
    p_orig = np.array([[0.6, 0.2, 0.2], [0.35, 0.45, 0.20]])
    p_swap = np.array([[0.2, 0.6, 0.2], [0.45, 0.35, 0.20]])

    cal_orig = calibrator.transform(p_orig)
    cal_swap = calibrator.transform(p_swap)

    # Simplex sum invariant
    np.testing.assert_allclose(cal_orig.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(cal_swap.sum(axis=1), 1.0, atol=1e-5)

    # Swapped invariant
    np.testing.assert_allclose(cal_orig[:, 0], cal_swap[:, 1], atol=1e-5)
    np.testing.assert_allclose(cal_orig[:, 1], cal_swap[:, 0], atol=1e-5)
    np.testing.assert_allclose(cal_orig[:, 2], cal_swap[:, 2], atol=1e-5)


def test_temperature_calibrator_monotonic_loss():
    # Synthetic uncalibrated overconfident probabilities
    np.random.seed(42)
    n = 100
    y_true = np.zeros((n, 3))
    true_labels = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    for i, lbl in enumerate(true_labels):
        y_true[i, lbl] = 1.0

    # Overconfident noisy predictions
    probs = np.random.uniform(0.1, 0.9, size=(n, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)

    uncal_loss = compute_log_loss(y_true, probs)
    calibrator = TemperatureCalibrator()
    cal_probs = calibrator.fit_transform(probs, y_true)
    cal_loss = compute_log_loss(y_true, cal_probs)

    assert cal_loss <= uncal_loss + 1e-5, f"Calibration degraded loss: {cal_loss:.5f} > {uncal_loss:.5f}"


def test_dirichlet_calibrator_anti_symmetry():
    calibrator = DirichletSymmetricCalibrator(t_pair=0.92, beta_tie=0.85, alpha_tie=0.05)

    p_orig = np.array([[0.6, 0.2, 0.2], [0.35, 0.45, 0.20], [0.1, 0.7, 0.2]])
    p_swap = np.array([[0.2, 0.6, 0.2], [0.45, 0.35, 0.20], [0.7, 0.1, 0.2]])

    cal_orig = calibrator.transform(p_orig)
    cal_swap = calibrator.transform(p_swap)

    # Simplex sum invariant
    np.testing.assert_allclose(cal_orig.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(cal_swap.sum(axis=1), 1.0, atol=1e-5)

    # Swapped invariant: P(A) -> P(B), P(B) -> P(A), P(tie) -> P(tie)
    np.testing.assert_allclose(cal_orig[:, 0], cal_swap[:, 1], atol=1e-5)
    np.testing.assert_allclose(cal_orig[:, 1], cal_swap[:, 0], atol=1e-5)
    np.testing.assert_allclose(cal_orig[:, 2], cal_swap[:, 2], atol=1e-5)


def test_dirichlet_calibrator_monotonic_loss():
    np.random.seed(42)
    n = 200
    y_true = np.zeros((n, 3))
    true_labels = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    for i, lbl in enumerate(true_labels):
        y_true[i, lbl] = 1.0

    probs = np.random.uniform(0.1, 0.9, size=(n, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)

    uncal_loss = compute_log_loss(y_true, probs)
    calibrator = DirichletSymmetricCalibrator()
    cal_probs = calibrator.fit_transform(probs, y_true)
    cal_loss = compute_log_loss(y_true, cal_probs)

    assert cal_loss <= uncal_loss + 1e-5, f"Dirichlet calibration degraded loss: {cal_loss:.5f} > {uncal_loss:.5f}"


if __name__ == "__main__":
    test_temperature_calibrator_anti_symmetry()
    test_temperature_calibrator_monotonic_loss()
    test_dirichlet_calibrator_anti_symmetry()
    test_dirichlet_calibrator_monotonic_loss()
    print("ALL CALIBRATION TESTS PASSED (Exact Anti-Symmetry & Monotonic Improvement)!")
