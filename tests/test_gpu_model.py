import numpy as np
import pytest
import torch

from src.models import (
    TORCH_AVAILABLE,
    build_pytorch_gpu_model,
    build_tri_ensemble_model,
)

feature_names = [
    "diff_char_len",
    "diff_word_count",
    "diff_code_blocks",
    "sym_abs_char_diff",
    "sym_total_char_len",
]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for GPU model")
def test_pytorch_gpu_model_symmetry_and_simplex():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_pytorch_gpu_model(feature_names, device=device)

    # Synthetic training data
    np.random.seed(42)
    X_train = np.random.randn(200, len(feature_names)).astype(np.float32)
    y_train = np.random.choice([0, 1, 2], size=200)

    model.fit(X_train, y_train)

    # Test sample
    X_test = np.random.randn(20, len(feature_names)).astype(np.float32)
    p_orig = model.predict_proba(X_test)

    # Simplex assertion
    assert np.allclose(p_orig.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1.0"
    assert (p_orig >= 0.0).all(), "Probabilities must be non-negative"

    # Swapped test
    diff_indices = [i for i, n in enumerate(feature_names) if n.startswith("diff_")]
    X_swap = X_test.copy()
    X_swap[:, diff_indices] = -X_swap[:, diff_indices]
    p_swapped = model.predict_proba(X_swap)

    # Anti-symmetry assertion
    np.testing.assert_allclose(p_orig[:, 0], p_swapped[:, 1], atol=1e-5, err_msg="P(A) != P(B_swap)")
    np.testing.assert_allclose(p_orig[:, 1], p_swapped[:, 0], atol=1e-5, err_msg="P(B) != P(A_swap)")
    np.testing.assert_allclose(p_orig[:, 2], p_swapped[:, 2], atol=1e-5, err_msg="P(tie) != P(tie_swap)")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for Tri-Ensemble model")
def test_tri_ensemble_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = build_tri_ensemble_model(feature_names, device=device)

    np.random.seed(42)
    X_train = np.random.randn(150, len(feature_names)).astype(np.float32)
    y_train = np.random.choice([0, 1, 2], size=150)

    model.fit(X_train, y_train)

    X_test = np.random.randn(10, len(feature_names)).astype(np.float32)
    p = model.predict_proba(X_test)

    assert p.shape == (10, 3)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
