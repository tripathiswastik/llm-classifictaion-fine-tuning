import numpy as np
import pandas as pd
import pytest


def test_submission_schema_and_simplex():
    # Mock submission
    sub = pd.DataFrame({
        "id": [136060, 211333, 1233961],
        "winner_model_a": [0.45, 0.25, 0.33],
        "winner_model_b": [0.25, 0.45, 0.33],
        "winner_tie": [0.30, 0.30, 0.34],
    })

    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert list(sub.columns) == expected_cols, f"Columns mismatch: {list(sub.columns)}"
    assert not sub.isnull().any().any(), "Submission contains nulls"

    sums = sub[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-5, err_msg="Probabilities do not sum to 1.0")
