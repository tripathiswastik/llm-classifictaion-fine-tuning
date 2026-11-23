import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .calibration import TemperatureCalibrator
    from .data_loader import (
        TARGET_COLUMNS,
        get_stratified_group_folds,
        load_dataset,
    )
    from .features import FeatureExtractor
    from .metric import compute_log_loss, evaluate_breakdown
    from .models import (
        build_ensemble_model,
        build_hist_gbdt_model,
        build_logistic_model,
        build_pytorch_gpu_model,
        build_tri_ensemble_model,
    )
except ImportError:
    from calibration import TemperatureCalibrator  # type: ignore
    from data_loader import (  # type: ignore
        TARGET_COLUMNS,
        get_stratified_group_folds,
        load_dataset,
    )
    from features import FeatureExtractor  # type: ignore
    from metric import compute_log_loss, evaluate_breakdown  # type: ignore
    from models import (  # type: ignore
        build_ensemble_model,
        build_hist_gbdt_model,
        build_logistic_model,
        build_pytorch_gpu_model,
        build_tri_ensemble_model,
    )


def run_cross_validation(
    data_dir: str,
    model_type: str = "ensemble",
    n_splits: int = 5,
    sample_rows: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray, FeatureExtractor, list, TemperatureCalibrator]:
    """
    Executes leak-free StratifiedGroupKFold cross-validation with temperature calibration.
    """
    train_path = os.path.join(data_dir, "train.csv")
    print(f"Loading training data from {train_path}...")
    df_train = load_dataset(train_path, nrows=sample_rows, is_test=False)
    print(f"Loaded {len(df_train)} rows across {df_train['prompt_group'].nunique()} prompt groups.")

    print("Extracting features (Pairwise + Repetition + TF-IDF Cosine)...")
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X = extractor.fit_transform(df_train)
    y = df_train["target_label"].values
    y_true_onehot = df_train[TARGET_COLUMNS].values
    print(f"Extracted feature matrix: {X.shape} with {len(extractor.feature_names_)} features.")

    oof_preds = np.zeros((len(df_train), 3), dtype=np.float64)
    fold_models = []
    fold_scores = []

    print(f"\n--- Starting {n_splits}-Fold StratifiedGroupKFold CV ({model_type}) ---")
    for fold, (train_idx, val_idx) in enumerate(get_stratified_group_folds(df_train, n_splits=n_splits)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val_onehot = X[val_idx], y_true_onehot[val_idx]

        if model_type == "hist_gbdt":
            model = build_hist_gbdt_model(extractor.feature_names_)
        elif model_type == "logistic":
            model = build_logistic_model(extractor.feature_names_, C=0.5)
        elif model_type == "ensemble":
            model = build_ensemble_model(extractor.feature_names_, lr_weight=0.44)
        elif model_type == "pytorch_gpu":
            model = build_pytorch_gpu_model(extractor.feature_names_)
        elif model_type == "tri_ensemble":
            model = build_tri_ensemble_model(extractor.feature_names_)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)
        oof_preds[val_idx] = val_pred

        fold_loss = compute_log_loss(y_val_onehot, val_pred)
        fold_scores.append(fold_loss)
        fold_models.append(model)
        print(f"Fold {fold + 1}/{n_splits} Log Loss: {fold_loss:.5f} (Val size: {len(val_idx)})")

    raw_metrics = evaluate_breakdown(y_true_onehot, oof_preds)
    print("\n--- Raw Out-Of-Fold Evaluation ---")
    for k, v in raw_metrics.items():
        print(f"  {k}: {v:.5f}")

    print("\n--- Fitting Symmetric Temperature & Tie-Scale Calibration ---")
    calibrator = TemperatureCalibrator()
    calibrator.fit(oof_preds, y_true_onehot)
    cal_oof_preds = calibrator.transform(oof_preds)
    cal_metrics = evaluate_breakdown(y_true_onehot, cal_oof_preds)

    print(f"Optimal Temperature: {calibrator.temperature:.4f}, Tie Scale: {calibrator.tie_scale:.4f}")
    print("\n--- Calibrated Out-Of-Fold Evaluation ---")
    for k, v in cal_metrics.items():
        print(f"  {k}: {v:.5f}")

    return cal_metrics["overall_log_loss"], cal_oof_preds, y_true_onehot, extractor, fold_models, calibrator


def generate_submission(
    data_dir: str,
    output_path: str,
    extractor: FeatureExtractor,
    fold_models: list,
    calibrator: Optional[TemperatureCalibrator] = None,
) -> pd.DataFrame:
    """
    Generates verified Kaggle submission by ensembling fold models and applying calibration.
    """
    test_path = os.path.join(data_dir, "test.csv")
    sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

    print(f"\nLoading test data from {test_path}...")
    df_test = load_dataset(test_path, is_test=True)
    raw_sample = pd.read_csv(sample_sub_path)
    if not isinstance(raw_sample, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from pd.read_csv, got {type(raw_sample)}")
    df_sample: pd.DataFrame = raw_sample

    X_test = extractor.transform(df_test)
    print(f"Transformed test features: {X_test.shape}")

    # Ensemble fold predictions
    preds_list = [model.predict_proba(X_test) for model in fold_models]
    avg_preds = np.mean(preds_list, axis=0)
    avg_preds = avg_preds / avg_preds.sum(axis=1, keepdims=True)

    # Apply calibration if available
    if calibrator is not None:
        avg_preds = calibrator.transform(avg_preds)
        print(f"Applied calibration (T={calibrator.temperature:.4f}, TieScale={calibrator.tie_scale:.4f}).")

    sub = pd.DataFrame({
        "id": df_test["id"].values,
        "winner_model_a": avg_preds[:, 0],
        "winner_model_b": avg_preds[:, 1],
        "winner_tie": avg_preds[:, 2],
    })

    # Verification assertions
    assert list(sub.columns) == list(df_sample.columns), "Column names mismatch"
    assert len(sub) == len(df_sample), f"Row count mismatch: {len(sub)} vs {len(df_sample)}"
    assert not sub.isnull().any().any(), "Submission contains null values"
    prob_sums = sub[TARGET_COLUMNS].sum(axis=1).values
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5, err_msg="Probabilities do not sum to 1.0")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub.to_csv(output_path, index=False)
    print(f"Submission written successfully to {output_path} ({len(sub)} rows).")
    print("Sample head:")
    print(sub.head())
    return sub
