import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration import TemperatureCalibrator
from data_loader import TARGET_COLUMNS, get_stratified_group_folds, load_dataset
from features import FeatureExtractor
from metric import compute_log_loss
from models import SymmetricModelWrapper, build_logistic_model, get_hist_gbdt_loss
from sklearn.ensemble import HistGradientBoostingClassifier


def main():
    print("=" * 70)
    print("EXP 06: CHAR-TFIDF ALIGNMENT & GBDT L2 REGULARIZATION TUNING")
    print("Champion 10k Benchmark to Beat: 1.02890")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)

    print("Extracting current 55 production features...")
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X_base = extractor.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values

    # Step A: Char n-gram TF-IDF alignment
    print("Extracting Character n-gram TF-IDF alignment (char_wb 3-5)...")
    corpus = (
        list(df["prompt_text"].values)
        + list(df["response_a_text"].values)
        + list(df["response_b_text"].values)
    )
    char_tfidf = TfidfVectorizer(
        max_features=15000,
        sublinear_tf=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    char_tfidf.fit(corpus)

    p_vecs = char_tfidf.transform(df["prompt_text"])
    a_vecs = char_tfidf.transform(df["response_a_text"])
    b_vecs = char_tfidf.transform(df["response_b_text"])

    sim_pa = np.asarray(p_vecs.multiply(a_vecs).sum(axis=1)).flatten()
    sim_pb = np.asarray(p_vecs.multiply(b_vecs).sum(axis=1)).flatten()
    sim_ab = np.asarray(a_vecs.multiply(b_vecs).sum(axis=1)).flatten()

    diff_char_sim = sim_pa - sim_pb
    sym_char_ab_sim = sim_ab
    sym_char_mean_p = 0.5 * (sim_pa + sim_pb)

    X_char = np.column_stack([diff_char_sim, sym_char_ab_sim, sym_char_mean_p]).astype(np.float32)
    char_feature_names = [
        "diff_char_tfidf_prompt_sim",
        "sym_char_tfidf_ab_sim",
        "sym_char_tfidf_mean_prompt",
    ]

    X_58 = np.hstack([X_base, X_char])
    all_feature_names = extractor.feature_names_ + char_feature_names
    print(f"Candidate feature matrix: {X_58.shape} (55 base + 3 char TF-IDF = 58 features)")

    # Test variations of HistGBDT:
    # 1. Baseline HistGBDT: l2_regularization=0.0, min_samples_leaf=50
    # 2. Regularized HistGBDT: l2_regularization=1.0, min_samples_leaf=75
    # 3. Smoothed HistGBDT: l2_regularization=2.0, min_samples_leaf=100
    loss_param = get_hist_gbdt_loss()
    gbdt_configs = {
        "Default (L2=0.0, leaf=50)": {"l2_regularization": 0.0, "min_samples_leaf": 50},
        "Regularized (L2=1.0, leaf=75)": {"l2_regularization": 1.0, "min_samples_leaf": 75},
        "High-Reg (L2=2.0, leaf=100)": {"l2_regularization": 2.0, "min_samples_leaf": 100},
    }

    folds = list(get_stratified_group_folds(df, n_splits=5, random_state=42))

    for config_name, gparams in gbdt_configs.items():
        print(f"\nEvaluating configuration: {config_name} on 58 features...")
        oof_lr = np.zeros((len(df), 3), dtype=np.float64)
        oof_gbdt = np.zeros((len(df), 3), dtype=np.float64)

        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X_58[train_idx], y[train_idx]
            X_val = X_58[val_idx]

            lr = build_logistic_model(all_feature_names, C=0.5)
            raw_gbdt = HistGradientBoostingClassifier(
                loss=loss_param,
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=gparams["min_samples_leaf"],
                l2_regularization=gparams["l2_regularization"],
                random_state=42,
            )
            gbdt = SymmetricModelWrapper(raw_gbdt, all_feature_names)

            lr.fit(X_train, y_train)
            gbdt.fit(X_train, y_train)

            oof_lr[val_idx] = lr.predict_proba(X_val)
            oof_gbdt[val_idx] = gbdt.predict_proba(X_val)

        # Optimize blend weight
        def blend_obj(alpha):
            a = float(alpha[0])
            p = a * oof_lr + (1.0 - a) * oof_gbdt
            p /= p.sum(axis=1, keepdims=True)
            return compute_log_loss(y_true_onehot, p)

        res_alpha = minimize(blend_obj, x0=[0.40], bounds=[(0.0, 1.0)], method="L-BFGS-B")
        best_alpha = float(res_alpha.x[0])
        best_blend = best_alpha * oof_lr + (1.0 - best_alpha) * oof_gbdt
        best_blend /= best_blend.sum(axis=1, keepdims=True)
        tuned_loss = compute_log_loss(y_true_onehot, best_blend)

        # Calibrate
        calibrator = TemperatureCalibrator()
        calibrator.fit(best_blend, y_true_onehot)
        cal_preds = calibrator.transform(best_blend)
        cal_loss = compute_log_loss(y_true_onehot, cal_preds)

        print(f"  Tuned LR Weight: {best_alpha:.4f} (GBDT: {1.0 - best_alpha:.4f})")
        print(f"  Raw Blend Loss: {tuned_loss:.5f}")
        print(f"  Calibrated Loss: {cal_loss:.5f} (T={calibrator.temperature:.4f}, TieScale={calibrator.tie_scale:.4f})")
        print(f"  Delta vs Champion 1.02890: {cal_loss - 1.02890:+.5f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
