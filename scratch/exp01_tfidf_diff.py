import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import TARGET_COLUMNS, get_stratified_group_folds, load_dataset
from src.features import FeatureExtractor
from src.metric import compute_log_loss, evaluate_breakdown
from src.models import build_ensemble_model, build_logistic_model


def main():
    print("=" * 70)
    print("EXP 01: TF-IDF DIFFERENCE & COSINE ALIGNMENT BENCHMARK")
    print("Baseline Champion Target: Log Loss < 1.03775")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)
    print(f"Loaded {len(df)} rows across {df['prompt_group'].nunique()} prompt groups.")

    # 1. Base tabular features
    print("Extracting baseline tabular features...")
    ext = FeatureExtractor()
    X_base = ext.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values
    feature_names = ext.feature_names_

    # 2. Extract TF-IDF cosine alignment features
    print("Computing TF-IDF cosine alignment features...")
    corpus = list(df["prompt_text"]) + list(df["response_a_text"]) + list(df["response_b_text"])
    tfidf = TfidfVectorizer(max_features=15000, sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(corpus)

    p_vecs = tfidf.transform(df["prompt_text"])
    a_vecs = tfidf.transform(df["response_a_text"])
    b_vecs = tfidf.transform(df["response_b_text"])

    # Compute dot products (cosine similarities because TfidfVectorizer outputs L2 normalized vectors)
    # sim_pa = sum(P * A), sim_pb = sum(P * B), sim_ab = sum(A * B)
    sim_pa = np.asarray(p_vecs.multiply(a_vecs).sum(axis=1)).flatten()
    sim_pb = np.asarray(p_vecs.multiply(b_vecs).sum(axis=1)).flatten()
    sim_ab = np.asarray(a_vecs.multiply(b_vecs).sum(axis=1)).flatten()

    diff_prompt_sim = sim_pa - sim_pb
    mean_prompt_sim = 0.5 * (sim_pa + sim_pb)
    sym_ab_sim = sim_ab

    # Candidate A: Augment tabular features with cosine features
    # Anti-symmetric: diff_prompt_sim (starts with diff_)
    # Symmetric: sym_ab_sim, sym_mean_prompt_sim (starts with sym_)
    X_aug = np.column_stack([
        X_base,
        diff_prompt_sim,
        sym_ab_sim,
        mean_prompt_sim
    ])
    aug_feature_names = feature_names + ["diff_tfidf_prompt_sim", "sym_tfidf_ab_sim", "sym_tfidf_mean_prompt"]
    print(f"Augmented feature matrix: {X_aug.shape}")

    # Run 5-fold CV on Candidate A
    print("\n--- Evaluating Candidate A (Tabular + TF-IDF Cosine Alignment) ---")
    oof_preds_a = np.zeros((len(df), 3), dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(get_stratified_group_folds(df, n_splits=5, random_state=42)):
        X_train, y_train = X_aug[train_idx], y[train_idx]
        X_val, y_val = X_aug[val_idx], y_true_onehot[val_idx]

        model = build_ensemble_model(aug_feature_names, lr_weight=0.60)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)
        oof_preds_a[val_idx] = pred
        fold_loss = compute_log_loss(y_val, pred)
        print(f"  Fold {fold+1}/5 Log Loss: {fold_loss:.5f}")

    loss_a = compute_log_loss(y_true_onehot, oof_preds_a)
    delta_a = loss_a - 1.03775
    print(f"\nCandidate A Overall OOF Log Loss: {loss_a:.5f} (Delta vs Champion: {delta_a:+.5f})")

    # Candidate B: High-dimensional sparse TF-IDF difference model
    print("\n--- Evaluating Candidate B (Sparse TF-IDF Difference Logistic Model) ---")
    # For sparse difference: delta_vec = a_vec - b_vec
    diff_sparse = a_vecs - b_vecs
    oof_preds_b = np.zeros((len(df), 3), dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(get_stratified_group_folds(df, n_splits=5, random_state=42)):
        X_sp_train = diff_sparse[train_idx]
        y_train = y[train_idx]
        X_sp_val = diff_sparse[val_idx]
        y_val = y_true_onehot[val_idx]

        # Symmetrized training: duplicate with negated difference and flipped targets
        X_sp_swap = -X_sp_train
        y_swap = y_train.copy()
        y_swap[y_train == 0] = 1
        y_swap[y_train == 1] = 0

        from scipy.sparse import vstack
        X_sp_aug = vstack([X_sp_train, X_sp_swap])
        y_sp_aug = np.concatenate([y_train, y_swap])

        lr_sp = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
        lr_sp.fit(X_sp_aug, y_sp_aug)

        # Test-time symmetrization
        p_orig = lr_sp.predict_proba(X_sp_val)
        p_swap = lr_sp.predict_proba(-X_sp_val)
        p_swap_inv = np.zeros_like(p_swap)
        p_swap_inv[:, 0] = p_swap[:, 1]
        p_swap_inv[:, 1] = p_swap[:, 0]
        p_swap_inv[:, 2] = p_swap[:, 2]
        p_sym = 0.5 * (p_orig + p_swap_inv)
        p_sym = p_sym / p_sym.sum(axis=1, keepdims=True)

        oof_preds_b[val_idx] = p_sym
        fold_loss = compute_log_loss(y_val, p_sym)
        print(f"  Fold {fold+1}/5 Log Loss: {fold_loss:.5f}")

    loss_b = compute_log_loss(y_true_onehot, oof_preds_b)
    print(f"Candidate B Standalone Sparse OOF Log Loss: {loss_b:.5f}")

    # Candidate C: Blend of Candidate A (Tabular+Cosine) and Candidate B (Sparse TF-IDF)
    print("\n--- Evaluating Candidate C (Blend of Tabular + Sparse TF-IDF) ---")
    best_blend_w = 0.0
    best_blend_loss = 999.0
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 1.0]:
        blend = w * oof_preds_a + (1.0 - w) * oof_preds_b
        blend = blend / blend.sum(axis=1, keepdims=True)
        score = compute_log_loss(y_true_onehot, blend)
        if score < best_blend_loss:
            best_blend_loss = score
            best_blend_w = w
        print(f"  Blend Weight (Tabular={w:.2f}, Sparse={1-w:.2f}): Log Loss = {score:.5f}")

    print("\n" + "=" * 70)
    print(f"FINAL DECISION GATE:")
    print(f"Baseline Champion: 1.03775")
    print(f"Candidate A (Cosine Features): {loss_a:.5f} (Delta: {loss_a - 1.03775:+.5f})")
    print(f"Candidate B (Sparse TF-IDF): {loss_b:.5f} (Delta: {loss_b - 1.03775:+.5f})")
    print(f"Candidate C (Best Blend at w={best_blend_w:.2f}): {best_blend_loss:.5f} (Delta: {best_blend_loss - 1.03775:+.5f})")

    if best_blend_loss < 1.03775:
        print(f"RESULT: STRICT IMPROVEMENT CONFIRMED! Delta = {best_blend_loss - 1.03775:.5f}. PROCEED TO PROMOTE.")
    else:
        print(f"RESULT: NO IMPROVEMENT (Delta >= 0). STRICT INVARIANT ENFORCED: TERMINATE AND DISCARD.")
    print("=" * 70)


if __name__ == "__main__":
    main()
