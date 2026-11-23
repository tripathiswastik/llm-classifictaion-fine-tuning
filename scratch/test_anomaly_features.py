import os
import sys
import math
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator
from features import FeatureExtractor
from models import build_logistic_model, build_hist_gbdt_model

print("=" * 65)
print("BENCHMARKING ANOMALY & EDGE CASE HARDENED FEATURES")
print("=" * 65)

df_train = load_dataset(os.path.join(BASE_DIR, "data", "train.csv"), nrows=10000, is_test=False)
y = df_train["target_label"].values
y_onehot = df_train[TARGET_COLUMNS].values
folds = list(get_stratified_group_folds(df_train, n_splits=5))

# Baseline extractor (170 features)
extractor = FeatureExtractor(
    use_tfidf=True,
    max_tfidf_features=15000,
    use_lsa=True,
    n_lsa_components=24,
    use_char_lsa=True,
    n_char_lsa_components=16,
)
X_base = extractor.fit_transform(df_train)

# Compute new anomaly-hardened features
extra_rows = []
for idx, row in df_train.iterrows():
    p_turns = row["prompt_turns"]
    a_turns = row["response_a_turns"]
    b_turns = row["response_b_turns"]

    # 1. Multi-turn final-turn signals
    is_multi = 1.0 if len(p_turns) > 1 else 0.0
    n_turns = float(len(p_turns))

    final_p = p_turns[-1] if p_turns else ""
    final_a = a_turns[-1] if a_turns else ""
    final_b = b_turns[-1] if b_turns else ""

    len_fa = float(len(final_a))
    len_fb = float(len(final_b))
    diff_final_len = len_fa - len_fb
    diff_final_log_len = math.log(len_fa + 1.0) - math.log(len_fb + 1.0)
    sym_final_len = len_fa + len_fb

    # Final turn prompt overlap
    words_fp = set(final_p.lower().split())
    words_fa = set(final_a.lower().split())
    words_fb = set(final_b.lower().split())
    jacc_fa = len(words_fp & words_fa) / max(len(words_fp | words_fa), 1)
    jacc_fb = len(words_fp & words_fb) / max(len(words_fp | words_fb), 1)
    diff_final_jacc = float(jacc_fa - jacc_fb)
    sym_final_jacc = 0.5 * float(jacc_fa + jacc_fb)

    # 2. Non-linear length ratio cliffs
    len_a = float(len(row["response_a_text"]))
    len_b = float(len(row["response_b_text"]))
    ratio_15 = float(len_a > 1.5 * len_b) - float(len_b > 1.5 * len_a)
    ratio_20 = float(len_a > 2.0 * len_b) - float(len_b > 2.0 * len_a)
    ratio_30 = float(len_a > 3.0 * len_b) - float(len_b > 3.0 * len_a)
    sym_extreme_len_imbalance = float((len_a > 2.0 * len_b) or (len_b > 2.0 * len_a))

    # 3. Exact match indicator (for model learning)
    is_exact = float(row["response_a_text"].strip() == row["response_b_text"].strip())

    extra_rows.append([
        diff_final_len, diff_final_log_len, sym_final_len,
        diff_final_jacc, sym_final_jacc,
        is_multi, n_turns,
        ratio_15, ratio_20, ratio_30, sym_extreme_len_imbalance,
        is_exact,
    ])

extra_mat = np.asarray(extra_rows, dtype=np.float32)
X_hardened = np.hstack([X_base, extra_mat])
print(f"Base Features: {X_base.shape[1]} -> Hardened Features: {X_hardened.shape[1]}")

new_names = list(extractor.feature_names_) + [
    "diff_final_len", "diff_final_log_len", "sym_final_len",
    "diff_final_jacc", "sym_final_jacc",
    "sym_is_multi_turn", "sym_num_turns",
    "diff_len_ratio_15", "diff_len_ratio_20", "diff_len_ratio_30", "sym_extreme_len_imbalance",
    "sym_is_exact_match",
]

# Run 5-fold CV with GBDT + LR
oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_hardened[train_idx], y[train_idx]
    X_va = X_hardened[val_idx]

    lr = build_logistic_model(new_names, C=0.5)
    lr.fit(X_tr, y_tr)
    oof_lr[val_idx] = lr.predict_proba(X_va)

    gb = build_hist_gbdt_model(new_names, max_iter=150, learning_rate=0.05, min_samples_leaf=50)
    gb.fit(X_tr, y_tr)
    oof_gbdt[val_idx] = gb.predict_proba(X_va)

loss_lr = compute_log_loss(y_onehot, oof_lr)
loss_gb = compute_log_loss(y_onehot, oof_gbdt)
print(f"LR Log Loss:   {loss_lr:.5f}")
print(f"GBDT Log Loss: {loss_gb:.5f}")

# Blend LR + GBDT
blend = 0.40 * oof_lr + 0.60 * oof_gbdt
cal = DirichletSymmetricCalibrator().fit(blend, y_onehot)
cal_blend = cal.transform(blend)
print(f"Calibrated Blend Loss: {compute_log_loss(y_onehot, cal_blend):.5f}")

# Exact-match Bayesian post-processing
exact_indices = np.where(df_train["response_a_text"].str.strip() == df_train["response_b_text"].str.strip())[0]
if len(exact_indices) > 0:
    post_blend = cal_blend.copy()
    post_blend[exact_indices] = [0.0491, 0.0491, 0.9018]
    post_loss = compute_log_loss(y_onehot, post_blend)
    print(f"With Exact-Match Bayesian Post-Processing: {post_loss:.5f} (Delta: {post_loss - compute_log_loss(y_onehot, cal_blend):.5f})")

np.savez(os.path.join(BASE_DIR, "scratch", "oof_hardened.npz"), lr=oof_lr, gbdt=oof_gbdt, y=y_onehot)
