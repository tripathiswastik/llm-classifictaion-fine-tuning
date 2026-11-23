import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")

# Load 10k dataset
train_csv = os.path.join(BASE_DIR, "data", "train.csv")
df_train = load_dataset(train_csv, nrows=10000, is_test=False)
y = df_train["target_label"].values
y_onehot = df_train[TARGET_COLUMNS].values
folds = list(get_stratified_group_folds(df_train, n_splits=5))

# -------------------------------------------------------------
# 1. Feature Extractor V7 (158 features)
# -------------------------------------------------------------
from features import compute_pairwise_features, RE_PARAGRAPHS, RE_STEP, RE_NEG_PROMPT

# Build clean V7 72-base feature extractor
def extract_v7_base(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    # Extract only the original 72 continuous base features
    rows = []
    p_texts = df["prompt_text"].values
    a_texts = df["response_a_text"].values
    b_texts = df["response_b_text"].values
    a_turns = [len(t) for t in df["response_a_turns"].values]
    b_turns = [len(t) for t in df["response_b_turns"].values]

    for p, a, b, ta, tb in zip(p_texts, a_texts, b_texts, a_turns, b_turns):
        feat = compute_pairwise_features(p, a, b, ta, tb)
        # Select original 72 keys
        base_keys = [
            "diff_char_len", "diff_word_count", "diff_avg_word_len", "diff_sent_count",
            "diff_avg_sent_len", "diff_punc_count", "diff_digit_count", "diff_upper_count",
            "diff_code_chars", "diff_code_blocks", "diff_bold_count", "diff_italic_count",
            "diff_bullet_count", "diff_flesch_reading_ease", "diff_syllable_count",
            "diff_first_person_count", "diff_content_word_overlap", "diff_log_char",
            "diff_log_word", "diff_turn_count", "diff_jaccard_prompt", "diff_contain_prompt",
            "diff_ttr", "diff_entropy", "diff_repetition_rate", "diff_compression_ratio",
            "diff_bigram_cov", "diff_connectives",
            "sym_char_len", "sym_word_count", "sym_avg_word_len", "sym_sent_count",
            "sym_avg_sent_len", "sym_punc_count", "sym_digit_count", "sym_upper_count",
            "sym_code_chars", "sym_code_blocks", "sym_bold_count", "sym_italic_count",
            "sym_bullet_count", "sym_flesch_reading_ease", "sym_syllable_count",
            "sym_first_person_count", "sym_content_word_overlap", "sym_log_char",
            "sym_log_word", "sym_turn_count", "sym_ttr", "sym_entropy", "sym_repetition_rate",
            "sym_compression_ratio", "sym_bigram_cov", "sym_connectives", "sym_jaccard_ab",
            "sym_contain_ab", "sym_jaccard_prompt_mean", "sym_prompt_char_len",
            "sym_prompt_word_count", "sym_prompt_avg_word_len", "sym_prompt_sent_count",
            "sym_prompt_digit_count", "sym_prompt_upper_count", "sym_prompt_flesch_reading_ease",
            "sym_prompt_syllables", "sym_prompt_ttr", "sym_prompt_entropy", "sym_prompt_compression",
            "sym_prompt_code_chars", "sym_prompt_code_blocks", "sym_prompt_turn_count",
        ]
        rows.append([feat[k] for k in base_keys])
    return np.asarray(rows, dtype=np.float32), base_keys

X_base_v7, base_keys = extract_v7_base(df_train)
print(f"Extracted V7 Base: {X_base_v7.shape} (72 features)")

# Add Word LSA + Char LSA
def extract_dual_lsa(df: pd.DataFrame, n_word: int = 24, n_char: int = 16) -> Tuple[np.ndarray, List[str]]:
    # Word TF-IDF + SVD
    p_texts = df["prompt_text"].values
    a_texts = df["response_a_text"].values
    b_texts = df["response_b_text"].values
    all_texts = list(p_texts) + list(a_texts) + list(b_texts)

    vec_w = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2), min_df=2)
    vec_w.fit(all_texts)
    svd_w = TruncatedSVD(n_components=n_word, random_state=42)
    svd_w.fit(vec_w.transform(all_texts))

    p_w = svd_w.transform(vec_w.transform(p_texts))
    a_w = svd_w.transform(vec_w.transform(a_texts))
    b_w = svd_w.transform(vec_w.transform(b_texts))

    def get_cos(u, v):
        nu = np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        nv = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return np.sum((u / nu) * (v / nv), axis=1, keepdims=True).astype(np.float32)

    cos_pa_w = get_cos(p_w, a_w)
    cos_pb_w = get_cos(p_w, b_w)
    cos_ab_w = get_cos(a_w, b_w)
    diff_cos_w = (cos_pa_w - cos_pb_w).astype(np.float32)
    sym_cos_ab_w = cos_ab_w.astype(np.float32)
    sym_mean_cos_w = (0.5 * (cos_pa_w + cos_pb_w)).astype(np.float32)

    diff_svd_w = (a_w - b_w).astype(np.float32)
    sym_svd_w = (0.5 * (a_w + b_w)).astype(np.float32)

    dim_cols_w = []
    names_w = ["diff_lsa_prompt_cos", "sym_lsa_ab_cos", "sym_lsa_mean_prompt_cos"]
    for i in range(n_word):
        dim_cols_w.append(diff_svd_w[:, i:i+1])
        dim_cols_w.append(sym_svd_w[:, i:i+1])
        names_w.append(f"diff_lsa_dim_{i}")
        names_w.append(f"sym_lsa_dim_{i}")

    X_w = np.hstack([diff_cos_w, sym_cos_ab_w, sym_mean_cos_w] + dim_cols_w)

    # Char TF-IDF + SVD
    vec_c = TfidfVectorizer(max_features=15000, analyzer="char", ngram_range=(3, 5), min_df=2)
    vec_c.fit(all_texts)
    svd_c = TruncatedSVD(n_components=n_char, random_state=42)
    svd_c.fit(vec_c.transform(all_texts))

    p_c = svd_c.transform(vec_c.transform(p_texts))
    a_c = svd_c.transform(vec_c.transform(a_texts))
    b_c = svd_c.transform(vec_c.transform(b_texts))

    cos_pa_c = get_cos(p_c, a_c)
    cos_pb_c = get_cos(p_c, b_c)
    cos_ab_c = get_cos(a_c, b_c)
    diff_cos_c = (cos_pa_c - cos_pb_c).astype(np.float32)
    sym_cos_ab_c = cos_ab_c.astype(np.float32)
    sym_mean_cos_c = (0.5 * (cos_pa_c + cos_pb_c)).astype(np.float32)

    diff_svd_c = (a_c - b_c).astype(np.float32)
    sym_svd_c = (0.5 * (a_c + b_c)).astype(np.float32)

    dim_cols_c = []
    names_c = ["diff_char_lsa_prompt_cos", "sym_char_lsa_ab_cos", "sym_char_lsa_mean_prompt_cos"]
    for i in range(n_char):
        dim_cols_c.append(diff_svd_c[:, i:i+1])
        dim_cols_c.append(sym_svd_c[:, i:i+1])
        names_c.append(f"diff_char_lsa_dim_{i}")
        names_c.append(f"sym_char_lsa_dim_{i}")

    X_c = np.hstack([diff_cos_c, sym_cos_ab_c, sym_mean_cos_c] + dim_cols_c)
    X_lsa = np.hstack([X_w, X_c])
    return X_lsa, names_w + names_c

X_lsa_v7, lsa_names_v7 = extract_dual_lsa(df_train, n_word=24, n_char=16)
X_v7 = np.hstack([X_base_v7, X_lsa_v7])
names_v7 = list(base_keys) + lsa_names_v7
print(f"Total V7 Feature Matrix: {X_v7.shape} ({len(names_v7)} features)")

# Compare 158 features with 182 features using HistGBDT and Logistic Regression
from models import build_logistic_model, build_hist_gbdt_model

print("\n--- Evaluating Version 7 (158 Features) 5-Fold CV ---")
oof_lr_v7 = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt_v7 = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt_reg = np.zeros_like(y_onehot, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_v7[train_idx], y[train_idx]
    X_va = X_v7[val_idx]

    lr = build_logistic_model(names_v7, C=0.5)
    lr.fit(X_tr, y_tr)
    oof_lr_v7[val_idx] = lr.predict_proba(X_va)

    # Standard GBDT
    gb = build_hist_gbdt_model(names_v7, max_iter=150, learning_rate=0.05, min_samples_leaf=20)
    gb.fit(X_tr, y_tr)
    oof_gbdt_v7[val_idx] = gb.predict_proba(X_va)

    # Regularized GBDT (l2_regularization=2.0, max_iter=200, lr=0.04)
    clf_reg = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.04, l2_regularization=2.0,
        min_samples_leaf=30, random_state=42
    )
    from models import SymmetricModelWrapper
    gb_reg = SymmetricModelWrapper(clf_reg, names_v7)
    gb_reg.fit(X_tr, y_tr)
    oof_gbdt_reg[val_idx] = gb_reg.predict_proba(X_va)

print(f"V7 (158 feats) LR Loss:             {compute_log_loss(y_onehot, oof_lr_v7):.5f}")
print(f"V7 (158 feats) Standard GBDT Loss:  {compute_log_loss(y_onehot, oof_gbdt_v7):.5f}")
print(f"V7 (158 feats) Regularized GBDT:    {compute_log_loss(y_onehot, oof_gbdt_reg):.5f}")

# Calibrated blend
blend_std = 0.40 * oof_lr_v7 + 0.60 * oof_gbdt_v7
cal_std = DirichletSymmetricCalibrator().fit(blend_std, y_onehot)
print(f"V7 (158 feats) Calibrated Blend:    {compute_log_loss(y_onehot, cal_std.transform(blend_std)):.5f}")

blend_reg = 0.38 * oof_lr_v7 + 0.62 * oof_gbdt_reg
cal_reg = DirichletSymmetricCalibrator().fit(blend_reg, y_onehot)
print(f"V7 (158 feats) Calibrated Reg Blend:{compute_log_loss(y_onehot, cal_reg.transform(blend_reg)):.5f}")

# Now test semantic expansion (Word LSA 32 + Char LSA 24 = 119 semantic features + 72 base = 191 features)
print("\n--- Evaluating Semantic Expansion (Word 32 + Char 24) ---")
X_lsa_exp, lsa_names_exp = extract_dual_lsa(df_train, n_word=32, n_char=24)
X_exp = np.hstack([X_base_v7, X_lsa_exp])
names_exp = list(base_keys) + lsa_names_exp
print(f"Expanded Semantic Matrix: {X_exp.shape} ({len(names_exp)} features)")

oof_lr_exp = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt_exp = np.zeros_like(y_onehot, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_exp[train_idx], y[train_idx]
    X_va = X_exp[val_idx]

    lr = build_logistic_model(names_exp, C=0.5)
    lr.fit(X_tr, y_tr)
    oof_lr_exp[val_idx] = lr.predict_proba(X_va)

    clf_reg = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.04, l2_regularization=2.0,
        min_samples_leaf=30, random_state=42
    )
    gb_reg = SymmetricModelWrapper(clf_reg, names_exp)
    gb_reg.fit(X_tr, y_tr)
    oof_gbdt_exp[val_idx] = gb_reg.predict_proba(X_va)

print(f"Expanded LSA LR Loss:   {compute_log_loss(y_onehot, oof_lr_exp):.5f}")
print(f"Expanded LSA GBDT Loss: {compute_log_loss(y_onehot, oof_gbdt_exp):.5f}")

blend_exp = 0.38 * oof_lr_exp + 0.62 * oof_gbdt_exp
cal_exp = DirichletSymmetricCalibrator().fit(blend_exp, y_onehot)
print(f"Expanded LSA Calibrated Blend: {compute_log_loss(y_onehot, cal_exp.transform(blend_exp)):.5f}")
