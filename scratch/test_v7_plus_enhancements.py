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
from models import SymmetricModelWrapper, PyTorchGPUPreferenceModel

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")

# Load 10k dataset
train_csv = os.path.join(BASE_DIR, "data", "train.csv")
df_train = load_dataset(train_csv, nrows=10000, is_test=False)
y = df_train["target_label"].values
y_onehot = df_train[TARGET_COLUMNS].values
folds = list(get_stratified_group_folds(df_train, n_splits=5))

# Use the exact Version 7 feature logic via git show 054c919
import subprocess
v7_feat_code = subprocess.check_output(
    ["git", "-C", "d:/ag-ai/llm-classification-finetuning", "show", "054c919:src/features.py"]
).decode("utf-8")

loc_v7 = {}
exec(v7_feat_code, loc_v7)
FeatureExtractorV7 = loc_v7["FeatureExtractor"]

print("=================================================================")
print("BENCHMARKING EXACT VERSION 7 (158 FEATS) VS PROPOSED V11 UPGRADES")
print("=================================================================")

# Extract 158 features
t0 = time.time()
extractor_v7 = FeatureExtractorV7(
    use_tfidf=True,
    max_tfidf_features=15000,
    use_lsa=True,
    n_lsa_components=24,
    use_char_lsa=True,
    n_char_lsa_components=16,
)
X_158 = extractor_v7.fit_transform(df_train)
names_158 = extractor_v7.feature_names_
print(f"Extracted 158 features in {time.time() - t0:.1f}s. Shape: {X_158.shape}")

# Model 1: Logistic Regression C=0.5
def build_lr(names, C=0.5):
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
    return SymmetricModelWrapper(clf, names)

# Model 2: HistGBDT (Standard V7: max_iter=150, lr=0.05, min_samples_leaf=50)
def build_gbdt_std(names):
    clf = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=42
    )
    return SymmetricModelWrapper(clf, names)

# Model 3: HistGBDT (Regularized shallow: max_iter=250, lr=0.03, max_leaf_nodes=15, min_samples_leaf=100, l2=5.0)
def build_gbdt_shallow(names):
    clf = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=5.0, random_state=42
    )
    return SymmetricModelWrapper(clf, names)

# Run 5-fold CV for these models
oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt_std = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt_shallow = np.zeros_like(y_onehot, dtype=np.float64)
oof_pt = np.zeros_like(y_onehot, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X_158[train_idx], y[train_idx]
    X_va = X_158[val_idx]

    # LR
    m_lr = build_lr(names_158, C=0.5).fit(X_tr, y_tr)
    oof_lr[val_idx] = m_lr.predict_proba(X_va)

    # GBDT Std
    m_gb_std = build_gbdt_std(names_158).fit(X_tr, y_tr)
    oof_gbdt_std[val_idx] = m_gb_std.predict_proba(X_va)

    # GBDT Shallow
    m_gb_shallow = build_gbdt_shallow(names_158).fit(X_tr, y_tr)
    oof_gbdt_shallow[val_idx] = m_gb_shallow.predict_proba(X_va)

    # PyTorch GPU (25 epochs, raw temperature=1.0)
    m_pt = PyTorchGPUPreferenceModel(names_158, hidden_dim=128, epochs=25, batch_size=256, device=device)
    m_pt.fit(X_tr, y_tr)
    oof_pt[val_idx] = m_pt.predict_proba(X_va)

    print(f"  Fold {fold+1}/5 complete.")

print(f"\n--- Individual Model OOF Log Loss ---")
print(f"LR OOF Log Loss:           {compute_log_loss(y_onehot, oof_lr):.5f}")
print(f"GBDT Std OOF Log Loss:     {compute_log_loss(y_onehot, oof_gbdt_std):.5f}")
print(f"GBDT Shallow OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt_shallow):.5f}")
print(f"PyTorch GPU OOF Log Loss:  {compute_log_loss(y_onehot, oof_pt):.5f}")

# 1. Exact Version 7 blend (0.35 LR + 0.50 GBDT Std + 0.15 PyTorch)
blend_v7 = 0.35 * oof_lr + 0.50 * oof_gbdt_std + 0.15 * oof_pt
blend_v7 /= blend_v7.sum(axis=1, keepdims=True)
cal_v7 = DirichletSymmetricCalibrator().fit(blend_v7, y_onehot)
oof_v7_cal = cal_v7.transform(blend_v7)
print(f"\n[Version 7 Exact Replication] Calibrated OOF Log Loss: {compute_log_loss(y_onehot, oof_v7_cal):.5f}")

# 2. Dual Tree Quad-Ensemble:
# 0.30 LR + 0.35 GBDT Std + 0.20 GBDT Shallow + 0.15 PyTorch
blend_quad = 0.30 * oof_lr + 0.35 * oof_gbdt_std + 0.20 * oof_gbdt_shallow + 0.15 * oof_pt
blend_quad /= blend_quad.sum(axis=1, keepdims=True)
cal_quad = DirichletSymmetricCalibrator().fit(blend_quad, y_onehot)
oof_quad_cal = cal_quad.transform(blend_quad)
print(f"[Quad-Ensemble with Dual GBDT] Calibrated OOF Log Loss: {compute_log_loss(y_onehot, oof_quad_cal):.5f}")

# 3. Add Exact-Match Bayesian prior
exact_mask = (df_train["response_a_text"].str.strip() == df_train["response_b_text"].str.strip()).values
if np.any(exact_mask):
    oof_v7_post = oof_v7_cal.copy()
    oof_v7_post[exact_mask] = np.array([0.0491, 0.0491, 0.9018])
    print(f"[Version 7 + Exact-Match Prior] OOF Log Loss: {compute_log_loss(y_onehot, oof_v7_post):.5f} (Delta: {compute_log_loss(y_onehot, oof_v7_post) - compute_log_loss(y_onehot, oof_v7_cal):.5f})")

    oof_quad_post = oof_quad_cal.copy()
    oof_quad_post[exact_mask] = np.array([0.0491, 0.0491, 0.9018])
    print(f"[Quad-Ensemble + Exact-Match Prior] OOF Log Loss: {compute_log_loss(y_onehot, oof_quad_post):.5f} (Delta: {compute_log_loss(y_onehot, oof_quad_post) - compute_log_loss(y_onehot, oof_quad_cal):.5f})")

# Optimize blend weights directly via Dirichlet calibration
def obj(weights):
    w1, w2, w3 = weights
    w4 = max(1.0 - w1 - w2 - w3, 0.0)
    w_sum = w1 + w2 + w3 + w4
    w1, w2, w3, w4 = w1/w_sum, w2/w_sum, w3/w_sum, w4/w_sum
    b = w1 * oof_lr + w2 * oof_gbdt_std + w3 * oof_gbdt_shallow + w4 * oof_pt
    b /= b.sum(axis=1, keepdims=True)
    c = DirichletSymmetricCalibrator().fit(b, y_onehot)
    p = c.transform(b)
    if np.any(exact_mask):
        p[exact_mask] = np.array([0.0491, 0.0491, 0.9018])
    return compute_log_loss(y_onehot, p)

res = minimize(
    obj,
    x0=[0.30, 0.35, 0.20],
    bounds=[(0.1, 0.5), (0.1, 0.5), (0.05, 0.4)],
    method="Nelder-Mead",
)
w_opt = res.x
w_pt_opt = 1.0 - sum(w_opt)
print(f"\n[Optimal Quad-Ensemble Weights]: LR={w_opt[0]:.3f}, GBDT_std={w_opt[1]:.3f}, GBDT_shallow={w_opt[2]:.3f}, PyTorch={w_pt_opt:.3f}")
print(f"[Optimal Quad-Ensemble OOF Log Loss]: {res.fun:.5f}")
