import os
import sys
import time
import math
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator
from features import FeatureExtractor
from models import build_logistic_model, build_hist_gbdt_model, PyTorchGPUPreferenceModel

CHAMPION_10K_BENCHMARK = 1.01601

print("=" * 65)
print("EXP 17: DEEP PREFERENCE SIGNALS & CALIBRATED GPU ARCHITECTURE")
print(f"Target: Beat Champion Benchmark {CHAMPION_10K_BENCHMARK:.5f}")
print("=" * 65)

# Load 10k benchmark dataset
df_train = load_dataset(os.path.join(BASE_DIR, "data", "train.csv"), nrows=10000, is_test=False)
y = df_train["target_label"].values
y_onehot = df_train[TARGET_COLUMNS].values
folds = list(get_stratified_group_folds(df_train, n_splits=5))

# Extended Feature Extraction: Add LSA Norms & Semantic Distances + Negative Constraints
RE_NEG_PROMPT = re.compile(r"\b(do not|don't|never|avoid|without|no|must not)\s+(\w+)", re.IGNORECASE)
RE_STEP = re.compile(r"(?:^|\n)\s*(?:step\s*\d+|\d+\.)\s+", re.IGNORECASE)
RE_PARAGRAPHS = re.compile(r"\n\s*\n")

class Exp17FeatureExtractor(FeatureExtractor):
    """
    Extends FeatureExtractor with:
    1. LSA semantic norms (||a_svd||, ||b_svd||) and Euclidean semantic distances.
    2. Char LSA structural norms and inter-response distances.
    3. Paragraph chunking, step progression, and negative constraint adherence.
    """
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_base_mat = super().transform(df)

        # Compute additional engineered columns
        p_texts = df["prompt_text"].values
        a_texts = df["response_a_text"].values
        b_texts = df["response_b_text"].values

        p_vecs = self.tfidf_vectorizer.transform(df["prompt_text"])
        a_vecs = self.tfidf_vectorizer.transform(df["response_a_text"])
        b_vecs = self.tfidf_vectorizer.transform(df["response_b_text"])

        p_svd = self.svd.transform(p_vecs)
        a_svd = self.svd.transform(a_vecs)
        b_svd = self.svd.transform(b_vecs)

        p_char = self.char_vectorizer.transform(df["prompt_text"])
        a_char = self.char_vectorizer.transform(df["response_a_text"])
        b_char = self.char_vectorizer.transform(df["response_b_text"])

        p_csvd = self.char_svd.transform(p_char)
        a_csvd = self.char_svd.transform(a_char)
        b_csvd = self.char_svd.transform(b_char)

        # 1. Semantic LSA Norms and Distances
        norm_a = np.linalg.norm(a_svd, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b_svd, axis=1, keepdims=True)
        diff_lsa_norm = (norm_a - norm_b).astype(np.float32)
        sym_lsa_norm = (0.5 * (norm_a + norm_b)).astype(np.float32)

        dist_ab = np.linalg.norm(a_svd - b_svd, axis=1, keepdims=True).astype(np.float32)
        dist_pa = np.linalg.norm(p_svd - a_svd, axis=1, keepdims=True)
        dist_pb = np.linalg.norm(p_svd - b_svd, axis=1, keepdims=True)
        diff_dist_p = (dist_pb - dist_pa).astype(np.float32)

        # 2. Subword Char LSA Norms and Distances
        cnorm_a = np.linalg.norm(a_csvd, axis=1, keepdims=True)
        cnorm_b = np.linalg.norm(b_csvd, axis=1, keepdims=True)
        diff_char_norm = (cnorm_a - cnorm_b).astype(np.float32)
        sym_char_norm = (0.5 * (cnorm_a + cnorm_b)).astype(np.float32)
        cdist_ab = np.linalg.norm(a_csvd - b_csvd, axis=1, keepdims=True).astype(np.float32)

        # 3. Structural & Negative Constraint Features
        extra_rows = []
        for p, a, b in zip(p_texts, a_texts, b_texts):
            # Paragraphs
            para_a = len(RE_PARAGRAPHS.findall(a))
            para_b = len(RE_PARAGRAPHS.findall(b))
            diff_para = float(para_a - para_b)
            sym_para = float(para_a + para_b)

            # Step progression
            steps_a = len(RE_STEP.findall(a))
            steps_b = len(RE_STEP.findall(b))
            diff_steps = float(steps_a - steps_b)
            sym_steps = float(steps_a + steps_b)

            # Negative constraint violation check
            neg_matches = RE_NEG_PROMPT.findall(p)
            diff_neg_violation = 0.0
            if neg_matches:
                for _, forbidden_word in neg_matches:
                    w_lower = forbidden_word.lower()
                    if len(w_lower) >= 3:
                        in_a = 1.0 if w_lower in a.lower() else 0.0
                        in_b = 1.0 if w_lower in b.lower() else 0.0
                        diff_neg_violation += (in_b - in_a)

            extra_rows.append([
                diff_para, sym_para,
                diff_steps, sym_steps,
                diff_neg_violation,
            ])

        extra_mat = np.asarray(extra_rows, dtype=np.float32)

        # Combine all features
        X_all = np.hstack([
            X_base_mat,
            diff_lsa_norm, sym_lsa_norm, dist_ab, diff_dist_p,
            diff_char_norm, sym_char_norm, cdist_ab,
            extra_mat,
        ])
        return X_all

t0_feat = time.time()
extractor = Exp17FeatureExtractor(
    use_tfidf=True,
    max_tfidf_features=15000,
    use_lsa=True,
    n_lsa_components=24,
    use_char_lsa=True,
    n_char_lsa_components=16,
)
X = extractor.fit_transform(df_train)
new_feat_names = list(extractor.feature_names_) + [
    "diff_lsa_norm", "sym_lsa_norm", "sym_lsa_dist_ab", "diff_lsa_dist_p",
    "diff_char_lsa_norm", "sym_char_lsa_norm", "sym_char_lsa_dist_ab",
    "diff_para", "sym_para", "diff_steps", "sym_steps", "diff_neg_violation",
]
extractor.feature_names_ = new_feat_names
print(f"Total features extracted: {X.shape[1]} ({time.time() - t0_feat:.1f}s)", flush=True)

# Train 5 Folds with Calibrated PyTorch Temperature Scaling
oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)
oof_pt = np.zeros_like(y_onehot, dtype=np.float64)

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y_onehot[val_idx]

    # 1. Logistic Regression
    lr = build_logistic_model(new_feat_names, C=0.5)
    lr.fit(X_tr, y_tr)
    oof_lr[val_idx] = lr.predict_proba(X_va)

    # 2. HistGBDT
    gb = build_hist_gbdt_model(new_feat_names, max_iter=150, learning_rate=0.04)
    gb.fit(X_tr, y_tr)
    oof_gbdt[val_idx] = gb.predict_proba(X_va)

    # 3. PyTorch GPU Model
    pt = PyTorchGPUPreferenceModel(new_feat_names, hidden_dim=128, epochs=25, batch_size=256)
    pt.fit(X_tr, y_tr)
    oof_pt[val_idx] = pt.predict_proba(X_va)

    print(f"  Fold {fold+1}/5 finished ({time.time() - t0:.1f}s)", flush=True)

print(f"\nOOF Component Log Losses:")
print(f"  LR:      {compute_log_loss(y_onehot, oof_lr):.5f}")
print(f"  HistGBDT: {compute_log_loss(y_onehot, oof_gbdt):.5f}")
print(f"  PyTorch: {compute_log_loss(y_onehot, oof_pt):.5f}")

# Pre-calibrate PyTorch with optimal temperature scaling on OOF
cal_pt = DirichletSymmetricCalibrator().fit_transform(oof_pt, y_onehot)
print(f"  PyTorch (Calibrated): {compute_log_loss(y_onehot, cal_pt):.5f}")

# Tri-Ensemble with 3-Parameter Dirichlet Calibration
for w_lr, w_gb, w_pt in [(0.35, 0.50, 0.15), (0.355, 0.480, 0.165), (0.38, 0.52, 0.10), (0.40, 0.50, 0.10)]:
    blend = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * cal_pt
    blend = blend / blend.sum(axis=1, keepdims=True)
    cal_blend = DirichletSymmetricCalibrator().fit_transform(blend, y_onehot)
    raw_loss = compute_log_loss(y_onehot, blend)
    cal_loss = compute_log_loss(y_onehot, cal_blend)
    delta = cal_loss - CHAMPION_10K_BENCHMARK
    print(f"Blend ({w_lr:.3f}/{w_gb:.3f}/{w_pt:.3f}) -> Raw: {raw_loss:.5f} | Calibrated: {cal_loss:.5f} | Delta: {delta:+.5f}")

# Save OOF predictions for fast inspection
np.savez(
    os.path.join(BASE_DIR, "scratch", "oof_exp17.npz"),
    lr=oof_lr,
    gbdt=oof_gbdt,
    pt=oof_pt,
    cal_pt=cal_pt,
    y=y_onehot,
)
print("Exp 17 complete.")
