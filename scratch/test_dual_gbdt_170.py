import os
import sys
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator
from models import build_logistic_model, get_hist_gbdt_loss

print("=" * 65)
print("TESTING DUAL-GBDT + LOGISTIC REGRESSION ON 170 FEATURES")
print("=" * 65)

df_train = load_dataset(os.path.join(BASE_DIR, "data", "train.csv"), nrows=10000, is_test=False)
y = df_train["target_label"].values
y_onehot = df_train[TARGET_COLUMNS].values
folds = list(get_stratified_group_folds(df_train, n_splits=5))

extractor = FeatureExtractor(
    use_tfidf=True,
    max_tfidf_features=15000,
    use_lsa=True,
    n_lsa_components=24,
    use_char_lsa=True,
    n_char_lsa_components=16,
)
X = extractor.fit_transform(df_train)
feat_names = extractor.feature_names_
diff_idx = [i for i, name in enumerate(feat_names) if name.startswith("diff_")]

print(f"Features: {X.shape[1]}")

oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
oof_gb1 = np.zeros_like(y_onehot, dtype=np.float64)
oof_gb2 = np.zeros_like(y_onehot, dtype=np.float64)

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va = X[val_idx]

    # Augment for symmetry
    X_swap_tr = X_tr.copy()
    X_swap_tr[:, diff_idx] = -X_swap_tr[:, diff_idx]
    y_swap_tr = y_tr.copy()
    y_swap_tr[y_tr == 0] = 1
    y_swap_tr[y_tr == 1] = 0

    X_aug = np.vstack([X_tr, X_swap_tr])
    y_aug = np.concatenate([y_tr, y_swap_tr])

    scaler = StandardScaler()
    X_aug_sc = scaler.fit_transform(X_aug)

    X_va_sc = scaler.transform(X_va)
    X_va_swap = X_va.copy()
    X_va_swap[:, diff_idx] = -X_va_swap[:, diff_idx]
    X_va_swap_sc = scaler.transform(X_va_swap)

    # 1. LR
    lr = build_logistic_model(feat_names, C=0.5)
    lr.fit(X_tr, y_tr)
    oof_lr[val_idx] = lr.predict_proba(X_va)

    # 2. GBDT 1 (Champion config: lr=0.05, max_leaf_nodes=31, min_samples_leaf=50, max_iter=150)
    gb1 = HistGradientBoostingClassifier(
        loss=get_hist_gbdt_loss(),
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        random_state=42,
    )
    gb1.fit(X_aug_sc, y_aug)
    p_orig1 = gb1.predict_proba(X_va_sc)
    p_swap1 = gb1.predict_proba(X_va_swap_sc)
    p_sym1 = 0.5 * (p_orig1 + np.column_stack([p_swap1[:, 1], p_swap1[:, 0], p_swap1[:, 2]]))
    oof_gb1[val_idx] = p_sym1 / p_sym1.sum(axis=1, keepdims=True)

    # 3. GBDT 2 (Smoother regularized config: lr=0.03, max_leaf_nodes=21, min_samples_leaf=75, l2=2.0, max_iter=120)
    gb2 = HistGradientBoostingClassifier(
        loss=get_hist_gbdt_loss(),
        max_iter=120,
        learning_rate=0.03,
        max_leaf_nodes=21,
        min_samples_leaf=75,
        l2_regularization=2.0,
        random_state=123,
    )
    gb2.fit(X_aug_sc, y_aug)
    p_orig2 = gb2.predict_proba(X_va_sc)
    p_swap2 = gb2.predict_proba(X_va_swap_sc)
    p_sym2 = 0.5 * (p_orig2 + np.column_stack([p_swap2[:, 1], p_swap2[:, 0], p_swap2[:, 2]]))
    oof_gb2[val_idx] = p_sym2 / p_sym2.sum(axis=1, keepdims=True)

    print(f"Fold {fold+1}/5 done ({time.time() - t0:.1f}s)", flush=True)

print(f"\nComponent Log Losses:")
print(f"  LR:     {compute_log_loss(y_onehot, oof_lr):.5f}")
print(f"  GBDT 1: {compute_log_loss(y_onehot, oof_gb1):.5f}")
print(f"  GBDT 2: {compute_log_loss(y_onehot, oof_gb2):.5f}")

# Blend GBDT 1 + GBDT 2
tree_blend = 0.55 * oof_gb1 + 0.45 * oof_gb2
print(f"  Dual GBDT Blend: {compute_log_loss(y_onehot, tree_blend):.5f}")

# Tri-Blend: LR + GBDT 1 + GBDT 2
for w_lr in [0.30, 0.35, 0.40]:
    for w_g1 in [0.35, 0.40, 0.45]:
        w_g2 = max(0.0, 1.0 - w_lr - w_g1)
        b = w_lr * oof_lr + w_g1 * oof_gb1 + w_g2 * oof_gb2
        b = b / b.sum(axis=1, keepdims=True)
        cal = DirichletSymmetricCalibrator().fit_transform(b, y_onehot)
        print(f"  Weights ({w_lr:.2f}/{w_g1:.2f}/{w_g2:.2f}) -> Raw: {compute_log_loss(y_onehot, b):.5f} | Calibrated: {compute_log_loss(y_onehot, cal):.5f}")

# Save OOF
np.savez(
    os.path.join(BASE_DIR, "scratch", "oof_dual_gbdt.npz"),
    lr=oof_lr,
    gb1=oof_gb1,
    gb2=oof_gb2,
    y=y_onehot,
)
