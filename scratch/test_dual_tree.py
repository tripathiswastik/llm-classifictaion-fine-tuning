import os
import sys
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import FeatureExtractor
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator
from models import get_hist_gbdt_loss

# Load cached base OOFs
cached = np.load(os.path.join(BASE_DIR, "scratch", "oof_exp16.npz"))
oof_lr = cached["lr"]
oof_pt = cached["pt"]
oof_gb1 = cached["gbdt"]
y_onehot = cached["y"]

print(f"GBDT 1 Log Loss: {compute_log_loss(y_onehot, oof_gb1):.5f}")

# Train GBDT 2 with slower learning rate and more trees
df_train = load_dataset(os.path.join(BASE_DIR, "data", "train.csv"), nrows=10000, is_test=False)
y = df_train["target_label"].values
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

diff_idx = [i for i, name in enumerate(extractor.feature_names_) if name.startswith("diff_")]
oof_gb2 = np.zeros_like(y_onehot, dtype=np.float64)

t0 = time.time()
for fold, (train_idx, val_idx) in enumerate(folds):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    # Pair-swap augmentation
    X_swap_tr = X_tr.copy()
    X_swap_tr[:, diff_idx] = -X_swap_tr[:, diff_idx]
    y_swap_tr = y_tr.copy()
    y_swap_tr[y_tr == 0] = 1
    y_swap_tr[y_tr == 1] = 0

    X_aug = np.vstack([X_tr, X_swap_tr])
    y_aug = np.concatenate([y_tr, y_swap_tr])

    scaler = StandardScaler()
    X_aug_sc = scaler.fit_transform(X_aug)

    gb2 = HistGradientBoostingClassifier(
        loss=get_hist_gbdt_loss(),
        max_iter=250,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=123,
    )
    gb2.fit(X_aug_sc, y_aug)

    X_va_sc = scaler.transform(X_va)
    X_va_swap = X_va.copy()
    X_va_swap[:, diff_idx] = -X_va_swap[:, diff_idx]
    X_va_swap_sc = scaler.transform(X_va_swap)

    p_orig = gb2.predict_proba(X_va_sc)
    p_swap = gb2.predict_proba(X_va_swap_sc)
    p_sym = 0.5 * (p_orig + np.column_stack([p_swap[:, 1], p_swap[:, 0], p_swap[:, 2]]))
    oof_gb2[val_idx] = p_sym / p_sym.sum(axis=1, keepdims=True)

print(f"GBDT 2 Training Time: {time.time() - t0:.1f}s")
print(f"GBDT 2 Log Loss: {compute_log_loss(y_onehot, oof_gb2):.5f}")

# Tree blend
tree_blend = 0.5 * oof_gb1 + 0.5 * oof_gb2
print(f"Dual GBDT Blend Log Loss: {compute_log_loss(y_onehot, tree_blend):.5f}")

# Quad-Ensemble: LR + GBDT1 + GBDT2 + PyTorch GPU
quad_raw = 0.30 * oof_lr + 0.30 * oof_gb1 + 0.25 * oof_gb2 + 0.15 * oof_pt
quad_raw /= quad_raw.sum(axis=1, keepdims=True)
print(f"Quad-Ensemble Raw Log Loss: {compute_log_loss(y_onehot, quad_raw):.5f}")

# Calibrate with Margin-Coupled Calibrator
from exp16_margin_coupled_calibration import MarginCoupledDirichletCalibrator
cal = MarginCoupledDirichletCalibrator().fit(quad_raw, y_onehot)
cal_loss = compute_log_loss(y_onehot, cal.transform(quad_raw))
print(f"Quad-Ensemble Calibrated Log Loss: {cal_loss:.5f}")
print(f"Delta vs Champion (1.01601): {cal_loss - 1.01601:+.5f}")
