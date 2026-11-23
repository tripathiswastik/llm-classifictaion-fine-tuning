import os
import sys
import numpy as np
from scipy.optimize import minimize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator, MarginCoupledDirichletCalibrator

data = np.load(os.path.join(BASE_DIR, "scratch", "oof_exp16.npz"))
oof_lr = data["lr"]
oof_gbdt = data["gbdt"]
oof_pt = data["pt"]
y = data["y"]

print("=" * 60)
print("COMPONENT BASELINE LOG LOSSES (10k CV):")
print("=" * 60)
print(f"  Logistic Regression (LR):  {compute_log_loss(y, oof_lr):.5f}")
print(f"  HistGBDT:                  {compute_log_loss(y, oof_gbdt):.5f}")
print(f"  PyTorch GPU NN:            {compute_log_loss(y, oof_pt):.5f}")

# Calibrate PyTorch alone
pt_cal = DirichletSymmetricCalibrator().fit_transform(oof_pt, y)
print(f"  PyTorch GPU (Calibrated):  {compute_log_loss(y, pt_cal):.5f}")

print("\n" + "=" * 60)
print("1. DUAL BLEND: LR + GBDT (Without PyTorch):")
print("=" * 60)
for w_lr in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
    w_gb = 1.0 - w_lr
    dual_raw = w_lr * oof_lr + w_gb * oof_gbdt
    dual_dir = DirichletSymmetricCalibrator().fit_transform(dual_raw, y)
    dual_margin = MarginCoupledDirichletCalibrator().fit_transform(dual_raw, y)
    print(f"  LR {w_lr:.2f} + GBDT {w_gb:.2f} -> Raw: {compute_log_loss(y, dual_raw):.5f} | Dirichlet: {compute_log_loss(y, dual_dir):.5f} | Margin-Coupled: {compute_log_loss(y, dual_margin):.5f}")

print("\n" + "=" * 60)
print("2. TRI BLEND: LR + GBDT + PyTorch:")
print("=" * 60)
# V7 weights: 0.35 LR, 0.50 GBDT, 0.15 PT
v7_raw = 0.35 * oof_lr + 0.50 * oof_gbdt + 0.15 * oof_pt
v7_dir = DirichletSymmetricCalibrator().fit_transform(v7_raw, y)
v7_margin = MarginCoupledDirichletCalibrator().fit_transform(v7_raw, y)
print(f"  V7 weights (0.35/0.50/0.15) -> Raw: {compute_log_loss(y, v7_raw):.5f} | Dirichlet: {compute_log_loss(y, v7_dir):.5f} | Margin-Coupled: {compute_log_loss(y, v7_margin):.5f}")

# V8 weights: 0.355 LR, 0.480 GBDT, 0.165 PT
v8_raw = 0.355 * oof_lr + 0.480 * oof_gbdt + 0.165 * oof_pt
v8_dir = DirichletSymmetricCalibrator().fit_transform(v8_raw, y)
v8_margin = MarginCoupledDirichletCalibrator().fit_transform(v8_raw, y)
print(f"  V8 weights (0.355/0.480/0.165) -> Raw: {compute_log_loss(y, v8_raw):.5f} | Dirichlet: {compute_log_loss(y, v8_dir):.5f} | Margin-Coupled: {compute_log_loss(y, v8_margin):.5f}")

# Calibrated blending: blend the CALIBRATED probabilities of each model
print("\n" + "=" * 60)
print("3. POST-CALIBRATED ENSEMBLE (Pre-calibrating each model before blend):")
print("=" * 60)
cal_lr = DirichletSymmetricCalibrator().fit_transform(oof_lr, y)
cal_gb = DirichletSymmetricCalibrator().fit_transform(oof_gbdt, y)
cal_pt = DirichletSymmetricCalibrator().fit_transform(oof_pt, y)

print(f"  Calibrated LR loss:   {compute_log_loss(y, cal_lr):.5f}")
print(f"  Calibrated GBDT loss: {compute_log_loss(y, cal_gb):.5f}")
print(f"  Calibrated PT loss:   {compute_log_loss(y, cal_pt):.5f}")

for w_lr in [0.30, 0.35, 0.40, 0.45]:
    for w_gb in [0.45, 0.50, 0.55]:
        w_pt = max(0.0, 1.0 - w_lr - w_gb)
        if w_pt < 0.0 or w_pt > 0.3:
            continue
        p_ens = w_lr * cal_lr + w_gb * cal_gb + w_pt * cal_pt
        p_ens = p_ens / p_ens.sum(axis=1, keepdims=True)
        final_cal = DirichletSymmetricCalibrator().fit_transform(p_ens, y)
        print(f"  Weights ({w_lr:.2f}/{w_gb:.2f}/{w_pt:.2f}) -> Blend loss: {compute_log_loss(y, p_ens):.5f} | Re-calibrated: {compute_log_loss(y, final_cal):.5f}")
