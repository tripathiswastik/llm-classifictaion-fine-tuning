import os
import sys
import numpy as np
from scipy.optimize import minimize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from metric import compute_log_loss

data = np.load(os.path.join(BASE_DIR, "scratch", "oof_exp16.npz"))
oof_lr = data["lr"]
oof_gbdt = data["gbdt"]
oof_pt = data["pt"]
y = data["y"]

print(f"Loaded OOF shapes: LR={oof_lr.shape}, GBDT={oof_gbdt.shape}, PT={oof_pt.shape}, Y={y.shape}")
print(f"LR log loss: {compute_log_loss(y, oof_lr):.5f}")
print(f"GBDT log loss: {compute_log_loss(y, oof_gbdt):.5f}")
print(f"PyTorch log loss: {compute_log_loss(y, oof_pt):.5f}")

# 1. Arithmetic blend with Margin-Coupled Calibrator
def eval_arithmetic_blend(weights, t_p, b_t, a_t, g_t):
    w_lr, w_gb, w_pt = weights
    b = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * oof_pt
    b = b / b.sum(axis=1, keepdims=True)
    eps = 1e-15
    p = np.clip(b, eps, 1.0 - eps)
    log_a = np.log(p[:, 0])
    log_b = np.log(p[:, 1])
    log_tie = np.log(p[:, 2])
    z_ab = (log_a - log_b) / max(t_p, 1e-3)
    margin = np.abs(z_ab) / (1.0 + np.abs(z_ab))
    z_tie = b_t * (log_tie - 0.5 * (log_a + log_b)) + a_t - g_t * margin
    l_a = 0.5 * z_ab
    l_b = -0.5 * z_ab
    l_tie = z_tie
    l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
    exp_a = np.exp(l_a[:, None] - l_max)
    exp_b = np.exp(l_b[:, None] - l_max)
    exp_tie = np.exp(l_tie[:, None] - l_max)
    denom = exp_a + exp_b + exp_tie
    p_cal = np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])
    return compute_log_loss(y, p_cal)

# Optimize arithmetic blend jointly
def obj_arithmetic(params):
    w_lr, w_gb, t_p, b_t, a_t, g_t = params
    w_pt = max(1.0 - w_lr - w_gb, 0.0)
    w_total = w_lr + w_gb + w_pt
    w_lr, w_gb, w_pt = w_lr / w_total, w_gb / w_total, w_pt / w_total
    return eval_arithmetic_blend((w_lr, w_gb, w_pt), t_p, b_t, a_t, g_t)

res_arith = minimize(
    obj_arithmetic,
    x0=[0.35, 0.50, 1.0899, 0.7518, 0.1064, 0.3010],
    bounds=[(0.1, 0.8), (0.1, 0.8), (0.5, 3.0), (0.2, 3.0), (-2.0, 2.0), (-2.0, 3.0)],
    method="L-BFGS-B",
)
w_lr, w_gb, t_p, b_t, a_t, g_t = res_arith.x
w_pt = max(1.0 - w_lr - w_gb, 0.0)
w_tot = w_lr + w_gb + w_pt
w_lr, w_gb, w_pt = w_lr / w_tot, w_gb / w_tot, w_pt / w_tot
loss_arith = res_arith.fun
print(f"\n1. Optimized Arithmetic Blend + Margin-Coupled Calibrator:")
print(f"   Log Loss: {loss_arith:.5f}")
print(f"   Weights: LR={w_lr:.3f}, GBDT={w_gb:.3f}, GPU={w_pt:.3f}")
print(f"   Calibrator: T_pair={t_p:.4f}, Beta_tie={b_t:.4f}, Alpha_tie={a_t:.4f}, Gamma_tie={g_t:.4f}")

# 2. Geometric (Log-Linear) Blend with Margin-Coupled Calibrator
def eval_geometric_blend(params):
    w_lr, w_gb, t_p, b_t, a_t, g_t = params
    w_pt = max(1.0 - w_lr - w_gb, 0.0)
    w_total = w_lr + w_gb + w_pt
    w_lr, w_gb, w_pt = w_lr / w_total, w_gb / w_total, w_pt / w_total

    eps = 1e-15
    log_lr = np.log(np.clip(oof_lr, eps, 1.0))
    log_gb = np.log(np.clip(oof_gbdt, eps, 1.0))
    log_pt = np.log(np.clip(oof_pt, eps, 1.0))

    log_geom = w_lr * log_lr + w_gb * log_gb + w_pt * log_pt
    # Symmetrize geometric log-odds: log_geom[:, 0] and log_geom[:, 1]
    # Symmetrized pair difference:
    z_ab_raw = (log_geom[:, 0] - log_geom[:, 1])
    z_tie_raw = log_geom[:, 2] - 0.5 * (log_geom[:, 0] + log_geom[:, 1])

    z_ab = z_ab_raw / max(t_p, 1e-3)
    margin = np.abs(z_ab) / (1.0 + np.abs(z_ab))
    z_tie = b_t * z_tie_raw + a_t - g_t * margin

    l_a = 0.5 * z_ab
    l_b = -0.5 * z_ab
    l_tie = z_tie

    l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
    exp_a = np.exp(l_a[:, None] - l_max)
    exp_b = np.exp(l_b[:, None] - l_max)
    exp_tie = np.exp(l_tie[:, None] - l_max)
    denom = exp_a + exp_b + exp_tie
    p_cal = np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])
    return compute_log_loss(y, p_cal)

res_geom = minimize(
    eval_geometric_blend,
    x0=[0.35, 0.50, 1.0899, 0.7518, 0.1064, 0.3010],
    bounds=[(0.1, 0.8), (0.1, 0.8), (0.5, 3.0), (0.2, 3.0), (-2.0, 2.0), (-2.0, 3.0)],
    method="L-BFGS-B",
)
w_lr, w_gb, t_p, b_t, a_t, g_t = res_geom.x
w_pt = max(1.0 - w_lr - w_gb, 0.0)
w_tot = w_lr + w_gb + w_pt
w_lr, w_gb, w_pt = w_lr / w_tot, w_gb / w_tot, w_pt / w_tot
loss_geom = res_geom.fun
print(f"\n2. Geometric (Log-Linear) Blend + Margin-Coupled Calibrator:")
print(f"   Log Loss: {loss_geom:.5f}")
print(f"   Weights: LR={w_lr:.3f}, GBDT={w_gb:.3f}, GPU={w_pt:.3f}")
print(f"   Calibrator: T_pair={t_p:.4f}, Beta_tie={b_t:.4f}, Alpha_tie={a_t:.4f}, Gamma_tie={g_t:.4f}")

champion_delta = min(loss_arith, loss_geom) - 1.01601
print(f"\nFinal Best Log Loss: {min(loss_arith, loss_geom):.5f} (Delta vs Exp 14: {champion_delta:+.5f})")
