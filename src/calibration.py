from typing import Optional, Tuple
import numpy as np
from scipy.optimize import minimize

try:
    from .metric import compute_log_loss
except ImportError:
    from metric import compute_log_loss  # type: ignore


class TemperatureCalibrator:
    """
    Fits and applies symmetric temperature and tie scaling.
    Preserves anti-symmetry: A and B are scaled identically by temperature T,
    while Tie probability is scaled symmetrically by tie_scale.
    """

    def __init__(self, temperature: float = 1.0, tie_scale: float = 1.0):
        self.temperature = float(temperature)
        self.tie_scale = float(tie_scale)

    def fit(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> "TemperatureCalibrator":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        def objective(params: np.ndarray) -> float:
            temp, tie_sc = params
            log_p = np.log(p) / max(temp, 1e-4)
            log_p[:, 2] += np.log(max(tie_sc, 1e-5))
            exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
            p_cal = exp_p / exp_p.sum(axis=1, keepdims=True)
            return compute_log_loss(y_true_onehot, p_cal)

        res = minimize(
            objective,
            x0=[self.temperature, self.tie_scale],
            bounds=[(0.2, 5.0), (0.2, 5.0)],
            method="L-BFGS-B",
        )
        self.temperature = float(res.x[0])
        self.tie_scale = float(res.x[1])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)
        log_p = np.log(p) / max(self.temperature, 1e-4)
        log_p[:, 2] += np.log(max(self.tie_scale, 1e-5))
        exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
        p_cal = exp_p / exp_p.sum(axis=1, keepdims=True)
        return p_cal

    def fit_transform(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> np.ndarray:
        return self.fit(probs, y_true_onehot).transform(probs)


class DirichletSymmetricCalibrator:
    """
    3-Parameter Decoupled Symmetric Dirichlet Calibrator.
    Guarantees exact mathematical anti-symmetry and probability simplex constraints:
      z_AB = (log(p_A) - log(p_B)) / T_pair
      z_tie = beta_tie * (log(p_tie) - 0.5 * (log(p_A) + log(p_B))) + alpha_tie
    """
    def __init__(self, t_pair: float = 1.0, beta_tie: float = 1.0, alpha_tie: float = 0.0):
        self.t_pair = float(t_pair)
        self.beta_tie = float(beta_tie)
        self.alpha_tie = float(alpha_tie)

    def fit(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> "DirichletSymmetricCalibrator":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab = log_a - log_b
        z_tie = log_tie - 0.5 * (log_a + log_b)

        def objective(params: np.ndarray) -> float:
            t_p, b_t, a_t = params
            z_ab_scaled = z_ab / max(t_p, 1e-3)
            z_tie_scaled = b_t * z_tie + a_t

            l_a = 0.5 * z_ab_scaled
            l_b = -0.5 * z_ab_scaled
            l_tie = z_tie_scaled

            l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
            exp_a = np.exp(l_a[:, None] - l_max)
            exp_b = np.exp(l_b[:, None] - l_max)
            exp_tie = np.exp(l_tie[:, None] - l_max)

            denom = exp_a + exp_b + exp_tie
            p_cal = np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])
            return compute_log_loss(y_true_onehot, p_cal)

        res = minimize(
            objective,
            x0=[self.t_pair, self.beta_tie, self.alpha_tie],
            bounds=[(0.2, 5.0), (0.2, 5.0), (-3.0, 3.0)],
            method="L-BFGS-B",
        )
        self.t_pair = float(res.x[0])
        self.beta_tie = float(res.x[1])
        self.alpha_tie = float(res.x[2])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab = (log_a - log_b) / max(self.t_pair, 1e-3)
        z_tie = self.beta_tie * (log_tie - 0.5 * (log_a + log_b)) + self.alpha_tie

        l_a = 0.5 * z_ab
        l_b = -0.5 * z_ab
        l_tie = z_tie

        l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
        exp_a = np.exp(l_a[:, None] - l_max)
        exp_b = np.exp(l_b[:, None] - l_max)
        exp_tie = np.exp(l_tie[:, None] - l_max)

        denom = exp_a + exp_b + exp_tie
        return np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])

    def fit_transform(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> np.ndarray:
        return self.fit(probs, y_true_onehot).transform(probs)


class MarginCoupledDirichletCalibrator:
    """
    4-Parameter Margin-Coupled Symmetric Dirichlet Calibrator.
    Guarantees exact mathematical anti-symmetry and simplex constraints:
      z_AB = (log(p_A) - log(p_B)) / T_pair
      margin = |z_AB| / (1.0 + |z_AB|)
      z_tie = beta_tie * (log(p_tie) - 0.5 * (log(p_A) + log(p_B))) + alpha_tie - gamma_tie * margin
    """
    def __init__(
        self,
        t_pair: float = 1.0,
        beta_tie: float = 1.0,
        alpha_tie: float = 0.0,
        gamma_tie: float = 0.0,
    ):
        self.t_pair = float(t_pair)
        self.beta_tie = float(beta_tie)
        self.alpha_tie = float(alpha_tie)
        self.gamma_tie = float(gamma_tie)

    def fit(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> "MarginCoupledDirichletCalibrator":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab = log_a - log_b
        z_tie_base = log_tie - 0.5 * (log_a + log_b)

        def objective(params: np.ndarray) -> float:
            t_p, b_t, a_t, g_t = params
            z_ab_scaled = z_ab / max(t_p, 1e-3)
            margin = np.abs(z_ab_scaled) / (1.0 + np.abs(z_ab_scaled))
            z_tie_scaled = b_t * z_tie_base + a_t - g_t * margin

            l_a = 0.5 * z_ab_scaled
            l_b = -0.5 * z_ab_scaled
            l_tie = z_tie_scaled

            l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
            exp_a = np.exp(l_a[:, None] - l_max)
            exp_b = np.exp(l_b[:, None] - l_max)
            exp_tie = np.exp(l_tie[:, None] - l_max)

            denom = exp_a + exp_b + exp_tie
            p_cal = np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])
            return compute_log_loss(y_true_onehot, p_cal)

        res = minimize(
            objective,
            x0=[self.t_pair, self.beta_tie, self.alpha_tie, self.gamma_tie],
            bounds=[(0.2, 5.0), (0.2, 5.0), (-3.0, 3.0), (-3.0, 5.0)],
            method="L-BFGS-B",
        )
        self.t_pair = float(res.x[0])
        self.beta_tie = float(res.x[1])
        self.alpha_tie = float(res.x[2])
        self.gamma_tie = float(res.x[3])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        log_a = np.log(p[:, 0])
        log_b = np.log(p[:, 1])
        log_tie = np.log(p[:, 2])

        z_ab_scaled = (log_a - log_b) / max(self.t_pair, 1e-3)
        margin = np.abs(z_ab_scaled) / (1.0 + np.abs(z_ab_scaled))
        z_tie_scaled = self.beta_tie * (log_tie - 0.5 * (log_a + log_b)) + self.alpha_tie - self.gamma_tie * margin

        l_a = 0.5 * z_ab_scaled
        l_b = -0.5 * z_ab_scaled
        l_tie = z_tie_scaled

        l_max = np.maximum(np.maximum(l_a, l_b), l_tie)[:, None]
        exp_a = np.exp(l_a[:, None] - l_max)
        exp_b = np.exp(l_b[:, None] - l_max)
        exp_tie = np.exp(l_tie[:, None] - l_max)

        denom = exp_a + exp_b + exp_tie
        return np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])

    def fit_transform(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> np.ndarray:
        return self.fit(probs, y_true_onehot).transform(probs)
