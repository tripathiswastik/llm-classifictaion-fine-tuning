import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration import TemperatureCalibrator
from src.data_loader import TARGET_COLUMNS, get_stratified_group_folds, load_dataset
from src.features import FeatureExtractor
from src.metric import compute_log_loss
from src.models import build_ensemble_model, build_hist_gbdt_model, build_logistic_model

RE_VOWELS = re.compile(r"[aeiouy]+", re.IGNORECASE)
RE_WORDS = re.compile(r"\b\w+\b")
RE_NUM_CONSTRAINT = re.compile(r"\b(?:give me|list|write|provide)?\s*(\d{1,2})\s*(?:reasons|points|tips|ways|examples|items|steps|ideas)\b", re.IGNORECASE)


def count_syllables(word: str) -> int:
    """Estimates syllables in an English word."""
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    matches = RE_VOWELS.findall(w)
    return max(len(matches), 1)


def compute_readability_and_constraint_features(prompt: str, resp_a: str, resp_b: str):
    """
    Computes reading ease, code density, and numerical constraint adherence.
    """
    words_a = RE_WORDS.findall(resp_a)
    words_b = RE_WORDS.findall(resp_b)

    n_w_a = max(len(words_a), 1)
    n_w_b = max(len(words_b), 1)

    sents_a = max(len(re.split(r"[.!?\n]+", resp_a)), 1)
    sents_b = max(len(re.split(r"[.!?\n]+", resp_b)), 1)

    # Average syllables per word
    syl_a = sum(count_syllables(w) for w in words_a) / n_w_a
    syl_b = sum(count_syllables(w) for w in words_b) / n_w_b

    # Flesch Reading Ease estimate (higher = easier to read, 0-100 typical)
    # FRE = 206.835 - 1.015 * (words/sentence) - 84.6 * (syllables/word)
    fre_a = 206.835 - 1.015 * (n_w_a / sents_a) - 84.6 * syl_a
    fre_b = 206.835 - 1.015 * (n_w_b / sents_b) - 84.6 * syl_b

    diff_fre = fre_a - fre_b
    sym_mean_fre = 0.5 * (fre_a + fre_b)

    # Code density
    code_chars_a = sum(len(c) for c in re.findall(r"```[\s\S]*?```", resp_a))
    code_chars_b = sum(len(c) for c in re.findall(r"```[\s\S]*?```", resp_b))
    density_a = code_chars_a / max(len(resp_a), 1)
    density_b = code_chars_b / max(len(resp_b), 1)

    diff_code_density = density_a - density_b
    sym_mean_code_density = 0.5 * (density_a + density_b)

    # Constraint adherence
    # Check if prompt asked for N items
    match = RE_NUM_CONSTRAINT.search(prompt)
    if match:
        target_n = int(match.group(1))
        # Count bullets or numbered items in response
        bullets_a = len(re.findall(r"(?:^|\n)\s*(?:[-*+]|\d+\.)\s+", resp_a))
        bullets_b = len(re.findall(r"(?:^|\n)\s*(?:[-*+]|\d+\.)\s+", resp_b))
        err_a = abs(bullets_a - target_n)
        err_b = abs(bullets_b - target_n)
        diff_constraint_err = float(err_b - err_a)  # positive if A has smaller error than B
    else:
        diff_constraint_err = 0.0

    return {
        "diff_flesch_reading_ease": diff_fre,
        "sym_mean_flesch_reading_ease": sym_mean_fre,
        "diff_code_density": diff_code_density,
        "sym_mean_code_density": sym_mean_code_density,
        "diff_constraint_adherence": diff_constraint_err,
    }


def main():
    print("=" * 70)
    print("EXP 04: READABILITY, CODE DENSITY, CONSTRAINT ADHERENCE & BLEND TUNING")
    print("Champion Benchmark to Beat: 1.03292")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)

    print("Extracting baseline 39 features...")
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X_base = extractor.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values

    print("Extracting readability, code density, and constraint features...")
    new_feats_list = []
    p_texts = df["prompt_text"].values
    a_texts = df["response_a_text"].values
    b_texts = df["response_b_text"].values

    for p, a, b in zip(p_texts, a_texts, b_texts):
        new_feats_list.append(compute_readability_and_constraint_features(p, a, b))

    new_feat_names = list(new_feats_list[0].keys())
    X_new = np.array([[row[k] for k in new_feat_names] for row in new_feats_list], dtype=np.float32)

    X_combined = np.hstack([X_base, X_new])
    all_feature_names = extractor.feature_names_ + new_feat_names
    print(f"Total features: {X_combined.shape[1]} ({len(all_feature_names)})")

    # Step 1: Evaluate baseline 39 features vs 44 features with static 0.60/0.40 blend
    print("\nEvaluating 5-Fold StratifiedGroupKFold on 44 features...")
    oof_lr = np.zeros((len(df), 3), dtype=np.float64)
    oof_gbdt = np.zeros((len(df), 3), dtype=np.float64)

    folds = list(get_stratified_group_folds(df, n_splits=5, random_state=42))
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X_combined[train_idx], y[train_idx]
        X_val = X_combined[val_idx]

        lr = build_logistic_model(all_feature_names, C=0.5)
        gbdt = build_hist_gbdt_model(all_feature_names, max_iter=150, learning_rate=0.05)

        lr.fit(X_train, y_train)
        gbdt.fit(X_train, y_train)

        oof_lr[val_idx] = lr.predict_proba(X_val)
        oof_gbdt[val_idx] = gbdt.predict_proba(X_val)

        p_blend = 0.60 * oof_lr[val_idx] + 0.40 * oof_gbdt[val_idx]
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        fold_loss = compute_log_loss(y_true_onehot[val_idx], p_blend)
        print(f"  Fold {fold+1}/5 Log Loss (0.60/0.40 blend): {fold_loss:.5f}")

    raw_blend = 0.60 * oof_lr + 0.40 * oof_gbdt
    raw_blend /= raw_blend.sum(axis=1, keepdims=True)
    raw_loss = compute_log_loss(y_true_onehot, raw_blend)
    print(f"\nRaw 44-Feature Static Blend OOF Loss: {raw_loss:.5f} (vs 1.03292: {raw_loss - 1.03292:+.5f})")

    # Step 2: Optimal Blend Weight Tuning
    print("\nOptimizing Ensemble Blend Weight (alpha * LR + (1 - alpha) * GBDT)...")
    def blend_obj(alpha):
        a = float(alpha[0])
        p = a * oof_lr + (1.0 - a) * oof_gbdt
        p /= p.sum(axis=1, keepdims=True)
        return compute_log_loss(y_true_onehot, p)

    res_alpha = minimize(blend_obj, x0=[0.60], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    best_alpha = float(res_alpha.x[0])
    best_blend_loss = res_alpha.fun
    print(f"Optimal LR Weight: {best_alpha:.4f} (GBDT Weight: {1.0 - best_alpha:.4f})")
    print(f"Optimal Blend OOF Loss: {best_blend_loss:.5f} (Delta vs 1.03292: {best_blend_loss - 1.03292:+.5f})")

    # Step 3: Temperature & Tie Scaling on Optimal Blend
    opt_blend = best_alpha * oof_lr + (1.0 - best_alpha) * oof_gbdt
    opt_blend /= opt_blend.sum(axis=1, keepdims=True)

    calibrator = TemperatureCalibrator()
    calibrator.fit(opt_blend, y_true_onehot)
    cal_preds = calibrator.transform(opt_blend)
    cal_loss = compute_log_loss(y_true_onehot, cal_preds)

    print(f"\nCalibrated Optimal Blend Loss: {cal_loss:.5f} (T={calibrator.temperature:.4f}, TieScale={calibrator.tie_scale:.4f})")
    print(f"Delta vs Champion 1.03292: {cal_loss - 1.03292:+.5f}")

    print("\n" + "=" * 70)
    print("DECISION GATE EVALUATION:")
    print(f"Champion 10k Benchmark: 1.03292")
    print(f"Candidate Exp 04 Loss:  {cal_loss:.5f}")
    if cal_loss < 1.03292:
        delta = cal_loss - 1.03292
        print(f"RESULT: STRICT IMPROVEMENT CONFIRMED! Delta = {delta:.5f} < 0. PROCEED TO PROMOTE.")
    else:
        delta = cal_loss - 1.03292
        print(f"RESULT: NO IMPROVEMENT (Delta = {delta:+.5f} >= 0). STRICT INVARIANT: REJECT/REVERT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
