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

RE_TABLE = re.compile(r"(?:^|\n)\s*\|.*?\|.*?\|(?:\n\s*\|[-:\s|]+\|)", re.MULTILINE)
RE_SYNTAX_CODE = re.compile(r"```[a-zA-Z0-9_+#-]+")
RE_SYCOPHANCY = re.compile(
    r"\b(great question|good question|you are right|absolutely right|certainly|delighted to help|happy to assist|pleasure to help|as an expert)\b",
    re.IGNORECASE,
)
RE_CODE_PROMPT = re.compile(r"\b(code|python|java|c\+\+|javascript|function|algorithm|implement|script|sql|html|css|bug|error|exception)\b", re.IGNORECASE)
RE_MATH_PROMPT = re.compile(r"\b(calculate|math|equation|solve|integral|derivative|formula|probability|matrix|algebra|theorem)\b", re.IGNORECASE)
RE_COMPARE_PROMPT = re.compile(r"\b(compare|comparison|versus|vs|table|difference between|pros and cons)\b", re.IGNORECASE)


def compute_exp05_features(prompt: str, resp_a: str, resp_b: str):
    # Table presence
    table_a = float(bool(RE_TABLE.search(resp_a)))
    table_b = float(bool(RE_TABLE.search(resp_b)))
    diff_table = table_a - table_b
    sym_table = 0.5 * (table_a + table_b)

    # Syntax-highlighted code fences
    syn_code_a = float(len(RE_SYNTAX_CODE.findall(resp_a)))
    syn_code_b = float(len(RE_SYNTAX_CODE.findall(resp_b)))
    diff_syn_code = syn_code_a - syn_code_b
    sym_syn_code = 0.5 * (syn_code_a + syn_code_b)

    # Sycophancy / throat-clearing
    syco_a = float(len(RE_SYCOPHANCY.findall(resp_a)))
    syco_b = float(len(RE_SYCOPHANCY.findall(resp_b)))
    diff_syco = syco_a - syco_b
    sym_syco = 0.5 * (syco_a + syco_b)

    # Prompt Intent Classification
    is_code_p = float(bool(RE_CODE_PROMPT.search(prompt)))
    is_math_p = float(bool(RE_MATH_PROMPT.search(prompt)))
    is_compare_p = float(bool(RE_COMPARE_PROMPT.search(prompt)))

    # Anti-symmetric Prompt Interactions (Symmetric prompt indicator * Anti-symmetric difference)
    code_blocks_a = float(resp_a.count("```") // 2)
    code_blocks_b = float(resp_b.count("```") // 2)
    diff_code = code_blocks_a - code_blocks_b

    latex_a = float(resp_a.count("$") // 2)
    latex_b = float(resp_b.count("$") // 2)
    diff_latex = latex_a - latex_b

    diff_prompt_code_inter = is_code_p * diff_code
    diff_prompt_math_inter = is_math_p * diff_latex
    diff_prompt_table_inter = is_compare_p * diff_table

    return {
        "diff_markdown_table": diff_table,
        "sym_markdown_table": sym_table,
        "diff_syntax_code_blocks": diff_syn_code,
        "sym_syntax_code_blocks": sym_syn_code,
        "diff_sycophancy": diff_syco,
        "sym_sycophancy": sym_syco,
        "diff_prompt_code_inter": diff_prompt_code_inter,
        "diff_prompt_math_inter": diff_prompt_math_inter,
        "diff_prompt_table_inter": diff_prompt_table_inter,
        "sym_is_code_prompt": is_code_p,
        "sym_is_math_prompt": is_math_p,
    }


def main():
    print("=" * 70)
    print("EXP 05: PROMPT INTERACTIONS, TABLES, SYNTAX CODE & SYCOPHANCY")
    print("Current 10k Benchmark Champion to Beat: 1.02991")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)

    print("Extracting current 44 production features...")
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X_base = extractor.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values

    print("Extracting candidate Exp 05 features...")
    new_feats_list = []
    p_texts = df["prompt_text"].values
    a_texts = df["response_a_text"].values
    b_texts = df["response_b_text"].values

    for p, a, b in zip(p_texts, a_texts, b_texts):
        new_feats_list.append(compute_exp05_features(p, a, b))

    new_feat_names = list(new_feats_list[0].keys())
    X_new = np.array([[row[k] for k in new_feat_names] for row in new_feats_list], dtype=np.float32)

    X_combined = np.hstack([X_base, X_new])
    all_feature_names = extractor.feature_names_ + new_feat_names
    print(f"Total candidate features: {X_combined.shape[1]} (44 base + {len(new_feat_names)} new)")

    # 5-Fold StratifiedGroupKFold
    print("\nRunning 5-Fold StratifiedGroupKFold CV...")
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

        p_blend = 0.46 * oof_lr[val_idx] + 0.54 * oof_gbdt[val_idx]
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        fold_loss = compute_log_loss(y_true_onehot[val_idx], p_blend)
        print(f"  Fold {fold+1}/5 Log Loss (0.46/0.54 blend): {fold_loss:.5f}")

    raw_blend = 0.46 * oof_lr + 0.54 * oof_gbdt
    raw_blend /= raw_blend.sum(axis=1, keepdims=True)
    raw_loss = compute_log_loss(y_true_onehot, raw_blend)
    print(f"\nRaw Candidate Blend OOF Loss: {raw_loss:.5f} (vs 1.02991: {raw_loss - 1.02991:+.5f})")

    # Blend weight optimization
    def blend_obj(alpha):
        a = float(alpha[0])
        p = a * oof_lr + (1.0 - a) * oof_gbdt
        p /= p.sum(axis=1, keepdims=True)
        return compute_log_loss(y_true_onehot, p)

    res_alpha = minimize(blend_obj, x0=[0.46], bounds=[(0.0, 1.0)], method="L-BFGS-B")
    best_alpha = float(res_alpha.x[0])
    best_blend = best_alpha * oof_lr + (1.0 - best_alpha) * oof_gbdt
    best_blend /= best_blend.sum(axis=1, keepdims=True)
    tuned_loss = compute_log_loss(y_true_onehot, best_blend)
    print(f"Optimal LR Weight: {best_alpha:.4f} (GBDT: {1.0 - best_alpha:.4f}), Tuned Loss: {tuned_loss:.5f}")

    # Calibration
    calibrator = TemperatureCalibrator()
    calibrator.fit(best_blend, y_true_onehot)
    cal_preds = calibrator.transform(best_blend)
    cal_loss = compute_log_loss(y_true_onehot, cal_preds)
    print(f"Calibrated Loss: {cal_loss:.5f} (T={calibrator.temperature:.4f}, TieScale={calibrator.tie_scale:.4f})")
    print(f"Delta vs Champion 1.02991: {cal_loss - 1.02991:+.5f}")

    print("\n" + "=" * 70)
    print("DECISION GATE EVALUATION:")
    print(f"Champion 10k Benchmark: 1.02991")
    print(f"Exp 05 Candidate Loss:  {cal_loss:.5f}")
    if cal_loss < 1.02991:
        delta = cal_loss - 1.02991
        print(f"RESULT: STRICT IMPROVEMENT CONFIRMED! Delta = {delta:.5f} < 0. PROCEED TO PROMOTE.")
    else:
        delta = cal_loss - 1.02991
        print(f"RESULT: NO IMPROVEMENT (Delta = {delta:+.5f} >= 0). STRICT INVARIANT: REJECT/REVERT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
