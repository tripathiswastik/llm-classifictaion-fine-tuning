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

# Lexical Patterns
RE_WORDS = re.compile(r"\b\w+\b")
RE_FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|we|us|our|ours)\b", re.IGNORECASE)
RE_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.IGNORECASE)
RE_COMPLEX_PUNCT = re.compile(r"[;—\(\)\[\]\{\}]")

# English Stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
}


def compute_exp07_features(prompt: str, resp_a: str, resp_b: str):
    words_p = [w.lower() for w in RE_WORDS.findall(prompt)]
    words_a = [w.lower() for w in RE_WORDS.findall(resp_a)]
    words_b = [w.lower() for w in RE_WORDS.findall(resp_b)]

    # First and second person counts
    fp_a = len(RE_FIRST_PERSON.findall(resp_a))
    fp_b = len(RE_FIRST_PERSON.findall(resp_b))
    diff_first_person = float(fp_a - fp_b)
    sym_first_person = 0.5 * float(fp_a + fp_b)

    sp_a = len(RE_SECOND_PERSON.findall(resp_a))
    sp_b = len(RE_SECOND_PERSON.findall(resp_b))
    diff_second_person = float(sp_a - sp_b)
    sym_second_person = 0.5 * float(sp_a + sp_b)

    # Complex punctuation
    cp_a = len(RE_COMPLEX_PUNCT.findall(resp_a))
    cp_b = len(RE_COMPLEX_PUNCT.findall(resp_b))
    diff_complex_punct = float(cp_a - cp_b)
    sym_complex_punct = 0.5 * float(cp_a + cp_b)

    # Content word overlap (ignoring stopwords)
    content_p = set(w for w in words_p if w not in STOPWORDS)
    content_a = set(w for w in words_a if w not in STOPWORDS)
    content_b = set(w for w in words_b if w not in STOPWORDS)

    if content_p:
        cov_pa = len(content_p & content_a) / len(content_p)
        cov_pb = len(content_p & content_b) / len(content_p)
        union_pa = max(len(content_p | content_a), 1)
        union_pb = max(len(content_p | content_b), 1)
        jacc_pa = len(content_p & content_a) / union_pa
        jacc_pb = len(content_p & content_b) / union_pb
    else:
        cov_pa = cov_pb = jacc_pa = jacc_pb = 0.0

    diff_content_cov = float(cov_pa - cov_pb)
    diff_content_jacc = float(jacc_pa - jacc_pb)
    sym_mean_content_cov = 0.5 * float(cov_pa + cov_pb)

    return {
        "diff_first_person": diff_first_person,
        "sym_first_person": sym_first_person,
        "diff_second_person": diff_second_person,
        "sym_second_person": sym_second_person,
        "diff_complex_punct": diff_complex_punct,
        "sym_complex_punct": sym_complex_punct,
        "diff_content_cov": diff_content_cov,
        "diff_content_jacc": diff_content_jacc,
        "sym_mean_content_cov": sym_mean_content_cov,
    }


def main():
    print("=" * 70)
    print("EXP 07: PRONOUN PERSPECTIVE, COMPLEX PUNCTUATION & CONTENT OVERLAP")
    print("Champion 10k Benchmark to Beat: 1.02890")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)

    print("Extracting current 55 production features...")
    extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000)
    X_base = extractor.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values

    print("Extracting candidate Exp 07 features...")
    new_feats_list = []
    p_texts = df["prompt_text"].values
    a_texts = df["response_a_text"].values
    b_texts = df["response_b_text"].values

    for p, a, b in zip(p_texts, a_texts, b_texts):
        new_feats_list.append(compute_exp07_features(p, a, b))

    new_feat_names = list(new_feats_list[0].keys())
    X_new = np.array([[row[k] for k in new_feat_names] for row in new_feats_list], dtype=np.float32)

    X_combined = np.hstack([X_base, X_new])
    all_feature_names = extractor.feature_names_ + new_feat_names
    print(f"Total candidate features: {X_combined.shape[1]} (55 base + {len(new_feat_names)} new = {len(all_feature_names)})")

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

        p_blend = 0.40 * oof_lr[val_idx] + 0.60 * oof_gbdt[val_idx]
        p_blend /= p_blend.sum(axis=1, keepdims=True)
        fold_loss = compute_log_loss(y_true_onehot[val_idx], p_blend)
        print(f"  Fold {fold+1}/5 Log Loss (0.40/0.60 blend): {fold_loss:.5f}")

    raw_blend = 0.40 * oof_lr + 0.60 * oof_gbdt
    raw_blend /= raw_blend.sum(axis=1, keepdims=True)
    raw_loss = compute_log_loss(y_true_onehot, raw_blend)
    print(f"\nRaw Candidate Blend OOF Loss: {raw_loss:.5f} (vs 1.02890: {raw_loss - 1.02890:+.5f})")

    # Blend weight optimization
    def blend_obj(alpha):
        a = float(alpha[0])
        p = a * oof_lr + (1.0 - a) * oof_gbdt
        p /= p.sum(axis=1, keepdims=True)
        return compute_log_loss(y_true_onehot, p)

    res_alpha = minimize(blend_obj, x0=[0.40], bounds=[(0.0, 1.0)], method="L-BFGS-B")
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
    print(f"Delta vs Champion 1.02890: {cal_loss - 1.02890:+.5f}")

    print("\n" + "=" * 70)
    print("DECISION GATE EVALUATION:")
    print(f"Champion 10k Benchmark: 1.02890")
    print(f"Exp 07 Candidate Loss:  {cal_loss:.5f}")
    if cal_loss < 1.02890:
        delta = cal_loss - 1.02890
        print(f"RESULT: STRICT IMPROVEMENT CONFIRMED! Delta = {delta:.5f} < 0. PROCEED TO PROMOTE.")
    else:
        delta = cal_loss - 1.02890
        print(f"RESULT: NO IMPROVEMENT (Delta = {delta:+.5f} >= 0). STRICT INVARIANT: REJECT/REVERT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
