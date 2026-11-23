import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import TARGET_COLUMNS, get_stratified_group_folds, load_dataset
from src.features import FeatureExtractor
from src.metric import compute_log_loss
from src.models import build_ensemble_model

RE_WORDS = re.compile(r"\b\w+\b")


def compute_repetition_and_diversity(text: str):
    words = [w.lower() for w in RE_WORDS.findall(text)]
    n_words = len(words)
    if n_words < 5:
        return 0.0, 1.0, 0.0

    trigrams = [tuple(words[i:i+3]) for i in range(n_words - 2)]
    n_trigrams = len(trigrams)
    n_unique_trigrams = len(set(trigrams))
    rep_rate = 1.0 - (n_unique_trigrams / max(n_trigrams, 1))
    ttr = len(set(words)) / max(n_words, 1)
    sentences = max(len(re.split(r"[.!?\n]+", text)), 1)
    avg_sent_len = n_words / sentences
    return rep_rate, ttr, avg_sent_len


def main():
    print("=" * 70)
    print("EXP 03: COMBINED FEATURES + TEMPERATURE CALIBRATION")
    print("Target Benchmark: 1.03295")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)

    ext = FeatureExtractor()
    X_base = ext.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values

    # TF-IDF Cosine
    print("Extracting TF-IDF cosine alignment...")
    corpus = list(df["prompt_text"]) + list(df["response_a_text"]) + list(df["response_b_text"])
    tfidf = TfidfVectorizer(max_features=15000, sublinear_tf=True, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(corpus)
    p_vecs = tfidf.transform(df["prompt_text"])
    a_vecs = tfidf.transform(df["response_a_text"])
    b_vecs = tfidf.transform(df["response_b_text"])

    sim_pa = np.asarray(p_vecs.multiply(a_vecs).sum(axis=1)).flatten()
    sim_pb = np.asarray(p_vecs.multiply(b_vecs).sum(axis=1)).flatten()
    sim_ab = np.asarray(a_vecs.multiply(b_vecs).sum(axis=1)).flatten()

    diff_prompt_sim = sim_pa - sim_pb
    mean_prompt_sim = 0.5 * (sim_pa + sim_pb)
    sym_ab_sim = sim_ab

    # Repetition & Diversity
    print("Extracting repetition and diversity metrics...")
    stats_a = [compute_repetition_and_diversity(t) for t in df["response_a_text"]]
    stats_b = [compute_repetition_and_diversity(t) for t in df["response_b_text"]]
    rep_a, ttr_a, sent_a = zip(*stats_a)
    rep_b, ttr_b, sent_b = zip(*stats_b)

    diff_rep = np.array(rep_a) - np.array(rep_b)
    diff_ttr = np.array(ttr_a) - np.array(ttr_b)
    diff_sent = np.array(sent_a) - np.array(sent_b)
    sym_mean_ttr = 0.5 * (np.array(ttr_a) + np.array(ttr_b))

    X_all = np.column_stack([
        X_base,
        diff_prompt_sim,
        sym_ab_sim,
        mean_prompt_sim,
        diff_rep,
        diff_ttr,
        diff_sent,
        sym_mean_ttr
    ])
    all_feature_names = ext.feature_names_ + [
        "diff_tfidf_prompt_sim",
        "sym_tfidf_ab_sim",
        "sym_tfidf_mean_prompt",
        "diff_trigram_rep_rate",
        "diff_type_token_ratio",
        "diff_avg_sent_len",
        "sym_mean_ttr"
    ]
    print(f"Combined feature matrix: {X_all.shape} ({len(all_feature_names)} features)")

    # 5-Fold CV
    oof_preds = np.zeros((len(df), 3), dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(get_stratified_group_folds(df, n_splits=5, random_state=42)):
        X_train, y_train = X_all[train_idx], y[train_idx]
        X_val, y_val = X_all[val_idx], y_true_onehot[val_idx]

        model = build_ensemble_model(all_feature_names, lr_weight=0.60)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)
        oof_preds[val_idx] = pred
        fold_loss = compute_log_loss(y_val, pred)
        print(f"  Fold {fold+1}/5 Log Loss: {fold_loss:.5f}")

    raw_loss = compute_log_loss(y_true_onehot, oof_preds)
    print(f"\nRaw Combined Feature OOF Log Loss: {raw_loss:.5f} (vs 1.03295: {raw_loss - 1.03295:+.5f})")

    # Temperature and Tie-bias Calibration
    print("\nOptimizing In-Fold Temperature Calibration...")
    def calibrate_loss(params):
        temp, tie_scale = params
        eps = 1e-15
        p = np.clip(oof_preds, eps, 1.0 - eps)
        # Apply temperature
        log_p = np.log(p) / temp
        # Symmetric tie scale (A and B scaled equally, tie scaled by tie_scale)
        log_p[:, 2] += np.log(max(tie_scale, 1e-5))
        # Softmax normalize
        exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
        p_cal = exp_p / exp_p.sum(axis=1, keepdims=True)
        return compute_log_loss(y_true_onehot, p_cal)

    res = minimize(calibrate_loss, x0=[1.0, 1.0], bounds=[(0.5, 2.0), (0.5, 2.0)], method="L-BFGS-B")
    best_temp, best_tie_scale = res.x
    calibrated_loss = res.fun
    print(f"Optimal Temperature: {best_temp:.4f}, Tie Scale: {best_tie_scale:.4f}")
    print(f"Calibrated OOF Log Loss: {calibrated_loss:.5f} (Delta vs 1.03295: {calibrated_loss - 1.03295:+.5f})")

    print("\n" + "=" * 70)
    print(f"FINAL DECISION GATE:")
    print(f"Baseline Champion: 1.03775")
    print(f"Exp 02 Champion: 1.03295")
    print(f"Exp 03 Raw Combined: {raw_loss:.5f} (Delta: {raw_loss - 1.03295:+.5f})")
    print(f"Exp 03 Calibrated: {calibrated_loss:.5f} (Delta: {calibrated_loss - 1.03295:+.5f})")

    if calibrated_loss < 1.03295:
        print(f"RESULT: STRICT IMPROVEMENT CONFIRMED! Delta = {calibrated_loss - 1.03295:.5f}. PROCEED TO PROMOTE.")
    else:
        print(f"RESULT: NO IMPROVEMENT. STRICT INVARIANT: REJECT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
