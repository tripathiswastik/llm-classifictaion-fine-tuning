import os
import sys
import re
import numpy as np
import pandas as pd

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

    # 3-gram repetition rate
    trigrams = [tuple(words[i:i+3]) for i in range(n_words - 2)]
    n_trigrams = len(trigrams)
    n_unique_trigrams = len(set(trigrams))
    rep_rate = 1.0 - (n_unique_trigrams / max(n_trigrams, 1))

    # Type-Token Ratio (lexical diversity)
    ttr = len(set(words)) / max(n_words, 1)

    # Average sentence length (approximated by periods / linebreaks)
    sentences = max(len(re.split(r"[.!?\n]+", text)), 1)
    avg_sent_len = n_words / sentences

    return rep_rate, ttr, avg_sent_len


def main():
    print("=" * 70)
    print("EXP 02: REPETITION PENALTY & LEXICAL DIVERSITY BENCHMARK")
    print("Champion Benchmark: 1.03625 (Candidate A baseline: 1.03739)")
    print("=" * 70)

    data_dir = os.path.join(PROJECT_ROOT, "data")
    train_path = os.path.join(data_dir, "train.csv")
    df = load_dataset(train_path, nrows=10000, is_test=False)

    ext = FeatureExtractor()
    X_base = ext.fit_transform(df)
    y = df["target_label"].values
    y_true_onehot = df[TARGET_COLUMNS].values

    # Extract repetition and diversity features
    print("Extracting repetition and diversity metrics...")
    stats_a = [compute_repetition_and_diversity(t) for t in df["response_a_text"]]
    stats_b = [compute_repetition_and_diversity(t) for t in df["response_b_text"]]

    rep_a, ttr_a, sent_a = zip(*stats_a)
    rep_b, ttr_b, sent_b = zip(*stats_b)

    diff_rep = np.array(rep_a) - np.array(rep_b)
    diff_ttr = np.array(ttr_a) - np.array(ttr_b)
    diff_sent = np.array(sent_a) - np.array(sent_b)
    sym_mean_ttr = 0.5 * (np.array(ttr_a) + np.array(ttr_b))

    X_aug = np.column_stack([
        X_base,
        diff_rep,
        diff_ttr,
        diff_sent,
        sym_mean_ttr
    ])
    aug_feature_names = ext.feature_names_ + [
        "diff_trigram_rep_rate",
        "diff_type_token_ratio",
        "diff_avg_sent_len",
        "sym_mean_ttr"
    ]
    print(f"Augmented feature matrix: {X_aug.shape}")

    oof_preds = np.zeros((len(df), 3), dtype=np.float64)
    for fold, (train_idx, val_idx) in enumerate(get_stratified_group_folds(df, n_splits=5, random_state=42)):
        X_train, y_train = X_aug[train_idx], y[train_idx]
        X_val, y_val = X_aug[val_idx], y_true_onehot[val_idx]

        model = build_ensemble_model(aug_feature_names, lr_weight=0.60)
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)
        oof_preds[val_idx] = pred
        fold_loss = compute_log_loss(y_val, pred)
        print(f"  Fold {fold+1}/5 Log Loss: {fold_loss:.5f}")

    loss = compute_log_loss(y_true_onehot, oof_preds)
    delta_vs_champ = loss - 1.03775
    print("\n" + "=" * 70)
    print(f"EXP 02 OOF Log Loss: {loss:.5f} (Baseline was 1.03775, Delta: {delta_vs_champ:+.5f})")

    if loss < 1.03775:
        print(f"RESULT: STRICT IMPROVEMENT CONFIRMED ({delta_vs_champ:.5f}). PROCEED TO COMBINE.")
    else:
        print(f"RESULT: NO IMPROVEMENT OR DEGRADATION. STRICT INVARIANT: REJECT AND DISCARD.")
    print("=" * 70)


if __name__ == "__main__":
    main()
