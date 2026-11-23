import os
import sys
import json
import re
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, TARGET_COLUMNS
from features import FeatureExtractor, compute_text_stats, compute_pairwise_features

print("=" * 70)
print("COMPREHENSIVE 360-DEGREE ANOMALY, OUTLIER & EDGE CASE RESEARCH SCAN")
print("=" * 70)

train_csv = os.path.join(BASE_DIR, "data", "train.csv")
df = pd.read_csv(train_csv)
print(f"Total Rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# 1. Target Invariants & Multi-Label Checks
y_mat = df[TARGET_COLUMNS].values
row_sums = y_mat.sum(axis=1)
print(f"\n1. TARGET ANOMALIES:")
print(f"  Rows with sum != 1: {np.sum(row_sums != 1)} (min sum: {row_sums.min()}, max sum: {row_sums.max()})")
print(f"  Rows with sum == 0: {np.sum(row_sums == 0)}")
print(f"  Rows with sum > 1:  {np.sum(row_sums > 1)}")
p_a = df["winner_model_a"].mean()
p_b = df["winner_model_b"].mean()
p_tie = df["winner_tie"].mean()
print(f"  Class Distribution: P(A)={p_a:.4f}, P(B)={p_b:.4f}, P(Tie)={p_tie:.4f}")

# 2. Exact Duplicates & Exact Tie Anomalies
print(f"\n2. EXACT DUPLICATE / IDENTICAL RESPONSE ANOMALIES:")
exact_match = (df["response_a"] == df["response_b"])
n_exact = exact_match.sum()
print(f"  Rows where response_a == response_b: {n_exact}")
if n_exact > 0:
    exact_df = df[exact_match]
    print(f"  Target distribution when A == B: P(A)={exact_df['winner_model_a'].mean():.4f}, P(B)={exact_df['winner_model_b'].mean():.4f}, P(Tie)={exact_df['winner_tie'].mean():.4f}")

# 3. Missing, Null, Empty, and Whitespace Text
print(f"\n3. MISSING / NULL / WHITESPACE ANOMALIES:")
for col in ["prompt", "response_a", "response_b"]:
    null_count = df[col].isnull().sum()
    empty_count = (df[col].astype(str).str.strip() == "").sum()
    short_count = (df[col].astype(str).str.len() < 5).sum()
    print(f"  Col '{col}': Nulls={null_count}, Empty={empty_count}, Length < 5={short_count}")

# 4. Extreme Length Outliers
print(f"\n4. EXTREME LENGTH OUTLIERS:")
len_a = df["response_a"].astype(str).str.len()
len_b = df["response_b"].astype(str).str.len()
len_p = df["prompt"].astype(str).str.len()
print(f"  Prompt Length:   Min={len_p.min()}, Median={len_p.median()}, P99={len_p.quantile(0.99)}, Max={len_p.max()}")
print(f"  Response A Length: Min={len_a.min()}, Median={len_a.median()}, P99={len_a.quantile(0.99)}, Max={len_a.max()}")
print(f"  Response B Length: Min={len_b.min()}, Median={len_b.median()}, P99={len_b.quantile(0.99)}, Max={len_b.max()}")

ratio_ab = len_a / np.maximum(len_b, 1)
print(f"  Length Ratio A/B: Min={ratio_ab.min():.4f}, P01={ratio_ab.quantile(0.01):.4f}, P99={ratio_ab.quantile(0.99):.4f}, Max={ratio_ab.max():.4f}")

# Correlation between length ratio and target
a_longer = (len_a > len_b * 1.5)
b_longer = (len_b > len_a * 1.5)
print(f"  When A > 1.5x B ({a_longer.sum()} rows): P(A)={df.loc[a_longer, 'winner_model_a'].mean():.4f}, P(B)={df.loc[a_longer, 'winner_model_b'].mean():.4f}, P(Tie)={df.loc[a_longer, 'winner_tie'].mean():.4f}")
print(f"  When B > 1.5x A ({b_longer.sum()} rows): P(A)={df.loc[b_longer, 'winner_model_a'].mean():.4f}, P(B)={df.loc[b_longer, 'winner_model_b'].mean():.4f}, P(Tie)={df.loc[b_longer, 'winner_tie'].mean():.4f}")

# 5. Non-ASCII / Multilingual Discrepancies
print(f"\n5. MULTILINGUAL & NON-ASCII DETECTION:")
RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")
non_ascii_p = df["prompt"].astype(str).apply(lambda s: bool(RE_NON_ASCII.search(s))).sum()
non_ascii_a = df["response_a"].astype(str).apply(lambda s: bool(RE_NON_ASCII.search(s))).sum()
print(f"  Prompts with non-ASCII: {non_ascii_p} ({non_ascii_p/len(df)*100:.2f}%)")
print(f"  Responses with non-ASCII: {non_ascii_a} ({non_ascii_a/len(df)*100:.2f}%)")

# Check if non-ASCII causes zero word counts in RE_WORDS
def check_zero_words(text):
    words = re.findall(r"\b\w+\b", text)
    return len(words) == 0

zero_words_p = df["prompt"].astype(str).apply(check_zero_words).sum()
zero_words_a = df["response_a"].astype(str).apply(check_zero_words).sum()
zero_words_b = df["response_b"].astype(str).apply(check_zero_words).sum()
print(f"  Prompts yielding 0 words: {zero_words_p}")
print(f"  Response A yielding 0 words: {zero_words_a}")
print(f"  Response B yielding 0 words: {zero_words_b}")

# 6. Multi-Turn Conversation Edge Cases
print(f"\n6. MULTI-TURN CONVERSATIONS:")
df_loaded = load_dataset(train_csv, nrows=5000, is_test=False)
turn_counts_p = df_loaded["prompt_turns"].apply(len)
turn_counts_a = df_loaded["response_a_turns"].apply(len)
print(f"  Prompt Turn Counts: Min={turn_counts_p.min()}, Max={turn_counts_p.max()}, Multi-turn (>1): {(turn_counts_p > 1).sum()} ({(turn_counts_p > 1).mean()*100:.1f}%)")
print(f"  Response A Turn Counts: Min={turn_counts_a.min()}, Max={turn_counts_a.max()}, Multi-turn (>1): {(turn_counts_a > 1).sum()} ({(turn_counts_a > 1).mean()*100:.1f}%)")

# 7. Model ID & Dataset Leakage Check
print(f"\n7. MODEL NAMES & LEAKAGE IN TRAIN:")
if "model_a" in df.columns:
    print(f"  Unique Model A values: {df['model_a'].nunique()}")
    print(f"  Top 5 Models: {df['model_a'].value_counts().head(5).to_dict()}")
else:
    print("  'model_a' not in train.csv")

# 8. Test Data Structure
test_csv = os.path.join(BASE_DIR, "data", "test.csv")
df_test = pd.read_csv(test_csv)
print(f"\n8. TEST DATA COLUMNS & SAMPLES:")
print(f"  Test rows in sample: {len(df_test)}")
print(f"  Test columns: {df_test.columns.tolist()}")

print("\n" + "=" * 70)
print("EDGE CASE ANALYSIS COMPLETE")
print("=" * 70)
