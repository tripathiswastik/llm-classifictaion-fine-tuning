import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Read code blocks from existing source files to ensure exact parity
with open(os.path.join(BASE_DIR, "src", "metric.py"), "r", encoding="utf-8") as f:
    metric_code = f.read()

with open(os.path.join(BASE_DIR, "src", "data_loader.py"), "r", encoding="utf-8") as f:
    data_loader_code = f.read()

with open(os.path.join(BASE_DIR, "src", "features.py"), "r", encoding="utf-8") as f:
    features_code = f.read()

with open(os.path.join(BASE_DIR, "src", "models.py"), "r", encoding="utf-8") as f:
    models_code = f.read()

with open(os.path.join(BASE_DIR, "src", "calibration.py"), "r", encoding="utf-8") as f:
    calibration_code = f.read()

# Strip relative imports from source modules for self-contained notebook execution
def sanitize_code(code: str) -> str:
    code = re.sub(r"try:\s*\n\s*from \.features import [^\n]+\nexcept ImportError:\s*\n\s*from features import [^\n]+\n", "", code)
    code = re.sub(r"try:\s*\n\s*from \.metric import [^\n]+\nexcept ImportError:\s*\n\s*from metric import [^\n]+\n", "", code)
    return code

features_code_clean = sanitize_code(features_code)
models_code_clean = sanitize_code(models_code)
calibration_code_clean = sanitize_code(calibration_code)

cell_0_md = """# LMSYS Chatbot Arena - LLM Classification Finetuning
## Self-Contained Champion Inference & Training Pipeline (Version 10: 182-Feature Anomaly-Hardened Dual LSA + Rebalanced Tri-Ensemble + Decoupled Dirichlet Calibration + Exact-Match Empirical Prior)

### Kaggle Execution Settings:
- **Accelerator**: **GPU T4 x2** or **GPU P100** (Session options -> Accelerator)
- **Internet**: **OFF** (Required by competition rules)
- **Target**: Predict human preference (`winner_model_a`, `winner_model_b`, `winner_tie`)
- **Metric**: Multi-class Log Loss (eps=1e-15)
- **Architecture**:
  - **182 Anti-Symmetric Features**: 96 base tabular + 51 Word LSA components + 35 Char LSA components via TruncatedSVD.
  - **Anomaly Hardening**: Fixed 0-word Flesch Reading Ease edge cases, engineered multi-turn closing turn signals (`diff_final_len`, `diff_final_log_len`, `diff_final_jacc`, `sym_is_multi_turn`, `sym_num_turns`), non-linear length preference cliffs (`diff_len_ratio_15`, `diff_len_ratio_20`, `diff_len_ratio_30`, `sym_extreme_len_imbalance`), and exact duplicate detector (`sym_is_exact_match`).
  - **Rebalanced Tri-Ensemble**: 0.38 Logistic Regression (L2 C=0.08) + 0.52 HistGradientBoosting (150 trees, max_leaf_nodes=31, min_samples_leaf=50) + 0.10 GPU PyTorch Neural Network (temperature=1.0 unflattened logits).
  - **Calibration**: 3-Parameter Decoupled Symmetric Dirichlet Calibrator (temperature $T_{pair}$, tie scale $\\beta$, tie intercept $\\alpha$).
  - **Inference Post-Processing**: Exact-Match Bayesian Prior override ($[0.0491, 0.0491, 0.9018]$) for 100% byte-identical response pairs.
- **Validation**: 5-Fold StratifiedGroupKFold on SHA-256 prompt hashes (0.0000 prompt leakage).
- **Kaggle Public Leaderboard Track Record**:
  - Marginal Baseline: `1.09410`
  - Version 1 (44 feats, CPU): `1.02685`
  - Version 2 (55 feats, CPU): `1.02559`
  - Version 3 (64 feats, CPU): `1.02335`
  - Version 4 (64 feats, GPU Tri-Ensemble): `1.02125`
  - Version 5 (72 feats, GPU Tri-Ensemble): `1.01163`
  - Version 7 (158 feats, Dual LSA + Dirichlet): `1.00628` (All-Time Kaggle Public LB Champion)
  - Version 8 (158 feats, Margin-Coupled Calibrator): `1.00639`
  - Version 9 (170 feats, Over-flattened MLP 0.225): `1.00787`
  - **Version 10 (182 feats, Anomaly-Hardened + Exact-Match Prior + Rebalanced Ensemble): Target `<= 1.00000` Public Log Loss**"""

cell_1_setup = """import os
import sys
import re
import math
import zlib
from collections import Counter
import json
import glob
import hashlib
from typing import List, Dict, Tuple, Generator, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Hardware & GPU Acceleration Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU ACCELERATED] Running on NVIDIA GPU: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
    print("[CPU MODE] Warning: CUDA GPU not detected. Running on CPU.")
    print("TIP: In the right sidebar under 'Session options' -> 'Accelerator', select 'GPU T4 x2' or 'GPU P100' for 5-10x faster training!")

def find_data_dir() -> str:
    if os.path.exists("/kaggle/input"):
        print("Mounted items in /kaggle/input:", os.listdir("/kaggle/input"))
        matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if matches:
            found = os.path.dirname(matches[0])
            print(f"Auto-discovered competition data at: {found}")
            return found

    for cand in ["./data", "./", "../data", "../../data", "d:/ag-ai/llm-classification-finetuning/data"]:
        cand_path = os.path.join(cand, "train.csv")
        if os.path.exists(cand_path):
            print(f"Found local competition data at: {cand}")
            return cand

    available = os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "No /kaggle/input"
    raise FileNotFoundError(
        f"Could not locate 'train.csv'!\\n"
        f"Available in /kaggle/input: {available}\\n\\n"
        f"ACTION REQUIRED IN KAGGLE:\\n"
        f"Please click '+ Add Input' in the right-hand panel of your Kaggle notebook, search for 'llm-classification-finetuning', and add the competition dataset."
    )

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
DATA_DIR = find_data_dir()
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
print(f"Data directory: {DATA_DIR}")
print(f"Submission target: {SUBMISSION_PATH}")"""

cell_2_metric = """def compute_log_loss(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, eps: float = 1e-15) -> float:
    \"\"\"
    Computes exact Kaggle competition multi-class log loss with eps-clipping.
    \"\"\"
    p_clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
    p_normalized = p_clipped / p_clipped.sum(axis=1, keepdims=True)
    sample_losses = -np.sum(y_true_onehot * np.log(p_normalized), axis=1)
    return float(np.mean(sample_losses))"""

cell_3_loader = """def parse_conversation_turns(raw_val: Union[str, list, None]) -> List[str]:
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return []
    if isinstance(raw_val, list):
        return [str(x) for x in raw_val if x is not None]
    if not isinstance(raw_val, str):
        return [str(raw_val)]

    s = raw_val.strip()
    if not s:
        return []

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None]
        return [str(parsed)]
    except Exception:
        pass

    if s.startswith("[") and s.endswith("]"):
        items = re.findall(r'\"(.*?)\"', s)
        if items:
            return items
    return [s]

def flatten_turns(turns: List[str]) -> str:
    return " \\n\\n ".join(turns)

def load_dataset(csv_path: str, nrows: Optional[int] = None, is_test: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    if not is_test:
        for col in TARGET_COLUMNS:
            assert col in df.columns, f"Missing required target column '{col}' in train data"

    df["prompt_turns"] = df["prompt"].apply(parse_conversation_turns)
    df["response_a_turns"] = df["response_a"].apply(parse_conversation_turns)
    df["response_b_turns"] = df["response_b"].apply(parse_conversation_turns)

    df["prompt_text"] = df["prompt_turns"].apply(flatten_turns)
    df["response_a_text"] = df["response_a_turns"].apply(flatten_turns)
    df["response_b_text"] = df["response_b_turns"].apply(flatten_turns)

    df["prompt_group"] = df["prompt_text"].apply(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())

    if not is_test:
        df["target_label"] = np.argmax(df[TARGET_COLUMNS].values, axis=1)

    return df"""

cell_4_features = features_code_clean

cell_5_models = models_code_clean

cell_6_calibration = calibration_code_clean

cell_7_train = """# Load Training Data
print(f"Loading training data from {TRAIN_PATH}...")
df_train = load_dataset(TRAIN_PATH, is_test=False)
print(f"Loaded {len(df_train)} training battles across {df_train['prompt_group'].nunique()} prompt groups.")

# Extract 182 Anomaly-Hardened Features
print("Extracting 182-feature matrix (96 base + 51 Word LSA + 35 Char LSA semantic features)...")
extractor = FeatureExtractor(
    use_tfidf=True,
    max_tfidf_features=15000,
    use_lsa=True,
    n_lsa_components=24,
    use_char_lsa=True,
    n_char_lsa_components=16,
)
X = extractor.fit_transform(df_train)
y = df_train["target_label"].values
y_true_onehot = df_train[TARGET_COLUMNS].values
print(f"Extracted feature matrix shape: {X.shape} ({X.shape[1]} features)")

# 5-Fold StratifiedGroupKFold
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros((len(df_train), 3), dtype=np.float64)
fold_models = []

print(f"\\n--- Running {n_splits}-Fold StratifiedGroupKFold CV ---")
for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_train, y, groups=df_train["prompt_group"].values)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val_onehot = X[val_idx], y_true_onehot[val_idx]

    model = TriEnsemblePreferenceModel(
        extractor.feature_names_,
        lr_weight=0.38,
        gbdt_weight=0.52,
        gpu_pt_weight=0.10,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict_proba(X_val)
    oof_preds[val_idx] = val_pred
    fold_models.append(model)

    fold_loss = compute_log_loss(y_val_onehot, val_pred)
    print(f"  Fold {fold + 1}/{n_splits} Log Loss: {fold_loss:.5f} (Val size: {len(val_idx)})")

raw_oof_loss = compute_log_loss(y_true_onehot, oof_preds)
print(f"\\nRaw Rebalanced Tri-Ensemble OOF Log Loss: {raw_oof_loss:.5f}")

# Fit Decoupled Symmetric Dirichlet Calibrator
print("Optimizing Decoupled Symmetric Dirichlet Calibration...")
calibrator = DirichletSymmetricCalibrator()
calibrator.fit(oof_preds, y_true_onehot)
cal_oof_preds = calibrator.transform(oof_preds)

# Apply Exact-Match Anomaly Override to OOF
exact_train_mask = (df_train["response_a_text"].str.strip() == df_train["response_b_text"].str.strip()).values
if np.any(exact_train_mask):
    cal_oof_preds[exact_train_mask] = np.array([0.0491, 0.0491, 0.9018])
    print(f"Applied exact-match empirical prior to {np.sum(exact_train_mask)} training pairs.")

cal_oof_loss = compute_log_loss(y_true_onehot, cal_oof_preds)
print(f"Optimal Dirichlet Parameters: T_pair={calibrator.t_pair:.4f}, Beta_tie={calibrator.beta_tie:.4f}, Alpha_tie={calibrator.alpha_tie:.4f}")
print(f"Calibrated Ensemble OOF Log Loss: {cal_oof_loss:.5f}")"""

cell_8_inference = """# Load Test Data
print(f"\\nLoading test data from {TEST_PATH}...")
df_test = load_dataset(TEST_PATH, is_test=True)
df_sample = pd.read_csv(SAMPLE_SUB_PATH)

X_test = extractor.transform(df_test)
print(f"Transformed test features: {X_test.shape}")

# Average predictions from all fold models
preds_list = [m.predict_proba(X_test) for m in fold_models]
avg_preds = np.mean(preds_list, axis=0)
avg_preds = avg_preds / avg_preds.sum(axis=1, keepdims=True)

# Apply Decoupled Dirichlet Calibration
cal_test_preds = calibrator.transform(avg_preds)

# Apply Exact-Match Bayesian Prior Anomaly Override
exact_test_mask = (df_test["response_a_text"].str.strip() == df_test["response_b_text"].str.strip()).values
num_exact = np.sum(exact_test_mask)
if num_exact > 0:
    print(f"Applying exact-match empirical prior [0.0491, 0.0491, 0.9018] to {num_exact} identical test pairs...")
    cal_test_preds[exact_test_mask] = np.array([0.0491, 0.0491, 0.9018])

# Format Submission DataFrame
submission = pd.DataFrame({
    "id": df_test["id"].values,
    "winner_model_a": cal_test_preds[:, 0],
    "winner_model_b": cal_test_preds[:, 1],
    "winner_tie": cal_test_preds[:, 2],
})

# Rigorous Invariant Verification Assertions
assert list(submission.columns) == list(df_sample.columns), "Column mismatch"
assert len(submission) == len(df_test), f"Row count mismatch: {len(submission)} vs {len(df_test)}"
assert not submission.isnull().any().any(), "Submission contains nulls"
prob_sums = submission[TARGET_COLUMNS].sum(axis=1).values
np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5, err_msg="Probabilities do not sum to 1.0")

# Write Submission CSV
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\\nSubmission verified and successfully written to {SUBMISSION_PATH} ({len(submission)} rows).")
print(submission.head())"""

def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.splitlines()]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": lines,
        "execution_count": None if cell_type == "code" else None,
    }

cells = [
    make_cell("markdown", cell_0_md),
    make_cell("code", cell_1_setup),
    make_cell("code", cell_2_metric),
    make_cell("code", cell_3_loader),
    make_cell("code", cell_4_features),
    make_cell("code", cell_5_models),
    make_cell("code", cell_6_calibration),
    make_cell("code", cell_7_train),
    make_cell("code", cell_8_inference),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.7.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for out_path in [
    os.path.join(BASE_DIR, "submission_notebook.ipynb"),
    os.path.join(BASE_DIR, "submissions", "submission_notebook.ipynb"),
]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote Version 10 notebook to: {out_path}")

print("Generation complete.")
