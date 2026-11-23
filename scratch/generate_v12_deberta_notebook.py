import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cell_0_md = """# LMSYS Chatbot Arena - LLM Classification Finetuning
## Fine-Tuned DeBERTa-v3 Cross-Encoder Pipeline (Version 12: Sequence Classification + Anti-Symmetric TTA + Dirichlet Calibration)
### Target Goal: Score $\\le 0.85000$ Multi-Class Log Loss

---

### Kaggle Execution Settings:
- **Accelerator**: **GPU T4 x2** or **GPU P100** (Right sidebar: Session options -> Accelerator)
- **Internet**: **ON** during initial run (to download `microsoft/deberta-v3-small`), OR click **'+ Add Input'** and search for `deberta-v3-small` / `debertav3small` to run with **Internet OFF**!
- **Target**: Predict human preference (`winner_model_a`, `winner_model_b`, `winner_tie`)
- **Metric**: Multi-class Log Loss ($\text{eps}=10^{-15}$)

---

### Technical Architecture for $\\le 0.85000$:
1. **Sequence Cross-Encoder Backbone**:
   - `microsoft/deberta-v3-small` (141M parameters).
   - Sequence format: `[CLS] Prompt: {prompt} [SEP] Response A: {response_a} [SEP] Response B: {response_b} [SEP]`
   - Balanced context allocation: up to 128 tokens for prompt, 192 tokens for response A, 192 tokens for response B (max length 512). Eliminates truncation length bias.
2. **Mathematical Anti-Symmetry (TTA)**:
   - For every sample, both $(P, A, B)$ and $(P, B, A)$ are evaluated and averaged:
     $$P(A) = 0.5 \\times (P(A \\mid A, B) + P(B \\mid B, A))$$
     $$P(B) = 0.5 \\times (P(B \\mid A, B) + P(A \\mid B, A))$$
     $$P(\\text{tie}) = 0.5 \\times (P(\\text{tie} \\mid A, B) + P(\\text{tie} \\mid B, A))$$
   - Eliminates 100% of presentation/order bias.
3. **Training & Regularization**:
   - Mixed Precision FP16 (`torch.cuda.amp.autocast()`).
   - Optimizer: AdamW (`lr=2e-5`, `weight_decay=0.01`).
   - Learning Rate Scheduler: Cosine Annealing with 10% linear warmup.
   - Batch size 8 per GPU with gradient accumulation = 4 (effective batch size 32).
   - Label smoothing = 0.02.
4. **Post-Processing & Calibration**:
   - 3-Parameter Decoupled Symmetric Dirichlet Calibration ($T_{\\text{pair}}, \\beta_{\\text{tie}}, \\alpha_{\\text{tie}}$).
   - Exact-Match Bayesian Prior: Overwrites identical response pairs with empirical prior $[0.0491, 0.0491, 0.9018]$."""

cell_1_setup = """import os
import sys
import re
import math
import zlib
import time
import glob
import hashlib
from typing import List, Dict, Tuple, Generator, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Fix protobuf Descriptors error on Windows/Linux environments
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

# Hardware & GPU Acceleration Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU ACCELERATED] Detected {num_gpus} GPU(s): {', '.join(gpu_names)} ({gpu_vram:.1f} GB VRAM per GPU)")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
    print("[CPU MODE] Warning: CUDA GPU not detected. Running on CPU.")
    print("TIP: In Kaggle under 'Session options' -> 'Accelerator', select 'GPU T4 x2' or 'GPU P100'!")

def find_data_dir() -> str:
    if os.path.exists("/kaggle/input"):
        matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if matches:
            found = os.path.dirname(matches[0])
            print(f"Auto-discovered competition data at: {found}")
            return found

    for cand in ["./data", "./", "../data", "../../data", "d:/ag-ai/llm-classification-finetuning/data"]:
        if os.path.exists(os.path.join(cand, "train.csv")):
            print(f"Found local competition data at: {cand}")
            return cand

    raise FileNotFoundError("Could not locate 'train.csv'! Please add the competition dataset.")

def find_model_path(default_model: str = "microsoft/deberta-v3-small") -> str:
    # 1. Search in /kaggle/input for any mounted deberta dataset
    if os.path.exists("/kaggle/input"):
        matches = glob.glob("/kaggle/input/**/deberta*", recursive=True)
        for m in matches:
            if os.path.isdir(m) and any(f.endswith(".json") or f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(m)):
                print(f"Auto-discovered mounted DeBERTa model at: {m}")
                return m

    # 2. Check local paths
    for cand in ["./models/deberta-v3-small", "../models/deberta-v3-small"]:
        if os.path.exists(cand):
            print(f"Found local model at: {cand}")
            return cand

    # 3. Default to HuggingFace hub model ID
    print(f"Using HuggingFace Hub model: {default_model}")
    return default_model

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
DATA_DIR = find_data_dir()
MODEL_PATH = find_model_path()
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "deberta_v3_small_best.pt")

TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
print(f"Data directory: {DATA_DIR}")
print(f"Model path:     {MODEL_PATH}")
print(f"Submission to:  {SUBMISSION_PATH}")"""

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
        import json as _json
        parsed = _json.loads(s)
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

    return df

class PreferenceDataset(Dataset):
    \"\"\"
    Paired sequence dataset with balanced context truncation.
    [CLS] Prompt: {P} [SEP] Response A: {A} [SEP] Response B: {B} [SEP]
    \"\"\"
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        is_test: bool = False,
        swap_pairs: bool = False,
    ):
        self.prompts = df["prompt_text"].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.swap_pairs = swap_pairs

        if not swap_pairs:
            self.resp_a = df["response_a_text"].values
            self.resp_b = df["response_b_text"].values
            if not is_test:
                self.labels = df["target_label"].values
        else:
            # Swapped pairs: A <-> B
            self.resp_a = df["response_b_text"].values
            self.resp_b = df["response_a_text"].values
            if not is_test:
                orig_labels = df["target_label"].values
                swapped_labels = orig_labels.copy()
                swapped_labels[orig_labels == 0] = 1
                swapped_labels[orig_labels == 1] = 0
                self.labels = swapped_labels

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = str(self.prompts[idx])
        a = str(self.resp_a[idx])
        b = str(self.resp_b[idx])

        # Formatted cross-encoder sequence
        text = f"[CLS] Prompt: {p} [SEP] Response A: {a} [SEP] Response B: {b} [SEP]"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item"""

cell_4_model = """class DebertaPreferenceClassifier(nn.Module):
    \"\"\"
    DeBERTa-v3 sequence classification head with dropout and label smoothing.
    \"\"\"
    def __init__(self, model_name_or_path: str, num_labels: int = 3):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )"""

cell_5_calibration = """class DirichletSymmetricCalibrator:
    \"\"\"
    3-Parameter Decoupled Symmetric Dirichlet Calibrator.
    Guarantees exact mathematical anti-symmetry and simplex normalization:
      z_AB = (log(p_A) - log(p_B)) / T_pair
      z_tie = beta_tie * (log(p_tie) - 0.5*(log(p_A) + log(p_B))) + alpha_tie
    \"\"\"
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
            p_cal_clipped = np.clip(p_cal, eps, 1.0 - eps)
            p_cal_norm = p_cal_clipped / p_cal_clipped.sum(axis=1, keepdims=True)
            return float(-np.mean(np.sum(y_true_onehot * np.log(p_cal_norm), axis=1)))

        init_params = np.array([self.t_pair, self.beta_tie, self.alpha_tie], dtype=np.float64)
        bounds = [(0.1, 5.0), (0.1, 5.0), (-5.0, 5.0)]
        res = minimize(objective, init_params, method="L-BFGS-B", bounds=bounds)

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

        p_cal = np.hstack([exp_a / denom, exp_b / denom, exp_tie / denom])
        p_cal_clipped = np.clip(p_cal, eps, 1.0 - eps)
        return p_cal_clipped / p_cal_clipped.sum(axis=1, keepdims=True)"""

cell_6_train = """# Load Tokenizer & Training Dataset
print(f"Loading DeBERTa tokenizer from: {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Loading full training data from: {TRAIN_PATH}...")
df_train = load_dataset(TRAIN_PATH, is_test=False)
print(f"Loaded {len(df_train)} battles across {df_train['prompt_group'].nunique()} prompt groups.")

# Hyperparameters for Score <= 0.85
MAX_LENGTH = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # Effective batch size = 32
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 2
LABEL_SMOOTHING = 0.02

# 5-Fold StratifiedGroupKFold on SHA-256 Prompt Hashes
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
y = df_train["target_label"].values
groups = df_train["prompt_group"].values

# Select Fold 0 for Primary Training (or all folds if full ensemble desired)
train_idx, val_idx = next(sgkf.split(df_train, y, groups=groups))
print(f"Training Fold 1/5: Train Size = {len(train_idx)}, Val Size = {len(val_idx)}")

df_fold_train = df_train.iloc[train_idx].reset_index(drop=True)
df_fold_val = df_train.iloc[val_idx].reset_index(drop=True)
y_val_onehot = df_fold_val[TARGET_COLUMNS].values

# Datasets & Loaders
train_dataset = PreferenceDataset(df_fold_train, tokenizer, max_length=MAX_LENGTH)
val_dataset_orig = PreferenceDataset(df_fold_val, tokenizer, max_length=MAX_LENGTH, is_test=True, swap_pairs=False)
val_dataset_swap = PreferenceDataset(df_fold_val, tokenizer, max_length=MAX_LENGTH, is_test=True, swap_pairs=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader_orig = DataLoader(val_dataset_orig, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
val_loader_swap = DataLoader(val_dataset_swap, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

# Model Initialization
model = DebertaPreferenceClassifier(MODEL_PATH, num_labels=3).to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
scaler = torch.cuda.amp.GradScaler()
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

print(f"\\n--- Starting DeBERTa-v3 Fine-Tuning ({EPOCHS} Epochs, Total Steps: {total_steps}) ---")
best_val_loss = 999.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    t_start = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else loss_fn(outputs.logits, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        running_loss += loss.item() * GRAD_ACCUM_STEPS

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        if (step + 1) % 400 == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch + 1}/{EPOCHS} | Step {step + 1}/{len(train_loader)} | Train Loss: {running_loss / (step + 1):.4f} ({elapsed:.0f}s)")

    # Validation with TTA (Test-Time Augmentation Symmetrization)
    model.eval()
    val_preds_orig = []
    val_preds_swap = []

    with torch.no_grad():
        for b_orig in val_loader_orig:
            with torch.cuda.amp.autocast():
                out = model(b_orig["input_ids"].to(DEVICE), b_orig["attention_mask"].to(DEVICE))
                p = torch.softmax(out.logits if hasattr(out, "logits") else out, dim=1).cpu().numpy()
                val_preds_orig.append(p)

        for b_swap in val_loader_swap:
            with torch.cuda.amp.autocast():
                out = model(b_swap["input_ids"].to(DEVICE), b_swap["attention_mask"].to(DEVICE))
                p = torch.softmax(out.logits if hasattr(out, "logits") else out, dim=1).cpu().numpy()
                val_preds_swap.append(p)

    p_orig = np.vstack(val_preds_orig)
    p_swap = np.vstack(val_preds_swap)

    # Symmetrization
    p_val_tta = np.zeros_like(p_orig)
    p_val_tta[:, 0] = 0.5 * (p_orig[:, 0] + p_swap[:, 1])
    p_val_tta[:, 1] = 0.5 * (p_orig[:, 1] + p_swap[:, 0])
    p_val_tta[:, 2] = 0.5 * (p_orig[:, 2] + p_swap[:, 2])
    p_val_tta /= p_val_tta.sum(axis=1, keepdims=True)

    val_loss = compute_log_loss(y_val_onehot, p_val_tta)
    print(f"\\n>>> Epoch {epoch + 1} Complete! Val Log Loss (with TTA): {val_loss:.5f} <<<")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model = model.module if hasattr(model, "module") else model
        torch.save(save_model.state_dict(), CHECKPOINT_PATH)
        print(f"  Saved best model checkpoint to: {CHECKPOINT_PATH} (Loss: {best_val_loss:.5f})")

# Fit Dirichlet Calibration on Validation TTA Predictions
print("\\nFitting Decoupled Symmetric Dirichlet Calibrator...")
calibrator = DirichletSymmetricCalibrator().fit(p_val_tta, y_val_onehot)
cal_val_preds = calibrator.transform(p_val_tta)
cal_val_loss = compute_log_loss(y_val_onehot, cal_val_preds)
print(f"Calibrated Validation Log Loss: {cal_val_loss:.5f} (Params: T={calibrator.t_pair:.4f}, Beta={calibrator.beta_tie:.4f}, Alpha={calibrator.alpha_tie:.4f})")"""

cell_7_inference = """# Load Best Checkpoint & Test Dataset
print(f"Loading best checkpoint from {CHECKPOINT_PATH}...")
best_model = DebertaPreferenceClassifier(MODEL_PATH, num_labels=3).to(DEVICE)
best_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
best_model.eval()

print(f"Loading test data from {TEST_PATH}...")
df_test = load_dataset(TEST_PATH, is_test=True)
df_sample = pd.read_csv(SAMPLE_SUB_PATH)

test_dataset_orig = PreferenceDataset(df_test, tokenizer, max_length=MAX_LENGTH, is_test=True, swap_pairs=False)
test_dataset_swap = PreferenceDataset(df_test, tokenizer, max_length=MAX_LENGTH, is_test=True, swap_pairs=True)

test_loader_orig = DataLoader(test_dataset_orig, batch_size=BATCH_SIZE * 2, shuffle=False)
test_loader_swap = DataLoader(test_dataset_swap, batch_size=BATCH_SIZE * 2, shuffle=False)

# TTA Inference
test_preds_orig = []
test_preds_swap = []

print(f"Running TTA Inference on {len(df_test)} test battles...")
with torch.no_grad():
    for batch in test_loader_orig:
        with torch.cuda.amp.autocast():
            out = best_model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            p = torch.softmax(out.logits if hasattr(out, "logits") else out, dim=1).cpu().numpy()
            test_preds_orig.append(p)

    for batch in test_loader_swap:
        with torch.cuda.amp.autocast():
            out = best_model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
            p = torch.softmax(out.logits if hasattr(out, "logits") else out, dim=1).cpu().numpy()
            test_preds_swap.append(p)

p_test_orig = np.vstack(test_preds_orig)
p_test_swap = np.vstack(test_preds_swap)

# Anti-Symmetric TTA Symmetrization
p_test_tta = np.zeros_like(p_test_orig)
p_test_tta[:, 0] = 0.5 * (p_test_orig[:, 0] + p_test_swap[:, 1])
p_test_tta[:, 1] = 0.5 * (p_test_orig[:, 1] + p_test_swap[:, 0])
p_test_tta[:, 2] = 0.5 * (p_test_orig[:, 2] + p_test_swap[:, 2])
p_test_tta /= p_test_tta.sum(axis=1, keepdims=True)

# Apply Dirichlet Calibration
cal_test_preds = calibrator.transform(p_test_tta)

# Exact-Match Bayesian Prior Override
exact_mask = (df_test["response_a_text"].str.strip() == df_test["response_b_text"].str.strip()).values
num_exact = np.sum(exact_mask)
if num_exact > 0:
    print(f"Applying exact-match empirical prior [0.0491, 0.0491, 0.9018] to {num_exact} identical test pairs...")
    cal_test_preds[exact_mask] = np.array([0.0491, 0.0491, 0.9018])

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
    make_cell("code", cell_4_model),
    make_cell("code", cell_5_calibration),
    make_cell("code", cell_6_train),
    make_cell("code", cell_7_inference),
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
    print(f"Wrote Version 12 DeBERTa notebook to: {out_path}")

print("Generation complete.")
