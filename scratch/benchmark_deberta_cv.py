import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import get_cosine_schedule_with_warmup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, TARGET_COLUMNS
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

def prepare_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 256,
    max_p: int = 64,
    max_a: int = 96,
    max_b: int = 96,
    is_test: bool = False,
    swap_pairs: bool = False,
) -> TensorDataset:
    prompts = df["prompt_text"].values
    if not swap_pairs:
        resp_a = df["response_a_text"].values
        resp_b = df["response_b_text"].values
        labels = df["target_label"].values if not is_test else None
    else:
        resp_a = df["response_b_text"].values
        resp_b = df["response_a_text"].values
        if not is_test:
            orig = df["target_label"].values
            swapped = orig.copy()
            swapped[orig == 0] = 1
            swapped[orig == 1] = 0
            labels = swapped

    cls_id = tokenizer.cls_token_id or 1
    sep_id = tokenizer.sep_token_id or 2
    pad_id = tokenizer.pad_token_id or 0

    n = len(df)
    all_ids = np.full((n, max_length), pad_id, dtype=np.int64)
    all_masks = np.zeros((n, max_length), dtype=np.int64)

    for i in range(n):
        p_ids = tokenizer.encode(f"Prompt: {prompts[i]}", truncation=True, max_length=max_p, add_special_tokens=False)
        a_ids = tokenizer.encode(f"Response A: {resp_a[i]}", truncation=True, max_length=max_a, add_special_tokens=False)
        b_ids = tokenizer.encode(f"Response B: {resp_b[i]}", truncation=True, max_length=max_b, add_special_tokens=False)

        seq = [cls_id] + p_ids + [sep_id] + a_ids + [sep_id] + b_ids + [sep_id]
        if len(seq) > max_length:
            seq = seq[:max_length]
        seq_len = len(seq)
        all_ids[i, :seq_len] = seq
        all_masks[i, :seq_len] = 1

    t_ids = torch.from_numpy(all_ids)
    t_masks = torch.from_numpy(all_masks)

    if not is_test:
        t_labels = torch.from_numpy(labels).long()
        return TensorDataset(t_ids, t_masks, t_labels)
    else:
        return TensorDataset(t_ids, t_masks)

# Load 1200 rows (1000 train, 200 val)
train_csv = os.path.join(BASE_DIR, "data", "train.csv")
df_all = load_dataset(train_csv, nrows=1200, is_test=False)
df_train = df_all.iloc[:1000].reset_index(drop=True)
df_val = df_all.iloc[1000:].reset_index(drop=True)
y_val_onehot = df_val[TARGET_COLUMNS].values

model_name = "microsoft/deberta-v3-small"
print(f"Loading tokenizer & model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Pre-tokenizing train & val datasets...")
t0 = time.time()
train_ds = prepare_tokenized_dataset(df_train, tokenizer, max_length=256)
val_ds_orig = prepare_tokenized_dataset(df_val, tokenizer, max_length=256, is_test=True, swap_pairs=False)
val_ds_swap = prepare_tokenized_dataset(df_val, tokenizer, max_length=256, is_test=True, swap_pairs=True)
print(f"Tokenization complete in {time.time() - t0:.2f}s.")

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader_orig = DataLoader(val_ds_orig, batch_size=16, shuffle=False)
val_loader_swap = DataLoader(val_ds_swap, batch_size=16, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)

print("\n--- Starting 2-Epoch Local DeBERTa Validation Benchmark ---")
for epoch in range(2):
    model.train()
    running_loss = 0.0
    t_ep = time.time()
    for step, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if outputs.loss is not None else loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # Validation with TTA
    model.eval()
    val_preds_orig = []
    val_preds_swap = []
    with torch.no_grad():
        for input_ids, attention_mask in val_loader_orig:
            with torch.cuda.amp.autocast():
                out = model(input_ids.to(device), attention_mask.to(device))
                p = torch.softmax(out.logits, dim=1).cpu().numpy()
                val_preds_orig.append(p)

        for input_ids, attention_mask in val_loader_swap:
            with torch.cuda.amp.autocast():
                out = model(input_ids.to(device), attention_mask.to(device))
                p = torch.softmax(out.logits, dim=1).cpu().numpy()
                val_preds_swap.append(p)

    p_orig = np.vstack(val_preds_orig)
    p_swap = np.vstack(val_preds_swap)

    # Symmetrization
    p_tta = np.zeros_like(p_orig)
    p_tta[:, 0] = 0.5 * (p_orig[:, 0] + p_swap[:, 1])
    p_tta[:, 1] = 0.5 * (p_orig[:, 1] + p_swap[:, 0])
    p_tta[:, 2] = 0.5 * (p_orig[:, 2] + p_swap[:, 2])
    p_tta /= p_tta.sum(axis=1, keepdims=True)

    raw_val_loss = compute_log_loss(y_val_onehot, p_orig)
    tta_val_loss = compute_log_loss(y_val_onehot, p_tta)
    ep_time = time.time() - t_ep
    print(f"Epoch {epoch+1}/2 ({ep_time:.1f}s) | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss (Raw): {raw_val_loss:.5f} | Val Loss (TTA): {tta_val_loss:.5f}")

# Calibrate
calibrator = DirichletSymmetricCalibrator().fit(p_tta, y_val_onehot)
p_cal = calibrator.transform(p_tta)
cal_val_loss = compute_log_loss(y_val_onehot, p_cal)
print(f"\nFinal Calibrated Val Log Loss: {cal_val_loss:.5f} (Improvement over raw: {raw_val_loss - cal_val_loss:.5f})")
print(f"Calibrator Params: T_pair={calibrator.t_pair:.4f}, Beta_tie={calibrator.beta_tie:.4f}, Alpha_tie={calibrator.alpha_tie:.4f}")
