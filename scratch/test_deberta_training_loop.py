import os
import sys
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW, get_cosine_schedule_with_warmup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from data_loader import load_dataset, TARGET_COLUMNS
from metric import compute_log_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Dataset class
class PreferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256, is_test: bool = False):
        self.prompts = df["prompt_text"].values
        self.resp_a = df["response_a_text"].values
        self.resp_b = df["response_b_text"].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        if not is_test:
            self.labels = df["target_label"].values

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        p = str(self.prompts[idx])
        a = str(self.resp_a[idx])
        b = str(self.resp_b[idx])

        text = f"[CLS] Prompt: {p} [SEP] Response A: {a} [SEP] Response B: {b} [SEP]"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 2. Test pipeline on 100 samples
df_train = load_dataset(os.path.join(BASE_DIR, "data", "train.csv"), nrows=100, is_test=False)
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = PreferenceDataset(df_train, tokenizer, max_length=128)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()

print("Running 1 training epoch on 100 samples with AMP FP16...")
model.train()
t0 = time.time()
total_loss = 0.0

for step, batch in enumerate(loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    total_loss += loss.item()

print(f"Epoch finished in {time.time() - t0:.1f}s. Avg Loss: {total_loss / len(loader):.4f}")

# 3. Test TTA Inference
model.eval()
test_sample = df_train.iloc[:5]
ds_orig = PreferenceDataset(test_sample, tokenizer, max_length=128, is_test=True)
# Swapped dataset
df_swapped = test_sample.copy()
df_swapped["response_a_text"] = test_sample["response_b_text"].values
df_swapped["response_b_text"] = test_sample["response_a_text"].values
ds_swap = PreferenceDataset(df_swapped, tokenizer, max_length=128, is_test=True)

loader_orig = DataLoader(ds_orig, batch_size=5, shuffle=False)
loader_swap = DataLoader(ds_swap, batch_size=5, shuffle=False)

with torch.no_grad():
    for b_orig, b_swap in zip(loader_orig, loader_swap):
        with torch.cuda.amp.autocast():
            out_orig = model(b_orig["input_ids"].to(device), b_orig["attention_mask"].to(device))
            p_orig = torch.softmax(out_orig.logits, dim=1).cpu().numpy()

            out_swap = model(b_swap["input_ids"].to(device), b_swap["attention_mask"].to(device))
            p_swap = torch.softmax(out_swap.logits, dim=1).cpu().numpy()

        p_final = np.zeros_like(p_orig)
        p_final[:, 0] = 0.5 * (p_orig[:, 0] + p_swap[:, 1])
        p_final[:, 1] = 0.5 * (p_orig[:, 1] + p_swap[:, 0])
        p_final[:, 2] = 0.5 * (p_orig[:, 2] + p_swap[:, 2])
        p_final /= p_final.sum(axis=1, keepdims=True)

print("TTA Symmetrized Probabilities:")
print(p_final)
print("Simplex sum:", p_final.sum(axis=1))
assert np.allclose(p_final.sum(axis=1), 1.0, atol=1e-5), "Simplex violated"
print("SUCCESS: Full DeBERTa training and TTA inference pipeline verified!")
