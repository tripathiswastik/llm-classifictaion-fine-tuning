import os
import sys
import time
import numpy as np
import pandas as pd
import torch

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)
from data_loader import load_dataset

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

def tokenize_balanced(prompt: str, resp_a: str, resp_b: str, max_p: int = 128, max_a: int = 190, max_b: int = 190, max_len: int = 512):
    # Encode segments individually
    p_ids = tokenizer.encode(f"Prompt: {prompt}", truncation=True, max_length=max_p, add_special_tokens=False)
    a_ids = tokenizer.encode(f"Response A: {resp_a}", truncation=True, max_length=max_a, add_special_tokens=False)
    b_ids = tokenizer.encode(f"Response B: {resp_b}", truncation=True, max_length=max_b, add_special_tokens=False)

    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 1
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 2

    input_ids = [cls_id] + p_ids + [sep_id] + a_ids + [sep_id] + b_ids + [sep_id]
    
    # Pad or truncate to exact max_len
    if len(input_ids) < max_len:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad_len = max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids = input_ids + [pad_id] * pad_len
    else:
        input_ids = input_ids[:max_len]
        attention_mask = [1] * max_len

    return np.array(input_ids, dtype=np.int64), np.array(attention_mask, dtype=np.int64)

# Test with long text where naive concatenation would completely truncate Response B
long_p = "What is relativity? " * 100  # 400 words
long_a = "Relativity was discovered by Einstein. " * 80  # 400 words
long_b = "General relativity describes gravity as curvature. " * 80  # 400 words

ids, mask = tokenize_balanced(long_p, long_a, long_b)
print(f"Total tokens: {len(ids)}, Attention 1s: {mask.sum()}")
decoded = tokenizer.decode(ids)
print("Contains Prompt:", "Prompt:" in decoded)
print("Contains Response A:", "Response A:" in decoded)
print("Contains Response B:", "Response B:" in decoded)
assert "Response B:" in decoded, "Response B was completely truncated!"
assert "Response A:" in decoded, "Response A was completely truncated!"
assert "Prompt:" in decoded, "Prompt was completely truncated!"
print("SUCCESS: Balanced segment tokenization verified! Both candidates symmetrically preserved.")
