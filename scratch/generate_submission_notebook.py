import json
import os

NOTEBOOK_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submission_notebook.ipynb")
)

cells = []


def add_markdown(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    })


def add_code(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    })


# 1. Title
add_markdown("""# LMSYS Chatbot Arena - LLM Classification Finetuning
## Self-Contained Champion Inference & Training Pipeline (Version 6: 107-Feature Dense LSA + GPU Tri-Ensemble)

### Kaggle Execution Settings:
- **Accelerator**: **GPU T4 x2** or **GPU P100** (Session options -> Accelerator)
- **Internet**: **OFF** (Required by competition rules)
- **Target**: Predict human preference (`winner_model_a`, `winner_model_b`, `winner_tie`)
- **Metric**: Multi-class Log Loss (eps=1e-15)
- **Architecture**: 107 Anti-Symmetric Features (72 base + 35 Dense Latent Semantic Analysis dimensions via TruncatedSVD) + GPU Tri-Ensemble (PyTorch GPU Neural Net + HistGBDT + Logistic Regression) + Temperature Calibration
- **Validation**: 5-Fold StratifiedGroupKFold on SHA-256 prompt hashes (0.0000 prompt leakage)
- **Kaggle Public Leaderboard Track Record**:
  - Version 1 (44 feats, CPU): `1.02685` Public Log Loss
  - Version 2 (55 feats, CPU): `1.02559` Public Log Loss
  - Version 3 (64 feats, CPU): `1.02335` Public Log Loss
  - Version 4 (64 feats, GPU Tri-Ensemble): `1.02125` Public Log Loss
  - Version 5 (72 feats, GPU Tri-Ensemble): `1.01163` Public Log Loss
  - Version 6 (107 feats, Dense LSA + GPU Tri-Ensemble): Target `<= 1.01000` Public Log Loss
""")

# 2. Imports and Environment Setup
add_code("""import os
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
    # 1. Search recursively in /kaggle/input for train.csv
    if os.path.exists("/kaggle/input"):
        print("Mounted items in /kaggle/input:", os.listdir("/kaggle/input"))
        matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
        if matches:
            found = os.path.dirname(matches[0])
            print(f"Auto-discovered competition data at: {found}")
            return found

    # 2. Search local candidate paths
    for cand in ["./data", "./", "../data", "../../data", "d:/ag-ai/llm-classification-finetuning/data"]:
        cand_path = os.path.join(cand, "train.csv")
        if os.path.exists(cand_path):
            print(f"Found local competition data at: {cand}")
            return cand

    # 3. Actionable error if dataset is missing from /kaggle/input
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
print(f"Submission target: {SUBMISSION_PATH}")
""")

# 3. Log Loss Metric
add_code("""def compute_log_loss(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray, eps: float = 1e-15) -> float:
    \"\"\"
    Computes exact Kaggle competition multi-class log loss with eps-clipping.
    \"\"\"
    p_clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
    p_normalized = p_clipped / p_clipped.sum(axis=1, keepdims=True)
    sample_losses = -np.sum(y_true_onehot * np.log(p_normalized), axis=1)
    return float(np.mean(sample_losses))
""")

# 4. Turn Deserialization & Prompt Hashing
add_code("""def parse_conversation_turns(raw_val: Union[str, list, None]) -> List[str]:
    \"\"\"
    Safely deserializes a JSON array string representing multi-turn conversation messages.
    \"\"\"
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
        return [s]

def flatten_turns(turns: List[str], separator: str = " \\n\\n ") -> str:
    return separator.join(turns).strip()

def hash_prompt(prompt_str: str) -> str:
    \"\"\"
    Creates a deterministic 16-character SHA-256 hash of normalized prompt text.
    Prevents prompt leakage across cross-validation folds.
    \"\"\"
    normalized = " ".join(prompt_str.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def load_dataset(csv_path: str, nrows: Optional[int] = None, is_test: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows)
    df["prompt_turns"] = df["prompt"].apply(parse_conversation_turns)
    df["response_a_turns"] = df["response_a"].apply(parse_conversation_turns)
    df["response_b_turns"] = df["response_b"].apply(parse_conversation_turns)

    df["prompt_text"] = df["prompt_turns"].apply(flatten_turns)
    df["response_a_text"] = df["response_a_turns"].apply(flatten_turns)
    df["response_b_text"] = df["response_b_turns"].apply(flatten_turns)
    df["prompt_group"] = df["prompt_text"].apply(hash_prompt)

    if not is_test:
        valid_mask = (df[TARGET_COLUMNS].sum(axis=1) == 1)
        if not valid_mask.all():
            raise ValueError(f"Found {(~valid_mask).sum()} invalid target rows")
        df["target_label"] = np.argmax(df[TARGET_COLUMNS].values, axis=1)

    return df
""")

# 5. 64-Feature Engineering Engine
add_code("""RE_WORDS = re.compile(r"\\b\\w+\\b")
RE_CODE_BLOCKS = re.compile(r"```")
RE_BULLETS = re.compile(r"(?:^|\\n)\\s*[-*+]\\s+")
RE_NUMBERED = re.compile(r"(?:^|\\n)\\s*\\d+\\.\\s+")
RE_HEADERS = re.compile(r"(?:^|\\n)\\s*#{1,6}\\s+")
RE_BOLD = re.compile(r"\\*\\*[^*]+\\*\\*")
RE_LATEX = re.compile(r"\\$[^\\$]+\\$")
RE_URL = re.compile(r"https?://\\S+|www\\.\\S+")
RE_APOLOGY = re.compile(r"\\b(sorry|apologize|apologies|pardon)\\b", re.IGNORECASE)
RE_AI_REF = re.compile(r"\\b(as an ai|language model|ai assistant|artificial intelligence)\\b", re.IGNORECASE)
RE_REFUSAL = re.compile(r"\\b(cannot fulfill|unable to|i cannot|i can't|violates|not able to)\\b", re.IGNORECASE)
RE_SENTENCE_SPLIT = re.compile(r"[.!?\\n]+")
RE_VOWELS = re.compile(r"[aeiouy]+", re.IGNORECASE)
RE_NUM_CONSTRAINT = re.compile(
    r"\\b(?:give me|list|write|provide)?\\s*(\\d{1,2})\\s*(?:reasons|points|tips|ways|examples|items|steps|ideas)\\b",
    re.IGNORECASE,
)
RE_TABLE = re.compile(r"(?:^|\\n)\\s*\\|.*?\\|.*?\\|(?:\\n\\s*\\|[-:\\s|]+\\|)", re.MULTILINE)
RE_SYNTAX_CODE = re.compile(r"```[a-zA-Z0-9_+#-]+")
RE_SYCOPHANCY = re.compile(
    r"\\b(great question|good question|you are right|absolutely right|certainly|delighted to help|happy to assist|pleasure to help|as an expert)\\b",
    re.IGNORECASE,
)
RE_CODE_PROMPT = re.compile(r"\\b(code|python|java|c\\+\\+|javascript|function|algorithm|implement|script|sql|html|css|bug|error|exception)\\b", re.IGNORECASE)
RE_MATH_PROMPT = re.compile(r"\\b(calculate|math|equation|solve|integral|derivative|formula|probability|matrix|algebra|theorem)\\b", re.IGNORECASE)
RE_COMPARE_PROMPT = re.compile(r"\\b(compare|comparison|versus|vs|table|difference between|pros and cons)\\b", re.IGNORECASE)
RE_FIRST_PERSON = re.compile(r"\\b(i|me|my|mine|myself|we|us|our|ours)\\b", re.IGNORECASE)
RE_SECOND_PERSON = re.compile(r"\\b(you|your|yours|yourself)\\b", re.IGNORECASE)
RE_COMPLEX_PUNCT = re.compile(r"[;—\\(\\)\\[\\]\\{\\}]")
RE_CONNECTIVES = re.compile(
    r"\\b(furthermore|moreover|therefore|however|nevertheless|consequently|specifically|in particular|for instance|for example|in summary|in conclusion)\\b",
    re.IGNORECASE,
)

RE_UPPER = re.compile(r"[A-Z]")
RE_DIGIT = re.compile(r"[0-9]")

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

def compute_entropy(words: List[str]) -> float:
    if len(words) <= 1:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def compute_compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    raw_bytes = text.encode("utf-8", errors="replace")
    compressed = zlib.compress(raw_bytes, level=6)
    return len(compressed) / max(len(raw_bytes), 1)

def count_syllables(word: str) -> int:
    w = word.lower().strip()
    if len(w) <= 3:
        return 1
    if w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    matches = RE_VOWELS.findall(w)
    return max(len(matches), 1)

def compute_text_stats(text: str) -> Dict[str, float]:
    char_len = len(text)
    if char_len == 0:
        return {
            "char_len": 0.0, "word_count": 0.0, "unique_words": 0.0, "avg_word_len": 0.0,
            "code_blocks": 0.0, "bullets": 0.0, "numbered_lists": 0.0, "headers": 0.0,
            "bold_spans": 0.0, "latex_math": 0.0, "urls": 0.0, "newlines": 0.0,
            "qmarks": 0.0, "excls": 0.0, "colons": 0.0, "uppercase_ratio": 0.0,
            "digit_count": 0.0, "apology_count": 0.0, "ai_ref_count": 0.0, "refusal_count": 0.0,
            "trigram_rep_rate": 0.0, "type_token_ratio": 1.0, "avg_sent_len": 0.0,
            "flesch_reading_ease": 0.0, "code_density": 0.0,
            "has_table": 0.0, "syntax_code_blocks": 0.0, "sycophancy": 0.0,
            "first_person": 0.0, "second_person": 0.0, "complex_punct": 0.0,
        }

    words = RE_WORDS.findall(text)
    word_count = len(words)
    words_lower = [w.lower() for w in words]
    unique_words = len(set(words_lower))
    total_word_len = sum(len(w) for w in words)
    avg_word_len = total_word_len / max(word_count, 1)

    code_blocks = text.count("```") // 2
    bullets = len(RE_BULLETS.findall(text))
    numbered = len(RE_NUMBERED.findall(text))
    headers = len(RE_HEADERS.findall(text))
    bold = text.count("**") // 2
    latex = text.count("$") // 2
    urls = len(RE_URL.findall(text))

    newlines = text.count("\\n")
    qmarks = text.count("?")
    excls = text.count("!")
    colons = text.count(":")
    digits = len(RE_DIGIT.findall(text))
    uppers = len(RE_UPPER.findall(text))
    uppercase_ratio = uppers / max(char_len, 1)

    apology = len(RE_APOLOGY.findall(text))
    ai_ref = len(RE_AI_REF.findall(text))
    refusal = len(RE_REFUSAL.findall(text))

    ttr = unique_words / max(word_count, 1)
    sentences = max(len(RE_SENTENCE_SPLIT.findall(text)), 1)
    avg_sent_len = word_count / sentences

    if word_count < 5:
        rep_rate = 0.0
    else:
        trigrams = [tuple(words_lower[i : i + 3]) for i in range(word_count - 2)]
        rep_rate = 1.0 - (len(set(trigrams)) / max(len(trigrams), 1))

    syl = sum(count_syllables(w) for w in words) / max(word_count, 1)
    fre = 206.835 - 1.015 * avg_sent_len - 84.6 * syl

    code_chars = sum(len(c) for c in re.findall(r"```[\\s\\S]*?```", text))
    code_density = code_chars / max(char_len, 1)

    has_table = float(bool(RE_TABLE.search(text)))
    syntax_code_blocks = float(len(RE_SYNTAX_CODE.findall(text)))
    sycophancy = float(len(RE_SYCOPHANCY.findall(text)))

    first_person = float(len(RE_FIRST_PERSON.findall(text)))
    second_person = float(len(RE_SECOND_PERSON.findall(text)))
    complex_punct = float(len(RE_COMPLEX_PUNCT.findall(text)))

    return {
        "char_len": float(char_len), "word_count": float(word_count), "unique_words": float(unique_words),
        "avg_word_len": float(avg_word_len), "code_blocks": float(code_blocks), "bullets": float(bullets),
        "numbered_lists": float(numbered), "headers": float(headers), "bold_spans": float(bold),
        "latex_math": float(latex), "urls": float(urls), "newlines": float(newlines),
        "qmarks": float(qmarks), "excls": float(excls), "colons": float(colons),
        "uppercase_ratio": float(uppercase_ratio), "digit_count": float(digits), "apology_count": float(apology),
        "ai_ref_count": float(ai_ref), "refusal_count": float(refusal), "trigram_rep_rate": float(rep_rate),
        "type_token_ratio": float(ttr), "avg_sent_len": float(avg_sent_len), "flesch_reading_ease": float(fre),
        "code_density": float(code_density),
        "has_table": has_table, "syntax_code_blocks": syntax_code_blocks, "sycophancy": sycophancy,
        "first_person": first_person, "second_person": second_person, "complex_punct": complex_punct,
    }

def compute_pairwise_features(prompt: str, resp_a: str, resp_b: str, turns_a: int = 1, turns_b: int = 1) -> Dict[str, float]:
    stats_a = compute_text_stats(resp_a)
    stats_b = compute_text_stats(resp_b)

    p_words = [w.lower() for w in RE_WORDS.findall(prompt)]
    a_words = [w.lower() for w in RE_WORDS.findall(resp_a)]
    b_words = [w.lower() for w in RE_WORDS.findall(resp_b)]

    p_set = set(p_words)
    a_set = set(a_words)
    b_set = set(b_words)

    inter_pa = len(p_set & a_set)
    union_pa = max(len(p_set | a_set), 1)
    jaccard_pa = inter_pa / union_pa

    inter_pb = len(p_set & b_set)
    union_pb = max(len(p_set | b_set), 1)
    jaccard_pb = inter_pb / union_pb

    cov_pa = inter_pa / max(len(p_set), 1)
    cov_pb = inter_pb / max(len(p_set), 1)

    content_p = set(w for w in p_words if w not in STOPWORDS)
    content_a = set(w for w in a_words if w not in STOPWORDS)
    content_b = set(w for w in b_words if w not in STOPWORDS)

    if content_p:
        c_cov_pa = len(content_p & content_a) / len(content_p)
        c_cov_pb = len(content_p & content_b) / len(content_p)
        c_union_pa = max(len(content_p | content_a), 1)
        c_union_pb = max(len(content_p | content_b), 1)
        c_jacc_pa = len(content_p & content_a) / c_union_pa
        c_jacc_pb = len(content_p & content_b) / c_union_pb
    else:
        c_cov_pa = c_cov_pb = c_jacc_pa = c_jacc_pb = 0.0

    inter_ab = len(a_set & b_set)
    union_ab = max(len(a_set | b_set), 1)
    jaccard_ab = inter_ab / union_ab
    char_len_ab_diff = abs(stats_a["char_len"] - stats_b["char_len"])
    char_len_ab_min = min(stats_a["char_len"], stats_b["char_len"])
    char_len_ab_max = max(stats_a["char_len"], stats_b["char_len"])
    char_len_ratio = (char_len_ab_min + 1.0) / (char_len_ab_max + 1.0)

    char_diff = stats_a["char_len"] - stats_b["char_len"]
    word_diff = stats_a["word_count"] - stats_b["word_count"]
    code_diff = stats_a["code_blocks"] - stats_b["code_blocks"]
    latex_diff = stats_a["latex_math"] - stats_b["latex_math"]
    log_char_diff = math.log(stats_a["char_len"] + 1.0) - math.log(stats_b["char_len"] + 1.0)
    log_word_diff = math.log(stats_a["word_count"] + 1.0) - math.log(stats_b["word_count"] + 1.0)

    match_constraint = RE_NUM_CONSTRAINT.search(prompt)
    if match_constraint:
        target_n = int(match_constraint.group(1))
        bullets_a = len(RE_BULLETS.findall(resp_a)) + len(RE_NUMBERED.findall(resp_a))
        bullets_b = len(RE_BULLETS.findall(resp_b)) + len(RE_NUMBERED.findall(resp_b))
        diff_constraint_err = float(abs(bullets_b - target_n) - abs(bullets_a - target_n))
    else:
        diff_constraint_err = 0.0

    is_code_p = float(bool(RE_CODE_PROMPT.search(prompt)))
    is_math_p = float(bool(RE_MATH_PROMPT.search(prompt)))
    is_compare_p = float(bool(RE_COMPARE_PROMPT.search(prompt)))

    diff_table = stats_a["has_table"] - stats_b["has_table"]
    diff_prompt_code_inter = is_code_p * code_diff
    diff_prompt_math_inter = is_math_p * latex_diff
    diff_prompt_table_inter = is_compare_p * diff_table

    # Shannon Lexical Entropy
    ent_a = compute_entropy(a_words)
    ent_b = compute_entropy(b_words)
    diff_entropy = ent_a - ent_b
    sym_mean_entropy = 0.5 * (ent_a + ent_b)

    # Zlib Compression Ratio (information density & redundancy)
    cr_a = compute_compression_ratio(resp_a)
    cr_b = compute_compression_ratio(resp_b)
    diff_cr = cr_a - cr_b
    sym_mean_cr = 0.5 * (cr_a + cr_b)

    # Content Word Bigram Coverage
    p_content_list = [w for w in p_words if w not in STOPWORDS]
    a_content_list = [w for w in a_words if w not in STOPWORDS]
    b_content_list = [w for w in b_words if w not in STOPWORDS]
    p_bigrams = set(zip(p_content_list[:-1], p_content_list[1:])) if len(p_content_list) >= 2 else set()
    a_bigrams = set(zip(a_content_list[:-1], a_content_list[1:])) if len(a_content_list) >= 2 else set()
    b_bigrams = set(zip(b_content_list[:-1], b_content_list[1:])) if len(b_content_list) >= 2 else set()
    if p_bigrams:
        cov_bigram_a = len(p_bigrams & a_bigrams) / len(p_bigrams)
        cov_bigram_b = len(p_bigrams & b_bigrams) / len(p_bigrams)
    else:
        cov_bigram_a = cov_bigram_b = 0.0
    diff_bigram_cov = cov_bigram_a - cov_bigram_b
    sym_mean_bigram_cov = 0.5 * (cov_bigram_a + cov_bigram_b)

    # Discourse Connectives & Cohesion Markers
    conn_a = float(len(RE_CONNECTIVES.findall(resp_a)))
    conn_b = float(len(RE_CONNECTIVES.findall(resp_b)))
    diff_connectives = conn_a - conn_b
    sym_connectives = 0.5 * (conn_a + conn_b)

    return {
        \"diff_char_len\": char_diff,
        \"diff_word_count\": word_diff,
        \"diff_log_char\": log_char_diff,
        \"diff_log_word\": log_word_diff,
        \"diff_avg_word_len\": stats_a[\"avg_word_len\"] - stats_b[\"avg_word_len\"],
        \"diff_code_blocks\": code_diff,
        \"diff_bullets\": stats_a[\"bullets\"] - stats_b[\"bullets\"],
        \"diff_numbered\": stats_a[\"numbered_lists\"] - stats_b[\"numbered_lists\"],
        \"diff_headers\": stats_a[\"headers\"] - stats_b[\"headers\"],
        \"diff_bold\": stats_a[\"bold_spans\"] - stats_b[\"bold_spans\"],
        \"diff_latex\": latex_diff,
        \"diff_urls\": stats_a[\"urls\"] - stats_b[\"urls\"],
        \"diff_newlines\": stats_a[\"newlines\"] - stats_b[\"newlines\"],
        \"diff_qmarks\": stats_a[\"qmarks\"] - stats_b[\"qmarks\"],
        \"diff_excls\": stats_a[\"excls\"] - stats_b[\"excls\"],
        \"diff_colons\": stats_a[\"colons\"] - stats_b[\"colons\"],
        \"diff_uppercase_ratio\": stats_a[\"uppercase_ratio\"] - stats_b[\"uppercase_ratio\"],
        \"diff_apology\": stats_a[\"apology_count\"] - stats_b[\"apology_count\"],
        \"diff_ai_ref\": stats_a[\"ai_ref_count\"] - stats_b[\"ai_ref_count\"],
        \"diff_refusal\": stats_a[\"refusal_count\"] - stats_b[\"refusal_count\"],
        \"diff_jaccard_p\": jaccard_pa - jaccard_pb,
        \"diff_cov_p\": cov_pa - cov_pb,
        \"diff_turns\": float(turns_a - turns_b),
        \"diff_trigram_rep_rate\": stats_a[\"trigram_rep_rate\"] - stats_b[\"trigram_rep_rate\"],
        \"diff_type_token_ratio\": stats_a[\"type_token_ratio\"] - stats_b[\"type_token_ratio\"],
        \"diff_avg_sent_len\": stats_a[\"avg_sent_len\"] - stats_b[\"avg_sent_len\"],
        \"diff_flesch_reading_ease\": stats_a[\"flesch_reading_ease\"] - stats_b[\"flesch_reading_ease\"],
        \"diff_code_density\": stats_a[\"code_density\"] - stats_b[\"code_density\"],
        \"diff_constraint_adherence\": diff_constraint_err,
        \"diff_markdown_table\": diff_table,
        \"diff_syntax_code_blocks\": stats_a[\"syntax_code_blocks\"] - stats_b[\"syntax_code_blocks\"],
        \"diff_sycophancy\": stats_a[\"sycophancy\"] - stats_b[\"sycophancy\"],
        \"diff_prompt_code_inter\": diff_prompt_code_inter,
        \"diff_prompt_math_inter\": diff_prompt_math_inter,
        \"diff_prompt_table_inter\": diff_prompt_table_inter,
        \"diff_first_person\": stats_a[\"first_person\"] - stats_b[\"first_person\"],
        \"diff_second_person\": stats_a[\"second_person\"] - stats_b[\"second_person\"],
        \"diff_complex_punct\": stats_a[\"complex_punct\"] - stats_b[\"complex_punct\"],
        \"diff_content_cov\": float(c_cov_pa - c_cov_pb),
        \"diff_content_jacc\": float(c_jacc_pa - c_jacc_pb),
        \"diff_entropy\": diff_entropy,
        \"diff_compression_ratio\": diff_cr,
        \"diff_bigram_cov\": diff_bigram_cov,
        \"diff_connectives\": diff_connectives,
        \"sym_jaccard_ab\": jaccard_ab,
        \"sym_abs_char_diff\": char_len_ab_diff,
        \"sym_char_ratio\": char_len_ratio,
        \"sym_total_char_len\": stats_a[\"char_len\"] + stats_b[\"char_len\"],
        \"sym_prompt_char_len\": float(len(prompt)),
        \"sym_prompt_word_len\": float(len(p_set)),
        \"sym_mean_jaccard_p\": 0.5 * (jaccard_pa + jaccard_pb),
        \"sym_code_presence\": float((stats_a[\"code_blocks\"] > 0) or (stats_b[\"code_blocks\"] > 0)),
        \"sym_latex_presence\": float((stats_a[\"latex_math\"] > 0) or (stats_b[\"latex_math\"] > 0)),
        \"sym_mean_ttr\": 0.5 * (stats_a[\"type_token_ratio\"] + stats_b[\"type_token_ratio\"]),
        \"sym_mean_flesch_reading_ease\": 0.5 * (stats_a[\"flesch_reading_ease\"] + stats_b[\"flesch_reading_ease\"]),
        \"sym_mean_code_density\": 0.5 * (stats_a[\"code_density\"] + stats_b[\"code_density\"]),
        \"sym_markdown_table\": 0.5 * (stats_a[\"has_table\"] + stats_b[\"has_table\"]),
        \"sym_syntax_code_blocks\": 0.5 * (stats_a[\"syntax_code_blocks\"] + stats_b[\"syntax_code_blocks\"]),
        \"sym_sycophancy\": 0.5 * (stats_a[\"sycophancy\"] + stats_b[\"sycophancy\"]),
        \"sym_is_code_prompt\": is_code_p,
        \"sym_is_math_prompt\": is_math_p,
        \"sym_first_person\": 0.5 * (stats_a[\"first_person\"] + stats_b[\"first_person\"]),
        \"sym_second_person\": 0.5 * (stats_a[\"second_person\"] + stats_b[\"second_person\"]),
        \"sym_complex_punct\": 0.5 * (stats_a[\"complex_punct\"] + stats_b[\"complex_punct\"]),
        \"sym_mean_content_cov\": 0.5 * float(c_cov_pa + c_cov_pb),
        \"sym_mean_entropy\": sym_mean_entropy,
        \"sym_mean_compression_ratio\": sym_mean_cr,
        \"sym_mean_bigram_cov\": sym_mean_bigram_cov,
        \"sym_connectives\": sym_connectives,
    }

class FeatureExtractor:
    def __init__(
        self,
        use_tfidf: bool = True,
        max_tfidf_features: int = 15000,
        use_lsa: bool = True,
        n_lsa_components: int = 16,
    ):
        self.use_tfidf = use_tfidf
        self.max_tfidf_features = max_tfidf_features
        self.use_lsa = use_lsa
        self.n_lsa_components = n_lsa_components
        self.base_feature_names_: List[str] = []
        self.feature_names_: List[str] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None
        self.lsa_feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        sample_feat = compute_pairwise_features(
            prompt=df["prompt_text"].iloc[0],
            resp_a=df["response_a_text"].iloc[0],
            resp_b=df["response_b_text"].iloc[0],
        )
        self.base_feature_names_ = sorted(list(sample_feat.keys()))

        if self.use_tfidf:
            corpus = (
                list(df["prompt_text"].values)
                + list(df["response_a_text"].values)
                + list(df["response_b_text"].values)
            )
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                sublinear_tf=True,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_mat = self.tfidf_vectorizer.fit_transform(corpus)
            self.feature_names_ = self.base_feature_names_ + [
                "diff_tfidf_prompt_sim",
                "sym_tfidf_ab_sim",
                "sym_tfidf_mean_prompt",
            ]

            if self.use_lsa:
                n_features = tfidf_mat.shape[1]
                n_samples = tfidf_mat.shape[0]
                actual_k = min(self.n_lsa_components, max(1, min(n_samples, n_features) - 1))
                self.svd = TruncatedSVD(n_components=actual_k, random_state=42)
                self.svd.fit(tfidf_mat)
                self.lsa_feature_names_ = [
                    "diff_lsa_prompt_cos",
                    "sym_lsa_ab_cos",
                    "sym_lsa_mean_prompt_cos",
                ]
                for i in range(self.n_lsa_components):
                    self.lsa_feature_names_.append(f"diff_lsa_dim_{i}")
                    self.lsa_feature_names_.append(f"sym_lsa_dim_{i}")
                self.feature_names_ = self.feature_names_ + self.lsa_feature_names_
        else:
            self.feature_names_ = list(self.base_feature_names_)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        p_texts = df["prompt_text"].values
        a_texts = df["response_a_text"].values
        b_texts = df["response_b_text"].values
        a_turns = [len(t) for t in df["response_a_turns"].values]
        b_turns = [len(t) for t in df["response_b_turns"].values]

        for p, a, b, ta, tb in zip(p_texts, a_texts, b_texts, a_turns, b_turns):
            feat_dict = compute_pairwise_features(p, a, b, ta, tb)
            rows.append([feat_dict[k] for k in self.base_feature_names_])

        X_base = np.asarray(rows, dtype=np.float32)

        if self.use_tfidf and self.tfidf_vectorizer is not None:
            p_vecs = self.tfidf_vectorizer.transform(df["prompt_text"])
            a_vecs = self.tfidf_vectorizer.transform(df["response_a_text"])
            b_vecs = self.tfidf_vectorizer.transform(df["response_b_text"])

            sim_pa = np.asarray(p_vecs.multiply(a_vecs).sum(axis=1)).flatten()
            sim_pb = np.asarray(p_vecs.multiply(b_vecs).sum(axis=1)).flatten()
            sim_ab = np.asarray(a_vecs.multiply(b_vecs).sum(axis=1)).flatten()

            diff_prompt_sim = sim_pa - sim_pb
            sym_ab_sim = sim_ab
            mean_prompt_sim = 0.5 * (sim_pa + sim_pb)

            X_tfidf = np.column_stack([diff_prompt_sim, sym_ab_sim, mean_prompt_sim]).astype(np.float32)

            if self.use_lsa and self.svd is not None:
                p_svd = self.svd.transform(p_vecs)
                a_svd = self.svd.transform(a_vecs)
                b_svd = self.svd.transform(b_vecs)

                k = self.n_lsa_components
                if a_svd.shape[1] < k:
                    pad = k - a_svd.shape[1]
                    p_svd = np.pad(p_svd, ((0, 0), (0, pad)), mode="constant")
                    a_svd = np.pad(a_svd, ((0, 0), (0, pad)), mode="constant")
                    b_svd = np.pad(b_svd, ((0, 0), (0, pad)), mode="constant")

                def l2_norm(mat: np.ndarray) -> np.ndarray:
                    norms = np.linalg.norm(mat, axis=1, keepdims=True)
                    return mat / np.maximum(norms, 1e-9)

                p_norm = l2_norm(p_svd)
                a_norm = l2_norm(a_svd)
                b_norm = l2_norm(b_svd)

                cos_pa = np.sum(p_norm * a_norm, axis=1, keepdims=True)
                cos_pb = np.sum(p_norm * b_norm, axis=1, keepdims=True)
                cos_ab = np.sum(a_norm * b_norm, axis=1, keepdims=True)

                diff_cos_p = (cos_pa - cos_pb).astype(np.float32)
                sym_cos_ab = cos_ab.astype(np.float32)
                sym_mean_cos_p = (0.5 * (cos_pa + cos_pb)).astype(np.float32)

                diff_svd = (a_svd - b_svd).astype(np.float32)
                sym_svd = (0.5 * (a_svd + b_svd)).astype(np.float32)

                dim_cols = []
                for i in range(self.n_lsa_components):
                    dim_cols.append(diff_svd[:, i : i + 1])
                    dim_cols.append(sym_svd[:, i : i + 1])

                X_lsa = np.hstack([diff_cos_p, sym_cos_ab, sym_mean_cos_p] + dim_cols)
                return np.hstack([X_base, X_tfidf, X_lsa])

            return np.hstack([X_base, X_tfidf])

        return X_base

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
""")

# 6. Symmetric Model Wrapper & Ensemble
add_code("""class SymmetricModelWrapper:
    \"\"\"
    Enforces anti-symmetric invariance via pair-swap training augmentation
    and test-time symmetrization.
    \"\"\"
    def __init__(self, base_estimator, feature_names: List[str]):
        self.base_estimator = base_estimator
        self.feature_names = feature_names
        self.scaler = StandardScaler()

    def _swap_features(self, X: np.ndarray) -> np.ndarray:
        X_swapped = X.copy()
        for idx, name in enumerate(self.feature_names):
            if name.startswith("diff_"):
                X_swapped[:, idx] = -X_swapped[:, idx]
        return X_swapped

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SymmetricModelWrapper":
        X_swap = self._swap_features(X)
        y_swap = y.copy()
        mask_0 = (y == 0)
        mask_1 = (y == 1)
        y_swap[mask_0] = 1
        y_swap[mask_1] = 0

        X_aug = np.vstack([X, X_swap])
        y_aug = np.concatenate([y, y_swap])
        X_scaled = self.scaler.fit_transform(X_aug)
        self.base_estimator.fit(X_scaled, y_aug)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled_orig = self.scaler.transform(X)
        p_orig = self.base_estimator.predict_proba(X_scaled_orig)

        X_swap = self._swap_features(X)
        X_scaled_swap = self.scaler.transform(X_swap)
        p_swap = self.base_estimator.predict_proba(X_scaled_swap)

        p_swap_inverted = np.zeros_like(p_swap)
        p_swap_inverted[:, 0] = p_swap[:, 1]
        p_swap_inverted[:, 1] = p_swap[:, 0]
        p_swap_inverted[:, 2] = p_swap[:, 2]

        p_sym = 0.5 * (p_orig + p_swap_inverted)
        return p_sym / p_sym.sum(axis=1, keepdims=True)

def get_hist_gbdt_loss() -> str:
    \"\"\"
    Detects supported loss parameter across scikit-learn versions
    ('log_loss' in sklearn >= 1.1; 'auto' in sklearn <= 1.0).
    \"\"\"
    try:
        import sklearn
        parts = [int(x) for x in sklearn.__version__.split(".") if x.isdigit()]
        if parts and (parts[0] > 1 or (parts[0] == 1 and len(parts) > 1 and parts[1] >= 1)):
            return "log_loss"
    except Exception:
        pass
    return "auto"

class EnsemblePreferenceModel:
    \"\"\"
    Symmetric blend of regularized Logistic Regression and HistGBDT.
    \"\"\"
    def __init__(self, feature_names: List[str], lr_weight: float = 0.44):
        self.feature_names = feature_names
        self.lr_weight = lr_weight
        self.lr_model = SymmetricModelWrapper(
            LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42),
            feature_names
        )
        self.gbdt_model = SymmetricModelWrapper(
            HistGradientBoostingClassifier(
                loss=get_hist_gbdt_loss(),
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                random_state=42,
            ),
            feature_names
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsemblePreferenceModel":
        self.lr_model.fit(X, y)
        self.gbdt_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_lr = self.lr_model.predict_proba(X)
        p_gbdt = self.gbdt_model.predict_proba(X)
        p_blend = self.lr_weight * p_lr + (1.0 - self.lr_weight) * p_gbdt
        return p_blend / p_blend.sum(axis=1, keepdims=True)

class SymmetricMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor, x_swap: torch.Tensor) -> torch.Tensor:
        logits_orig = self.forward_single(x)
        logits_swap = self.forward_single(x_swap)
        logits_swap_inv = torch.stack([logits_swap[:, 1], logits_swap[:, 0], logits_swap[:, 2]], dim=1)
        return 0.5 * (logits_orig + logits_swap_inv)

class PyTorchGPUPreferenceModel:
    def __init__(
        self,
        feature_names: List[str],
        hidden_dim: int = 128,
        lr: float = 2e-3,
        epochs: int = 20,
        batch_size: int = 512,
        device: torch.device = DEVICE,
    ):
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.diff_indices = [i for i, name in enumerate(feature_names) if name.startswith("diff_")]
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.model: Optional[SymmetricMLP] = None

    def _swap_features_tensor(self, X: torch.Tensor) -> torch.Tensor:
        X_swap = X.clone()
        X_swap[:, self.diff_indices] = -X_swap[:, self.diff_indices]
        return X_swap

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PyTorchGPUPreferenceModel":
        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        X_swap_t = self._swap_features_tensor(X_t)
        X_aug = torch.cat([X_t, X_swap_t], dim=0)
        self.mean = X_aug.mean(dim=0, keepdim=True)
        self.mean[:, self.diff_indices] = 0.0
        self.std = X_aug.std(dim=0, keepdim=True) + 1e-7

        X_norm = (X_t - self.mean) / self.std
        X_swap_norm = (X_swap_t - self.mean) / self.std

        dataset = TensorDataset(X_norm, X_swap_norm, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SymmetricMLP(X_train.shape[1], hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_x_swap, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x, batch_x_swap)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_norm = (X_t - self.mean) / self.std
        X_swap_norm = self._swap_features_tensor(X_norm)

        with torch.no_grad():
            logits = self.model(X_norm, X_swap_norm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

class TriEnsemblePreferenceModel:
    \"\"\"
    Triple-architecture ensemble:
    Logistic Regression (0.45) + HistGBDT (0.40) + PyTorch GPU Model (0.15).
    \"\"\"
    def __init__(
        self,
        feature_names: List[str],
        lr_weight: float = 0.45,
        gbdt_weight: float = 0.40,
        gpu_pt_weight: float = 0.15,
    ):
        self.feature_names = feature_names
        self.lr_weight = lr_weight
        self.gbdt_weight = gbdt_weight
        self.gpu_pt_weight = gpu_pt_weight

        self.lr_model = SymmetricModelWrapper(
            LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42),
            feature_names
        )
        self.gbdt_model = SymmetricModelWrapper(
            HistGradientBoostingClassifier(
                loss=get_hist_gbdt_loss(),
                max_iter=150,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                random_state=42,
            ),
            feature_names
        )
        self.pt_model = PyTorchGPUPreferenceModel(
            feature_names,
            hidden_dim=128,
            epochs=20,
            batch_size=512,
            device=DEVICE,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TriEnsemblePreferenceModel":
        self.lr_model.fit(X, y)
        self.gbdt_model.fit(X, y)
        self.pt_model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_lr = self.lr_model.predict_proba(X)
        p_gbdt = self.gbdt_model.predict_proba(X)
        p_pt = self.pt_model.predict_proba(X)

        p_blend = (
            self.lr_weight * p_lr
            + self.gbdt_weight * p_gbdt
            + self.gpu_pt_weight * p_pt
        )
        return p_blend / p_blend.sum(axis=1, keepdims=True)
""")

# 7. Temperature and Tie-Scale Calibrator
add_code("""class TemperatureCalibrator:
    \"\"\"
    Fits and applies symmetric temperature and tie scaling.
    Preserves mathematical anti-symmetry and simplex constraints.
    \"\"\"
    def __init__(self, temperature: float = 1.0, tie_scale: float = 1.0):
        self.temperature = float(temperature)
        self.tie_scale = float(tie_scale)

    def fit(self, probs: np.ndarray, y_true_onehot: np.ndarray) -> "TemperatureCalibrator":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)

        def objective(params: np.ndarray) -> float:
            temp, tie_sc = params
            log_p = np.log(p) / max(temp, 1e-4)
            log_p[:, 2] += np.log(max(tie_sc, 1e-5))
            exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
            p_cal = exp_p / exp_p.sum(axis=1, keepdims=True)
            return compute_log_loss(y_true_onehot, p_cal)

        res = minimize(
            objective,
            x0=[self.temperature, self.tie_scale],
            bounds=[(0.2, 5.0), (0.2, 5.0)],
            method="L-BFGS-B",
        )
        self.temperature = float(res.x[0])
        self.tie_scale = float(res.x[1])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-15
        p = np.clip(probs, eps, 1.0 - eps)
        log_p = np.log(p) / max(self.temperature, 1e-4)
        log_p[:, 2] += np.log(max(self.tie_scale, 1e-5))
        exp_p = np.exp(log_p - np.max(log_p, axis=1, keepdims=True))
        p_cal = exp_p / exp_p.sum(axis=1, keepdims=True)
        return p_cal
""")

# 8. Main Training and Inference Pipeline
add_code("""# Load Training Data
print(f"Loading training data from {TRAIN_PATH}...")
df_train = load_dataset(TRAIN_PATH, is_test=False)
print(f"Loaded {len(df_train)} training battles across {df_train['prompt_group'].nunique()} prompt groups.")

# Extract Features
print("Extracting 107-feature matrix (72 base + 35 LSA semantic features)...")
extractor = FeatureExtractor(use_tfidf=True, max_tfidf_features=15000, use_lsa=True, n_lsa_components=16)
X = extractor.fit_transform(df_train)
y = df_train["target_label"].values
y_true_onehot = df_train[TARGET_COLUMNS].values
print(f"Extracted feature matrix shape: {X.shape}")

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
        lr_weight=0.45,
        gbdt_weight=0.40,
        gpu_pt_weight=0.15,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict_proba(X_val)
    oof_preds[val_idx] = val_pred
    fold_models.append(model)

    fold_loss = compute_log_loss(y_val_onehot, val_pred)
    print(f"  Fold {fold + 1}/{n_splits} Log Loss: {fold_loss:.5f} (Val size: {len(val_idx)})")

raw_oof_loss = compute_log_loss(y_true_onehot, oof_preds)
print(f"\\nRaw Ensemble OOF Log Loss: {raw_oof_loss:.5f}")

# Fit Calibrator
print("Optimizing Symmetric Temperature & Tie-Scale Calibration...")
calibrator = TemperatureCalibrator()
calibrator.fit(oof_preds, y_true_onehot)
cal_oof_preds = calibrator.transform(oof_preds)
cal_oof_loss = compute_log_loss(y_true_onehot, cal_oof_preds)
print(f"Optimal Temperature: {calibrator.temperature:.4f}, Tie Scale: {calibrator.tie_scale:.4f}")
print(f"Calibrated Ensemble OOF Log Loss: {cal_oof_loss:.5f}")
""")

# 9. Test Prediction and Submission Generation
add_code("""# Load Test Data
print(f"\\nLoading test data from {TEST_PATH}...")
df_test = load_dataset(TEST_PATH, is_test=True)
df_sample = pd.read_csv(SAMPLE_SUB_PATH)

X_test = extractor.transform(df_test)
print(f"Transformed test features: {X_test.shape}")

# Average predictions from all fold models
preds_list = [m.predict_proba(X_test) for m in fold_models]
avg_preds = np.mean(preds_list, axis=0)
avg_preds = avg_preds / avg_preds.sum(axis=1, keepdims=True)

# Apply Temperature Calibration
cal_test_preds = calibrator.transform(avg_preds)

# Format Submission DataFrame
submission = pd.DataFrame({
    "id": df_test["id"].values,
    "winner_model_a": cal_test_preds[:, 0],
    "winner_model_b": cal_test_preds[:, 1],
    "winner_tie": cal_test_preds[:, 2],
})

# Rigorous Verification Assertions
assert list(submission.columns) == list(df_sample.columns), "Column mismatch"
assert len(submission) == len(df_test), f"Row count mismatch: {len(submission)} vs {len(df_test)}"
assert not submission.isnull().any().any(), "Submission contains nulls"
prob_sums = submission[TARGET_COLUMNS].sum(axis=1).values
np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5, err_msg="Probabilities do not sum to 1.0")

# Write Submission CSV
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\\nSubmission verified and successfully written to {SUBMISSION_PATH} ({len(submission)} rows).")
print(submission.head())
""")

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

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook generated successfully at {NOTEBOOK_PATH}")
