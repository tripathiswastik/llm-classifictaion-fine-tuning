import hashlib
import json
import os
from typing import Generator, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def parse_conversation_turns(raw_val: Union[str, list, None]) -> List[str]:
    """
    Safely deserializes a JSON array string representing multi-turn conversation messages.
    """
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
        # Fallback to literal string if malformed JSON
        return [s]


def flatten_turns(turns: List[str], separator: str = " \n\n ") -> str:
    """
    Combines conversation turns into a unified string.
    """
    return separator.join(turns).strip()


def hash_prompt(prompt_str: str) -> str:
    """
    Creates a deterministic 16-character SHA-256 hash of normalized prompt text for grouping.
    """
    normalized = " ".join(prompt_str.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def load_dataset(
    csv_path: str,
    nrows: Optional[int] = None,
    is_test: bool = False
) -> pd.DataFrame:
    """
    Loads raw CSV and applies parsed conversation transformations and prompt grouping keys.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    raw = pd.read_csv(csv_path, nrows=nrows)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from pd.read_csv, got {type(raw)}")
    df: pd.DataFrame = raw

    # Parse conversation turns
    df["prompt_turns"] = df["prompt"].apply(parse_conversation_turns)
    df["response_a_turns"] = df["response_a"].apply(parse_conversation_turns)
    df["response_b_turns"] = df["response_b"].apply(parse_conversation_turns)

    # Flatten text
    df["prompt_text"] = df["prompt_turns"].apply(flatten_turns)
    df["response_a_text"] = df["response_a_turns"].apply(flatten_turns)
    df["response_b_text"] = df["response_b_turns"].apply(flatten_turns)

    # Prompt group hash
    df["prompt_group"] = df["prompt_text"].apply(hash_prompt)

    if not is_test:
        # Ensure target labels are valid
        valid_mask = (df[TARGET_COLUMNS].sum(axis=1) == 1)
        if not valid_mask.all():
            raise ValueError(f"Found {(~valid_mask).sum()} rows where target sum != 1")

        # Discrete target: 0 -> winner_model_a, 1 -> winner_model_b, 2 -> winner_tie
        df["target_label"] = np.argmax(df[TARGET_COLUMNS].values, axis=1)

    return df


def get_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields (train_idx, val_idx) using StratifiedGroupKFold on prompt groups.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = df["prompt_group"].values
    labels = df["target_label"].values

    for train_idx, val_idx in sgkf.split(df, labels, groups=groups):
        yield train_idx, val_idx
