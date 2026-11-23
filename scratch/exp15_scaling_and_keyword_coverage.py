import math
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
for p in (BASE_DIR, SRC_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_dataset, get_stratified_group_folds, TARGET_COLUMNS
from features import compute_pairwise_features, RE_WORDS, STOPWORDS
from metric import compute_log_loss
from calibration import DirichletSymmetricCalibrator
from models import get_hist_gbdt_loss, SymmetricModelWrapper

assert torch.cuda.is_available(), "CUDA required"
device = torch.device("cuda:0")

CHAMPION_10K_BENCHMARK = 1.01601


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
    def __init__(self, feature_names: list, hidden_dim: int = 128, lr: float = 2e-3, epochs: int = 20, batch_size: int = 512):
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
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
        if self.model is None or self.mean is None or self.std is None:
            raise ValueError("Model must be fitted before predict_proba")
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_norm = (X_t - self.mean) / self.std
        X_swap_norm = self._swap_features_tensor(X_norm)

        with torch.no_grad():
            logits = self.model(X_norm, X_swap_norm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class ScaledDualLSAExtractor:
    """
    Enhanced Dual LSA Extractor:
    1. Word LSA: k=32 (67 dims)
    2. Char LSA: k=24 (51 dims)
    3. Multi-Turn Trajectory Features (3 dims)
    4. Rare Prompt Keyword Coverage (3 dims)
    Total expected feature space: 72 base + 3 tfidf + 67 word_lsa + 51 char_lsa + 6 new = 199 features.
    """
    def __init__(self, n_word_lsa: int = 32, n_char_lsa: int = 24):
        self.n_word_lsa = n_word_lsa
        self.n_char_lsa = n_char_lsa
        self.word_vectorizer: Optional[TfidfVectorizer] = None
        self.word_svd: Optional[TruncatedSVD] = None
        self.char_vectorizer: Optional[TfidfVectorizer] = None
        self.char_svd: Optional[TruncatedSVD] = None
        self.feature_names_: List[str] = []
        self.base_feature_names_: List[str] = []
        self.idf_dict_: Dict[str, float] = {}
        self.idf_median_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "ScaledDualLSAExtractor":
        sample_feat = compute_pairwise_features(
            prompt=df["prompt_text"].iloc[0],
            resp_a=df["response_a_text"].iloc[0],
            resp_b=df["response_b_text"].iloc[0],
        )
        self.base_feature_names_ = sorted(list(sample_feat.keys()))

        corpus = (
            list(df["prompt_text"].values)
            + list(df["response_a_text"].values)
            + list(df["response_b_text"].values)
        )

        # 1. Word TF-IDF + SVD
        self.word_vectorizer = TfidfVectorizer(
            max_features=20000,
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        word_mat = self.word_vectorizer.fit_transform(corpus)
        actual_k_word = min(self.n_word_lsa, max(1, min(word_mat.shape[0], word_mat.shape[1]) - 1))
        self.word_svd = TruncatedSVD(n_components=actual_k_word, random_state=42)
        self.word_svd.fit(word_mat)

        # IDF dictionary for rare keywords
        vocab = self.word_vectorizer.vocabulary_
        idf = self.word_vectorizer.idf_
        self.idf_dict_ = {w: float(idf[idx]) for w, idx in vocab.items()}
        self.idf_median_ = float(np.median(idf))

        # 2. Char TF-IDF + SVD
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=15000,
            sublinear_tf=True,
        )
        char_mat = self.char_vectorizer.fit_transform(corpus)
        actual_k_char = min(self.n_char_lsa, max(1, min(char_mat.shape[0], char_mat.shape[1]) - 1))
        self.char_svd = TruncatedSVD(n_components=actual_k_char, random_state=42)
        self.char_svd.fit(char_mat)

        # Build feature names
        names = list(self.base_feature_names_)

        # TF-IDF Cosine
        names.extend(["diff_tfidf_prompt_sim", "sym_tfidf_ab_sim", "sym_tfidf_mean_prompt"])

        # Word LSA
        names.extend(["diff_lsa_prompt_cos", "sym_lsa_ab_cos", "sym_lsa_mean_prompt_cos"])
        for i in range(self.n_word_lsa):
            names.append(f"diff_lsa_dim_{i}")
            names.append(f"sym_lsa_dim_{i}")

        # Char LSA
        names.extend(["diff_char_lsa_prompt_cos", "sym_char_lsa_ab_cos", "sym_char_lsa_mean_prompt_cos"])
        for i in range(self.n_char_lsa):
            names.append(f"diff_char_lsa_dim_{i}")
            names.append(f"sym_char_lsa_dim_{i}")

        # Rare Keyword Coverage
        names.extend(["diff_rare_kw_cov", "sym_mean_rare_kw_cov", "sym_min_rare_kw_cov"])

        # Multi-Turn Dynamics
        names.extend(["diff_final_turn_len", "sym_final_turn_len", "diff_turn_growth"])

        self.feature_names_ = names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        p_texts = df["prompt_text"].values
        a_texts = df["response_a_text"].values
        b_texts = df["response_b_text"].values
        a_turns_list = df["response_a_turns"].values
        b_turns_list = df["response_b_turns"].values
        a_turns_count = [len(t) for t in a_turns_list]
        b_turns_count = [len(t) for t in b_turns_list]

        # 1. Base hand-crafted features
        base_rows = []
        for p, a, b, ta, tb in zip(p_texts, a_texts, b_texts, a_turns_count, b_turns_count):
            feat_dict = compute_pairwise_features(p, a, b, ta, tb)
            base_rows.append([feat_dict[k] for k in self.base_feature_names_])
        X_base = np.asarray(base_rows, dtype=np.float32)

        # 2. Word TF-IDF & Word LSA
        assert self.word_vectorizer is not None and self.word_svd is not None
        p_w = self.word_vectorizer.transform(df["prompt_text"])
        a_w = self.word_vectorizer.transform(df["response_a_text"])
        b_w = self.word_vectorizer.transform(df["response_b_text"])

        sim_pa = np.asarray(p_w.multiply(a_w).sum(axis=1)).flatten()
        sim_pb = np.asarray(p_w.multiply(b_w).sum(axis=1)).flatten()
        sim_ab = np.asarray(a_w.multiply(b_w).sum(axis=1)).flatten()
        diff_tfidf = sim_pa - sim_pb
        sym_tfidf_ab = sim_ab
        sym_tfidf_mean = 0.5 * (sim_pa + sim_pb)
        X_tfidf = np.column_stack([diff_tfidf, sym_tfidf_ab, sym_tfidf_mean]).astype(np.float32)

        p_wsvd = self.word_svd.transform(p_w)
        a_wsvd = self.word_svd.transform(a_w)
        b_wsvd = self.word_svd.transform(b_w)

        k_w = self.n_word_lsa
        if a_wsvd.shape[1] < k_w:
            pad = k_w - a_wsvd.shape[1]
            p_wsvd = np.pad(p_wsvd, ((0, 0), (0, pad)), mode="constant")
            a_wsvd = np.pad(a_wsvd, ((0, 0), (0, pad)), mode="constant")
            b_wsvd = np.pad(b_wsvd, ((0, 0), (0, pad)), mode="constant")

        def l2_norm(mat: np.ndarray) -> np.ndarray:
            return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)

        p_wnorm = l2_norm(p_wsvd)
        a_wnorm = l2_norm(a_wsvd)
        b_wnorm = l2_norm(b_wsvd)

        cos_wpa = np.sum(p_wnorm * a_wnorm, axis=1, keepdims=True)
        cos_wpb = np.sum(p_wnorm * b_wnorm, axis=1, keepdims=True)
        cos_wab = np.sum(a_wnorm * b_wnorm, axis=1, keepdims=True)

        diff_wcos = (cos_wpa - cos_wpb).astype(np.float32)
        sym_wcos = cos_wab.astype(np.float32)
        sym_mean_wcos = (0.5 * (cos_wpa + cos_wpb)).astype(np.float32)

        diff_wsvd = (a_wsvd - b_wsvd).astype(np.float32)
        sym_wsvd = (0.5 * (a_wsvd + b_wsvd)).astype(np.float32)
        w_dim_cols = []
        for i in range(k_w):
            w_dim_cols.append(diff_wsvd[:, i : i + 1])
            w_dim_cols.append(sym_wsvd[:, i : i + 1])
        X_wlsa = np.hstack([diff_wcos, sym_wcos, sym_mean_wcos] + w_dim_cols)

        # 3. Char LSA
        assert self.char_vectorizer is not None and self.char_svd is not None
        p_c = self.char_vectorizer.transform(df["prompt_text"])
        a_c = self.char_vectorizer.transform(df["response_a_text"])
        b_c = self.char_vectorizer.transform(df["response_b_text"])

        p_csvd = self.char_svd.transform(p_c)
        a_csvd = self.char_svd.transform(a_c)
        b_csvd = self.char_svd.transform(b_c)

        k_c = self.n_char_lsa
        if a_csvd.shape[1] < k_c:
            pad = k_c - a_csvd.shape[1]
            p_csvd = np.pad(p_csvd, ((0, 0), (0, pad)), mode="constant")
            a_csvd = np.pad(a_csvd, ((0, 0), (0, pad)), mode="constant")
            b_csvd = np.pad(b_csvd, ((0, 0), (0, pad)), mode="constant")

        p_cnorm = l2_norm(p_csvd)
        a_cnorm = l2_norm(a_csvd)
        b_cnorm = l2_norm(b_csvd)

        cos_cpa = np.sum(p_cnorm * a_cnorm, axis=1, keepdims=True)
        cos_cpb = np.sum(p_cnorm * b_cnorm, axis=1, keepdims=True)
        cos_cab = np.sum(a_cnorm * b_cnorm, axis=1, keepdims=True)

        diff_ccos = (cos_cpa - cos_cpb).astype(np.float32)
        sym_ccos = cos_cab.astype(np.float32)
        sym_mean_ccos = (0.5 * (cos_cpa + cos_cpb)).astype(np.float32)

        diff_csvd = (a_csvd - b_csvd).astype(np.float32)
        sym_csvd = (0.5 * (a_csvd + b_csvd)).astype(np.float32)
        c_dim_cols = []
        for i in range(k_c):
            c_dim_cols.append(diff_csvd[:, i : i + 1])
            c_dim_cols.append(sym_csvd[:, i : i + 1])
        X_clsa = np.hstack([diff_ccos, sym_ccos, sym_mean_ccos] + c_dim_cols)

        # 4. Rare Keyword Coverage (exact anti-symmetry)
        rare_cov_rows = []
        for p, a, b in zip(p_texts, a_texts, b_texts):
            p_words = set(RE_WORDS.findall(p.lower())) - STOPWORDS
            a_words = set(RE_WORDS.findall(a.lower())) - STOPWORDS
            b_words = set(RE_WORDS.findall(b.lower())) - STOPWORDS

            # Rare words in prompt
            rare_p = {w for w in p_words if self.idf_dict_.get(w, 0.0) >= self.idf_median_}
            n_rare = len(rare_p)
            if n_rare == 0:
                rare_p = p_words
                n_rare = len(rare_p)

            if n_rare > 0:
                cov_a = len(rare_p & a_words) / n_rare
                cov_b = len(rare_p & b_words) / n_rare
            else:
                cov_a, cov_b = 0.0, 0.0

            rare_cov_rows.append([cov_a - cov_b, 0.5 * (cov_a + cov_b), min(cov_a, cov_b)])
        X_rare = np.asarray(rare_cov_rows, dtype=np.float32)

        # 5. Multi-Turn Trajectory Dynamics
        turn_rows = []
        for ta, tb in zip(a_turns_list, b_turns_list):
            len_a_last = len(ta[-1]) if ta else 0
            len_b_last = len(tb[-1]) if tb else 0
            len_a_first = len(ta[0]) if ta else 0
            len_b_first = len(tb[0]) if tb else 0

            diff_last = float(len_a_last - len_b_last)
            sym_last = 0.5 * float(len_a_last + len_b_last)
            diff_growth = float((len_a_last - len_a_first) - (len_b_last - len_b_first))
            turn_rows.append([diff_last, sym_last, diff_growth])
        X_turns = np.asarray(turn_rows, dtype=np.float32)

        return np.hstack([X_base, X_tfidf, X_wlsa, X_clsa, X_rare, X_turns]).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


if __name__ == "__main__":
    train_csv = os.path.join(BASE_DIR, "data", "train.csv")
    print("================================================================")
    print("EXP 15: SCALED DUAL LSA (WORD 32 + CHAR 24) & KEYWORD COVERAGE")
    print(f"Champion 10k Benchmark to beat: {CHAMPION_10K_BENCHMARK:.5f}")
    print("================================================================")

    df_train = load_dataset(train_csv, nrows=10000, is_test=False)
    y = df_train["target_label"].values
    y_onehot = df_train[TARGET_COLUMNS].values
    folds = list(get_stratified_group_folds(df_train, n_splits=5))

    extractor = ScaledDualLSAExtractor(n_word_lsa=32, n_char_lsa=24)
    t0_feat = time.time()
    X = extractor.fit_transform(df_train)
    print(f"Feature Extraction Time: {time.time() - t0_feat:.1f}s", flush=True)
    print(f"Total Feature Space Extracted: {X.shape[1]} features", flush=True)

    oof_pytorch = np.zeros_like(y_onehot, dtype=np.float64)
    oof_lr = np.zeros_like(y_onehot, dtype=np.float64)
    oof_gbdt = np.zeros_like(y_onehot, dtype=np.float64)

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y_onehot[val_idx]

        # 1. PyTorch GPU Model
        pt_model = PyTorchGPUPreferenceModel(extractor.feature_names_, hidden_dim=128, epochs=20, batch_size=512)
        pt_model.fit(X_tr, y_tr)
        oof_pytorch[val_idx] = pt_model.predict_proba(X_va)

        # 2. Logistic Regression (C=0.5)
        scaler_lr = StandardScaler()
        # Pair-swap augmentation for LR
        diff_idx = [i for i, name in enumerate(extractor.feature_names_) if name.startswith("diff_")]
        X_swap_tr = X_tr.copy()
        X_swap_tr[:, diff_idx] = -X_swap_tr[:, diff_idx]
        y_swap_tr = y_tr.copy()
        y_swap_tr[y_tr == 0] = 1
        y_swap_tr[y_tr == 1] = 0
        X_aug = np.vstack([X_tr, X_swap_tr])
        y_aug = np.concatenate([y_tr, y_swap_tr])

        X_aug_sc = scaler_lr.fit_transform(X_aug)
        lr = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(X_aug_sc, y_aug)

        X_va_sc = scaler_lr.transform(X_va)
        X_va_swap = X_va.copy()
        X_va_swap[:, diff_idx] = -X_va_swap[:, diff_idx]
        X_va_swap_sc = scaler_lr.transform(X_va_swap)

        p_orig = lr.predict_proba(X_va_sc)
        p_swap = lr.predict_proba(X_va_swap_sc)
        p_sym = 0.5 * (p_orig + np.column_stack([p_swap[:, 1], p_swap[:, 0], p_swap[:, 2]]))
        oof_lr[val_idx] = p_sym / p_sym.sum(axis=1, keepdims=True)

        # 3. Tuned HistGBDT (max_leaf_nodes=45, min_samples_leaf=35, l2=1.0)
        gbdt = HistGradientBoostingClassifier(
            loss=get_hist_gbdt_loss(),
            max_iter=160,
            learning_rate=0.045,
            max_leaf_nodes=45,
            min_samples_leaf=35,
            l2_regularization=1.0,
            random_state=42,
        )
        gbdt.fit(X_aug_sc, y_aug)
        p_gb_orig = gbdt.predict_proba(X_va_sc)
        p_gb_swap = gbdt.predict_proba(X_va_swap_sc)
        p_gb_sym = 0.5 * (p_gb_orig + np.column_stack([p_gb_swap[:, 1], p_gb_swap[:, 0], p_gb_swap[:, 2]]))
        oof_gbdt[val_idx] = p_gb_sym / p_gb_sym.sum(axis=1, keepdims=True)

        print(f"  Fold {fold+1}/5 done ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n5-Fold Training Time: {time.time() - t0:.1f}s", flush=True)
    print(f"  PyTorch GPU OOF Log Loss: {compute_log_loss(y_onehot, oof_pytorch):.5f}", flush=True)
    print(f"  Logistic Regression OOF Log Loss: {compute_log_loss(y_onehot, oof_lr):.5f}", flush=True)
    print(f"  Tuned HistGBDT OOF Log Loss: {compute_log_loss(y_onehot, oof_gbdt):.5f}", flush=True)

    # Blend Evaluation with Dirichlet Calibrator
    print("\n--- Blend Optimization & Dirichlet Calibration ---", flush=True)
    best_loss = 999.0
    best_w: Optional[Tuple[float, float, float]] = None
    best_cal: Optional[DirichletSymmetricCalibrator] = None

    for w_pt in [0.05, 0.10, 0.15, 0.20]:
        for w_lr in [0.25, 0.30, 0.35, 0.40]:
            w_gb = 1.0 - w_pt - w_lr
            if w_gb < 0:
                continue
            b = w_lr * oof_lr + w_gb * oof_gbdt + w_pt * oof_pytorch
            b /= b.sum(axis=1, keepdims=True)
            c = DirichletSymmetricCalibrator().fit(b, y_onehot)
            l = compute_log_loss(y_onehot, c.transform(b))
            if l < best_loss:
                best_loss = l
                best_w = (float(w_lr), float(w_gb), float(w_pt))
                best_cal = c

    assert best_w is not None and best_cal is not None
    print(f"  Tuned Blend Dirichlet Log Loss: {best_loss:.5f} (Weights: LR={best_w[0]:.2f}, GBDT={best_w[1]:.2f}, GPU={best_w[2]:.2f})", flush=True)
    print(f"  Optimal Calibrator: T_pair={best_cal.t_pair:.4f}, Beta_tie={best_cal.beta_tie:.4f}, Alpha_tie={best_cal.alpha_tie:.4f}", flush=True)

    delta = best_loss - CHAMPION_10K_BENCHMARK
    print(f"  Delta vs Champion ({CHAMPION_10K_BENCHMARK:.5f}): {delta:+.5f}", flush=True)
    if delta < 0:
        print("  GATE STATUS: PASSED (Strict Monotonic Improvement)", flush=True)
    else:
        print("  GATE STATUS: FAILED (No Improvement)", flush=True)
