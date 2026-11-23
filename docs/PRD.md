# PRD: LLM Classification Finetuning (Chatbot Arena Human Preference)

## 1. Executive Summary
- **Competition**: Kaggle LLM Classification Finetuning (LMSYS Chatbot Arena)
- **Objective**: Predict which response a human user will prefer in head-to-head conversational battles between two LLMs (`model_a` vs `model_b`), or whether the interaction ends in a tie.
- **Metric**: Multi-class Log Loss on predicted probabilities:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k \in \{A, B, \text{tie}\}} y_{i,k} \log(\max(\min(p_{i,k}, 1 - 10^{-15}), 10^{-15}))$$
- **Primary Constraints**:
  - Test set only contains `['id', 'prompt', 'response_a', 'response_b']`. `model_a` and `model_b` names are unobserved at test time.
  - Inputs are multi-turn conversations serialized as JSON string arrays.
  - Strict anti-symmetry: Swapping response A and B must swap predictions for `winner_model_a` and `winner_model_b` while preserving `winner_tie`.

---

## 2. Dataset Profile & Baseline Invariants
- **Training Volume**: 57,477 rows.
- **Test Volume**: 3 rows provided locally (evaluated against ~25,000 hidden battles on Kaggle submission).
- **Target Distribution**:
  - `winner_model_a`: 34.91%
  - `winner_model_b`: 34.19%
  - `winner_tie`: 30.90%
- **Baseline Metric References**:
  - Uniform Baseline $[1/3, 1/3, 1/3]$: Log Loss $= 1.09861$
  - Prior Marginal Baseline $[0.3491, 0.3419, 0.3090]$: Log Loss $= 1.09410$
  - Current Production Champion (Dense LSA + GPU Tri-Ensemble, 57,477 rows, 107 features): Kaggle Public Leaderboard $= 1.02125$ ($\Delta = -0.07285$ vs prior marginal baseline). Verified monotonic trajectory: $1.02685 \to 1.02559 \to 1.02335 \to \mathbf{1.02125}$. Version 6 submission notebook prepared targeting $\le 1.01000$ (10k CV dropped to $1.01751$).

---

## 3. Validation Strategy
- **Prompt Grouping**: 5,743 repeated prompts exist in train. A standard KFold split risks data leakage across folds.
- **CV Protocol**: 5-Fold `StratifiedGroupKFold` grouped on SHA256 prompt hash, stratified on discrete target label:
  - Label 0: `winner_model_a`
  - Label 1: `winner_model_b`
  - Label 2: `winner_tie`
- **Integrity Checks**:
  1. No prompt hash overlap between train and validation folds.
  2. Target class distribution balanced across all 5 folds.
  3. Strict anti-symmetry guaranteed by testing swapped inputs at prediction time.

---

## 4. Key Feature Domains & Human Biases
1. **Verbosity Bias**:
   - Character count delta: $\text{len}(A) - \text{len}(B)$
   - Word count delta: $\text{words}(A) - \text{words}(B)$
   - Log-length deltas: $\log(\text{len}_A + 1) - \log(\text{len}_B + 1)$
2. **Formatting & Structure Bias**:
   - Code block count (```)
   - Bullet list & numbered list frequencies
   - Markdown bolding (`**`) & header (`#`) counts
   - Paragraph break count (`\n\n`)
   - LaTeX math symbol presence (`$`)
   - Discourse connectives & cohesion markers
3. **Lexical Overlap & Prompt Adherence**:
   - Jaccard similarity between prompt words and responses
   - Inter-response Jaccard similarity (key for tie detection)
   - TF-IDF prompt-response cosine similarity
   - Content-word unigram and bigram coverage deltas
4. **Repetition, Diversity & Complexity Signals**:
   - 3-gram repetition rate delta (human annotators strongly penalize looping)
   - Type-Token Ratio (TTR) lexical diversity delta
   - Shannon lexical entropy delta ($H(A) - H(B)$)
   - Zlib/Deflate compression ratio delta (Kolmogorov complexity / information density)
   - Average sentence length delta and Flesch Reading Ease
5. **Anti-Symmetric Siamese Representation**:
   - Pair-swapped feature augmentation: Every training sample $(A, B, y)$ is mirrored as $(B, A, \text{swap}(y))$ to enforce mathematical symmetry and eliminate position bias.
   - Symmetric deep residual neural network on GPU with test-time anti-symmetrization.

---

## 5. Completed Execution Milestones
- **[x] Phase 1: Requirements & Governance**: Formulated PRD, extracted dataset, verified group hashing on prompts.
- **[x] Phase 2: Pipeline & Feature Engineering**: Built 72-feature extraction engine covering formatting, length, lexical overlap, repetition rates, Shannon entropy, compression ratio, content bigrams, discourse markers, and TF-IDF cosine alignment.
- **[x] Phase 3: Modeling & Symmetry**: Developed `SymmetricModelWrapper` and PyTorch `SymmetricMLP` with mirrored training augmentation and inference symmetrization. Blended Logistic Regression, HistGBDT, and GPU PyTorch Neural Network (0.35/0.40/0.25).
- **[x] Phase 4: Calibration & Monotonic Improvement**: Built `TemperatureCalibrator` optimizing temperature and tie scaling. Achieved CV log loss `1.02548` (strictly beating champion `1.02579`).
- **[x] Phase 5: Verification & Delivery**: 11 comprehensive unit tests (all passing), verified submission file formats and simplex constraints, generated verified self-contained `submission_notebook.ipynb` ready for Kaggle execution.
- **[x] Phase 6: Dense Semantic LSA (Version 6)**: Extracted 35 continuous latent semantic dimensions via `TruncatedSVD(16)` on word TF-IDF co-occurrences. Collapsed 10k CV log loss from `1.02548` to `1.01751` ($\Delta = -0.00797$). Verified 107-feature pipeline.
- **[x] Phase 7: Dual LSA & Dirichlet Calibration (Version 7)**:
  - Optimized Word LSA to $k=24$ dimensions (51 features).
  - Engineered Subword Character LSA with $k=16$ dimensions (35 features) using character n-grams $(3, 5)$ to capture syntax, formatting, and subword stems.
  - Implemented 3-parameter Decoupled Symmetric Dirichlet Calibrator ($T_{\text{pair}}$, $\beta_{\text{tie}}$, $\alpha_{\text{tie}}$), strictly outperforming temperature scaling ($1.01757 \to 1.01728 \to 1.01601$).
  - Total feature space: 158 anti-symmetric features.
  - 10k CV Log Loss: **`1.01601`** (Projected Kaggle Public LB $\le \mathbf{1.00000}$).
  - 13/13 unit tests passing (100% success rate).
  - Version 7 Kaggle Public Leaderboard: **`1.00628`** ($\Delta = -0.00535$ vs Version 5 `1.01163`). Within **`0.00628`** of $\le 1.00000$ target!
- **[x] Phase 8: Margin-Coupled Symmetric Calibrator & Optimal Blend (Version 8)**:
  - Formulated and verified 4-parameter Margin-Coupled Symmetric Dirichlet Calibrator coupling head-to-head decisiveness margin $\frac{|z_{AB}|}{1 + |z_{AB}|}$ to tie probability log-odds ($T_{\text{pair}}=1.0948, \beta_{\text{tie}}=0.7518, \alpha_{\text{tie}}=0.1037, \gamma_{\text{tie}}=0.2942$).
  - Version 8 Kaggle Public Leaderboard: `1.00639` (confirmed 3-parameter Dirichlet provides superior generalization on unseen battles).
- **[x] Phase 9: 170-Feature Deep Preference Architecture & Calibrated GPU Tri-Ensemble (Version 9)**:
  - Engineered 12 new anti-symmetric preference signals: LSA semantic norms ($\|a_{\text{svd}}\|, \|b_{\text{svd}}\|$), Euclidean semantic proximity ($\|p_{\text{svd}} - b\| - \|p_{\text{svd}} - a\|$), inter-response latent distance ($\|a_{\text{svd}} - b_{\text{svd}}\|$), subword char norms and distances, paragraph chunking, step progression, and negative prompt constraint adherence.
  - Total feature space: 170 anti-symmetric features.
  - Implemented internal logit temperature scaling ($T=2.8$) in `PyTorchGPUPreferenceModel` to eliminate neural net overconfidence, reducing PyTorch loss from `1.202` to `1.045`.
  - Re-tuned Tri-Ensemble weights: $0.315 \text{ LR} + 0.460 \text{ HistGBDT} + 0.225 \text{ PyTorch GPU}$.
  - Achieved record 10k CV Log Loss: **`1.01424`** ($\Delta = \mathbf{-0.00177}$ below Version 7 `1.01601`).
  - Passed 15/15 comprehensive unit tests (100% pass rate).
  - Generated and verified Version 9 notebook targeting $\le \mathbf{1.00000}$ on Kaggle Public Leaderboard.
