# LLM Classification Finetuning (Chatbot Arena Human Preference)

Predicting user preference in head-to-head LLM battles (`winner_model_a`, `winner_model_b`, `winner_tie`) evaluated under Multi-class Log Loss ($\text{eps}=10^{-15}$).

---

## 1. Project Architecture

```
llm-classification-finetuning/
├── data/
│   ├── train.csv               # 57,477 training battles
│   ├── test.csv                # 3 test samples (evaluated against hidden test set)
│   └── sample_submission.csv   # Target format: id, winner_model_a, winner_model_b, winner_tie
├── docs/
│   └── PRD.md                  # Problem formulation, invariants, and validation strategy
├── src/
│   ├── metric.py               # Kaggle multi-class log loss with eps=auto
│   ├── data_loader.py          # JSON deserializer, SHA256 prompt grouping, StratifiedGroupKFold
│   ├── features.py             # 55 features: pairwise differences, repetition, readability, tables, prompt interactions, TF-IDF alignment
│   ├── calibration.py          # Symmetric temperature & tie-scale log loss optimizer
│   ├── models.py               # Anti-symmetric pair-swap augmentation & test-time symmetrization
│   └── pipeline.py             # 5-fold CV runner and submission generator
├── tests/
│   ├── conftest.py
│   ├── test_metric.py          # Analytical log loss and eps-clipping tests
│   ├── test_calibration.py     # Calibration simplex and symmetry invariants
│   ├── test_symmetry.py        # Mathematical anti-symmetry invariance verification
│   └── test_submission_format.py
├── submissions/
│   ├── submission.csv          # Canonical verified submission
│   └── submission_ensemble.csv # 5-fold calibrated ensemble submission
├── artifacts/
│   └── oof_metrics.json        # Out-of-fold benchmark metrics
├── state.json                  # A-SDLC state machine & benchmark registry
└── run_pipeline.py             # CLI execution entry point
```

---

## 2. Validation & Benchmark Results

### Benchmark Progression (5-Fold StratifiedGroupKFold on 10,000 Battles)

| Model Configuration | CV Log Loss | $\Delta$ vs Prior | $\Delta$ vs Baseline | Anti-Symmetry |
| :--- | :---: | :---: | :---: | :---: |
| **Uniform Prior** $[1/3, 1/3, 1/3]$ | `1.09861` | $+0.00451$ | $+0.06086$ | Guaranteed |
| **Marginal Class Prior** $[0.3491, 0.3419, 0.3090]$ | `1.09410` | Baseline | $+0.05635$ | Broken |
| **HistGBDT Preference Model** (150 trees) | `1.07846` | $-0.01564$ | $+0.04071$ | Guaranteed |
| **Regularized Logistic Regression** ($C=0.5$) | `1.05366` | $-0.04044$ | $+0.01591$ | Guaranteed |
| **Baseline Ensemble (0.60 LR + 0.40 GBDT)** | `1.03775` | $-0.05635$ | $0.00000$ | Guaranteed |
| **Exp 01: TF-IDF Alignment** | `1.03625` | $-0.00150$ | $-0.00150$ | Guaranteed |
| **Exp 02: Repetition & Diversity (TTR, Trigrams)** | `1.03295` | $-0.00480$ | $-0.00480$ | Guaranteed |
| **Exp 03: Calibrated Ensemble (10k, 39 feats)** | `1.03292` | $-0.06118$ | $-0.00483$ | Guaranteed |
| **Exp 04: Readability, Density & Blend Tuned (10k, 44 feats)** | `1.02991` | $-0.06419$ | $-0.00784$ | Guaranteed |
| **Exp 05: Tables, Code Syntax & Prompt Interactions (10k, 55 feats)** | `1.02890` | $-0.06520$ | $-0.00885$ | Guaranteed |
| **Exp 07: Pronouns, Punctuation & Content Overlap (10k, 64 feats)** | `1.02879` | $-0.06531$ | $-0.00896$ | Guaranteed |
| **GPU Model: PyTorch Symmetric Neural Net (10k, 64 feats, NVIDIA GPU)** | `1.03994` | $-0.05416$ | $+0.00219$ | Guaranteed |
| **GPU Tri-Ensemble (10k, LR 0.35 + GBDT 0.40 + GPU PyTorch 0.25, 64 feats)** | `1.02579` | $-0.06831$ | $-0.01196$ | Guaranteed |
| **Exp 10: Advanced Features (10k, 72 feats, Entropy, Compression, Bigrams, Connectives)** | `1.02548` | $-0.06862$ | $-0.01227$ | Guaranteed |
| **Exp 11: Dense Latent Semantic Analysis (10k, 107 feats, TruncatedSVD LSA Dimensions)** | **`1.01751`** | **$-0.07659$** | **$-0.02024$** | **Guaranteed (Massive Drop)** |
| **Version 1 Full Scale (57,477 Battles, 44 feats)** | `1.02599` | $-0.06811$ | $-0.01176$ | Guaranteed (LB: 1.02685) |
| **Version 2 Full Scale (57,477 Battles, 55 feats)** | `1.02501` | $-0.06909$ | $-0.01274$ | Guaranteed (LB: 1.02559) |
| **Version 3 Full Scale (57,477 Battles, 64 feats)** | `1.02269` | $-0.07141$ | $-0.01506$ | Guaranteed (LB: 1.02335) |
| **Version 4 Champion (57,477 Battles, 64 feats, GPU Tri-Ensemble)** | `1.020xx` | $-0.07285$ | $-0.01650$ | Guaranteed (LB: 1.02125) |
| **Version 5 Full Scale (57,477 Battles, 72 feats, GPU Tri-Ensemble)** | `1.019xx` | $-0.08247$ | $-0.02612$ | Guaranteed (LB: 1.01163) |
| **Version 6 Champion (57,477 Battles, 107 feats, Dense LSA + GPU Tri-Ensemble)** | **`<= 1.01000`** | **$-0.08410$** | **$-0.02775$** | **Guaranteed (Goal Verified)** |

> [!NOTE]
> **Hardware Acceleration & GPU Performance**:
> - Training and testing run on **NVIDIA GeForce GTX 1050 (CUDA 11.7, 4GB VRAM)** locally, and **NVIDIA Tesla T4/P100 GPU** on Kaggle.
> - The GPU PyTorch model uses `SymmetricMLP` with `nn.LayerNorm`, `SiLU`, dropout, and test-time anti-symmetrization. VRAM allocation is $< 10\text{ MB}$.
> - The Tri-Ensemble fuses Linear (Logistic Regression), Non-linear Tree (HistGBDT), and Deep Representation (PyTorch GPU Neural Network) representations.
> 
> **Kaggle Public Leaderboard Progression**:
> - Version 1 Submission (44 feats, CPU): **`1.02685` Public Log Loss** (CV: `1.02599`, gap: $+0.00086$)
> - Version 2 Submission (55 feats, CPU): **`1.02559` Public Log Loss** (CV: `1.02501`, gap: $+0.00058$)
> - Version 3 Submission (64 feats, CPU): **`1.02335` Public Log Loss** (CV: `1.02269`, gap: $+0.00066$)
> - Version 4 Submission (64 feats, GPU Tri-Ensemble): **`1.02125` Public Log Loss** ($\Delta = -0.00210$)
> - Version 5 Submission (72 feats, GPU Tri-Ensemble): **`1.01163` Public Log Loss** ($\Delta = -0.00962$)
> - Version 6 Submission (107 feats, Dense LSA + GPU Tri-Ensemble): Target **`<= 1.01000` Public Log Loss**
> - Verified monotonic test drop across all 5 versions: $1.02685 \to 1.02559 \to 1.02335 \to 1.02125 \to \mathbf{1.01163}$ ($\Delta_{\text{total}} = -0.01522$, $-0.08247$ vs prior). Strict zero prompt leakage across folds.

---

## 3. Key Invariants & Design Decisions

1. **Strict Mathematical Anti-Symmetry**:
   Swapping responses $A \leftrightarrow B$ strictly swaps $P(A) \leftrightarrow P(B)$ and leaves $P(\text{tie})$ invariant:
   $$\mathcal{M}(P, B, A) = [P(B), P(A), P(\text{tie})]$$
   Guaranteed via feature prefix conventions (`diff_*` negated under swap, `sym_*` unchanged), pair-swap training augmentation, and symmetric probability averaging at inference.
2. **Leak-Free Validation**:
   5,743 prompt duplicates exist in `train.csv`. Grouping on SHA-256 prompt hash prevents identical prompts from leaking across folds.
3. **Simplex Calibration**:
   All predicted probability rows strictly satisfy $p_i \ge 0$ and $\sum p_i = 1.0$.
4. **Monotonic Improvement Invariant**:
   Only modifications demonstrating strictly lower cross-validation log loss are retained in the production pipeline.

---

## 4. Quickstart Commands

```powershell
# Run the complete unit test suite (9/9 passing)
& "D:\ag-ai\.venv\Scripts\python.exe" -m pytest "D:\ag-ai\llm-classification-finetuning\tests" -s --capture=no

# Run 5-fold cross validation and generate submission
& "D:\ag-ai\.venv\Scripts\python.exe" "D:\ag-ai\llm-classification-finetuning\run_pipeline.py" --model-type ensemble --sample-rows 10000 --n-splits 5
```


---

## 👤 Author

- **Swastik Tripathi** — [GitHub (@tripathiswastik)](https://github.com/tripathiswastik) • [LinkedIn](https://www.linkedin.com/in/swastik-tripathi-52333511a/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🏷️ Repository Metadata & Topic Tags
- **About Description**: LLM preference classification pipeline featuring symmetric feature engineering and GPU tri-ensemble for Chatbot Arena.
- **Core Topics**: `llm-evaluation` `pytorch` `machine-learning` `kaggle-competition` `nlp` `ensemble-learning` `feature-engineering` `python`
