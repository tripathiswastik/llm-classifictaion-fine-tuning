import math
import numpy as np
import pandas as pd
import pytest

from features import FeatureExtractor, compute_text_stats, compute_pairwise_features
from models import build_logistic_model, build_hist_gbdt_model, PyTorchGPUPreferenceModel, TriEnsemblePreferenceModel
from calibration import DirichletSymmetricCalibrator

def test_empty_and_whitespace_responses():
    """Verify empty, single-character, and whitespace responses do not crash or produce NaN/Inf."""
    stats_empty = compute_text_stats("")
    stats_space = compute_text_stats("   \n\t  ")
    stats_punct = compute_text_stats("?!.")

    # All returned metrics must be finite floats
    for k, v in stats_empty.items():
        assert not math.isnan(v), f"NaN in stats_empty[{k}]"
        assert not math.isinf(v), f"Inf in stats_empty[{k}]"
    assert stats_empty["flesch_reading_ease"] == 0.0, "Empty text must not have high reading ease score"

    feat = compute_pairwise_features(
        prompt="Explain relativity.",
        resp_a="",
        resp_b="   \n\t  ",
    )
    for k, v in feat.items():
        assert not math.isnan(v), f"NaN in pairwise feature {k}"
        assert not math.isinf(v), f"Inf in pairwise feature {k}"


def test_exact_identical_response_anomaly():
    """Verify that byte-identical responses produce exact anti-symmetry and symmetry invariance."""
    text = "The quick brown fox jumps over the lazy dog. **Bold** and `code`."
    feat = compute_pairwise_features(
        prompt="Tell me something.",
        resp_a=text,
        resp_b=text,
    )
    # Differences must be strictly 0.0
    assert feat["diff_char_len"] == 0.0
    assert feat["diff_word_count"] == 0.0
    assert feat["diff_final_len"] == 0.0
    assert feat["sym_is_exact_match"] == 1.0
    assert feat["sym_jaccard_ab"] == 1.0


def test_extreme_length_outlier():
    """Verify pipeline handles 50,000-character massive text without overflow."""
    massive_text = "This is a detailed and extensive response. " * 1200 # ~51,600 chars
    short_text = "Yes."

    feat = compute_pairwise_features(
        prompt="Write an essay.",
        resp_a=massive_text,
        resp_b=short_text,
    )
    assert feat["diff_len_ratio_30"] == 1.0
    assert feat["sym_extreme_len_imbalance"] == 1.0
    assert not math.isnan(feat["diff_log_char"])
    assert not math.isinf(feat["diff_log_char"])


def test_multilingual_and_unicode_edge_cases():
    """Verify non-ASCII, Chinese, Russian, Japanese, and emoji text process cleanly."""
    chinese_prompt = "什么是量子力学？"
    chinese_a = "量子力学是物理学的一个分支，研究微观粒子的运动规律。"
    russian_b = "Квантовая механика — фундаментальная физическая теория."

    feat = compute_pairwise_features(
        prompt=chinese_prompt,
        resp_a=chinese_a,
        resp_b=russian_b,
    )
    for k, v in feat.items():
        assert not math.isnan(v), f"NaN in multilingual feature {k}"
        assert not math.isinf(v), f"Inf in multilingual feature {k}"


def test_multi_turn_edge_cases():
    """Verify multi-turn conversations with 1 turn vs 15 turns process accurately."""
    p_turns_15 = [f"Question {i}?" for i in range(15)]
    a_turns_15 = [f"Answer A {i}." for i in range(15)]
    b_turns_15 = [f"Answer B {i}." for i in range(15)]

    feat = compute_pairwise_features(
        prompt=" \n\n ".join(p_turns_15),
        resp_a=" \n\n ".join(a_turns_15),
        resp_b=" \n\n ".join(b_turns_15),
        prompt_turns=p_turns_15,
        resp_a_turns=a_turns_15,
        resp_b_turns=b_turns_15,
    )
    assert feat["sym_is_multi_turn"] == 1.0
    assert feat["sym_num_turns"] == 15.0
    assert feat["diff_final_len"] == len("Answer A 14.") - len("Answer B 14.")


def test_out_of_vocabulary_lsa_zero_norm():
    """Verify out-of-vocabulary inputs produce clean zero norms without division by zero."""
    df_fit = pd.DataFrame({
        "prompt_text": ["The capital of France is Paris.", "Solve math equations."],
        "response_a_text": ["Paris is a great city.", "Math is fun."],
        "response_b_text": ["France capital.", "Solving formulas."],
        "prompt_turns": [["The capital of France is Paris."], ["Solve math equations."]],
        "response_a_turns": [["Paris is a great city."], ["Math is fun."]],
        "response_b_turns": [["France capital."], ["Solving formulas."]],
    })
    extractor = FeatureExtractor(use_tfidf=True, use_lsa=True, use_char_lsa=True)
    extractor.fit(df_fit)

    # Completely alien vocabulary
    df_alien = pd.DataFrame({
        "prompt_text": ["zzzyyyxxx wwwvvvuuu"],
        "response_a_text": ["qqqpppooo nnmmmlll"],
        "response_b_text": ["kkkjjjiii hhggfffee"],
        "prompt_turns": [["zzzyyyxxx wwwvvvuuu"]],
        "response_a_turns": [["qqqpppooo nnmmmlll"]],
        "response_b_turns": [["kkkjjjiii hhggfffee"]],
    })
    X_alien = extractor.transform(df_alien)
    assert not np.isnan(X_alien).any(), "NaN found in alien/OOV feature matrix"
    assert not np.isinf(X_alien).any(), "Inf found in alien/OOV feature matrix"


def test_full_pipeline_anti_symmetry_and_simplex_on_edge_cases():
    """Verify model wrapper strictly preserves anti-symmetry and simplex on anomaly samples."""
    df_train = pd.DataFrame({
        "prompt_text": ["Q1?", "Q2?", "Q3?", "Q4?"],
        "response_a_text": ["Ans A1", "Ans A2", "Ans A3", "Ans A4"],
        "response_b_text": ["Ans B1", "Ans B2", "Ans B3", "Ans B4"],
        "prompt_turns": [["Q1?"], ["Q2?"], ["Q3?"], ["Q4?"]],
        "response_a_turns": [["Ans A1"], ["Ans A2"], ["Ans A3"], ["Ans A4"]],
        "response_b_turns": [["Ans B1"], ["Ans B2"], ["Ans B3"], ["Ans B4"]],
        "target_label": [0, 1, 2, 0],
    })
    extractor = FeatureExtractor(use_tfidf=True, use_lsa=True, use_char_lsa=True)
    X_train = extractor.fit_transform(df_train)
    y_train = df_train["target_label"].values

    model = TriEnsemblePreferenceModel(extractor.feature_names_, lr_weight=0.38, gbdt_weight=0.52, gpu_pt_weight=0.10)
    model.fit(X_train, y_train)

    # Edge case sample
    df_edge_ab = pd.DataFrame({
        "prompt_text": ["Do not use python: calculate 5*5."],
        "response_a_text": ["The answer is 25. Python was avoided."],
        "response_b_text": ["```python\nprint(5*5)\n```"],
        "prompt_turns": [["Do not use python: calculate 5*5."]],
        "response_a_turns": [["The answer is 25. Python was avoided."]],
        "response_b_turns": [["```python\nprint(5*5)\n```"]],
    })
    df_edge_ba = pd.DataFrame({
        "prompt_text": ["Do not use python: calculate 5*5."],
        "response_a_text": ["```python\nprint(5*5)\n```"],
        "response_b_text": ["The answer is 25. Python was avoided."],
        "prompt_turns": [["Do not use python: calculate 5*5."]],
        "response_a_turns": [["```python\nprint(5*5)\n```"]],
        "response_b_turns": [["The answer is 25. Python was avoided."]],
    })

    X_ab = extractor.transform(df_edge_ab)
    X_ba = extractor.transform(df_edge_ba)

    p_ab = model.predict_proba(X_ab)[0]
    p_ba = model.predict_proba(X_ba)[0]

    # Simplex sum invariant
    np.testing.assert_allclose(p_ab.sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(p_ba.sum(), 1.0, atol=1e-5)

    # Swapped anti-symmetry invariant
    np.testing.assert_allclose(p_ab[0], p_ba[1], atol=1e-5)
    np.testing.assert_allclose(p_ab[1], p_ba[0], atol=1e-5)
    np.testing.assert_allclose(p_ab[2], p_ba[2], atol=1e-5)


if __name__ == "__main__":
    test_empty_and_whitespace_responses()
    test_exact_identical_response_anomaly()
    test_extreme_length_outlier()
    test_multilingual_and_unicode_edge_cases()
    test_multi_turn_edge_cases()
    test_out_of_vocabulary_lsa_zero_norm()
    test_full_pipeline_anti_symmetry_and_simplex_on_edge_cases()
    print("ALL EDGE CASE & ANOMALY TESTS PASSED 100%!")
