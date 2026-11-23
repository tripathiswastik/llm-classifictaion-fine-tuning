import numpy as np
import pandas as pd
import pytest
try:
    from src.features import FeatureExtractor
    from src.models import (
        build_ensemble_model,
        build_hist_gbdt_model,
        build_logistic_model,
    )
except (ImportError, AttributeError):
    from features import FeatureExtractor  # type: ignore
    from models import (  # type: ignore
        build_ensemble_model,
        build_hist_gbdt_model,
        build_logistic_model,
    )


def test_anti_symmetry_property():
    df_train = pd.DataFrame({
        "prompt_text": [
            "What is the capital of France?",
            "Explain quantum computing simply.",
            "Write a poem about trees.",
            "Solve 2+2."
        ],
        "response_a_text": [
            "The capital of France is Paris. It is known as the City of Light.",
            "Quantum computing uses qubits instead of classical bits to compute.",
            "Trees stand tall and green, the forest queen.",
            "The answer is 4."
        ],
        "response_b_text": [
            "Paris.",
            "Quantum computers are machines that use physics to solve complex problems very quickly.",
            "Leaves flutter softly in the gentle morning breeze.",
            "4."
        ],
        "response_a_turns": [[""], [""], [""], [""]],
        "response_b_turns": [[""], [""], [""], [""]],
        "target_label": [0, 1, 2, 2]
    })

    extractor = FeatureExtractor()
    X_train = extractor.fit_transform(df_train)
    y_train = df_train["target_label"].values

    model = build_logistic_model(extractor.feature_names_)
    model.fit(X_train, y_train)

    # Create test sample (A, B) and swapped sample (B, A)
    df_test_ab = pd.DataFrame({
        "prompt_text": ["Tell me a joke."],
        "response_a_text": ["Why did the chicken cross the road? To get to the other side!"],
        "response_b_text": ["Two bytes meet. One says: I'm feeling sick. The other replies: Bit of a cold?"],
        "response_a_turns": [[""]],
        "response_b_turns": [[""]]
    })

    df_test_ba = pd.DataFrame({
        "prompt_text": ["Tell me a joke."],
        "response_a_text": ["Two bytes meet. One says: I'm feeling sick. The other replies: Bit of a cold?"],
        "response_b_text": ["Why did the chicken cross the road? To get to the other side!"],
        "response_a_turns": [[""]],
        "response_b_turns": [[""]]
    })

    X_ab = extractor.transform(df_test_ab)
    X_ba = extractor.transform(df_test_ba)

    p_ab = model.predict_proba(X_ab)[0]
    p_ba = model.predict_proba(X_ba)[0]

    # Invariance assertions
    assert np.isclose(p_ab[0], p_ba[1], atol=1e-5)
    assert np.isclose(p_ab[1], p_ba[0], atol=1e-5)
    assert np.isclose(p_ab[2], p_ba[2], atol=1e-5)


def test_ensemble_anti_symmetry_property():
    df_train = pd.DataFrame({
        "prompt_text": [
            "What is 2+2?",
            "What is 3+3?",
            "What is 4+4?",
            "What is 5+5?"
        ],
        "response_a_text": [
            "It is 4.",
            "Six.",
            "Eight.",
            "Ten."
        ],
        "response_b_text": [
            "Four.",
            "It is 6.",
            "Eight is the answer.",
            "10."
        ],
        "response_a_turns": [[""], [""], [""], [""]],
        "response_b_turns": [[""], [""], [""], [""]],
        "target_label": [0, 1, 2, 2]
    })

    extractor = FeatureExtractor()
    X_train = extractor.fit_transform(df_train)
    y_train = df_train["target_label"].values

    ens = build_ensemble_model(extractor.feature_names_, lr_weight=0.60)
    ens.fit(X_train, y_train)

    df_test_ab = pd.DataFrame({
        "prompt_text": ["Test prompt"],
        "response_a_text": ["Longer and detailed answer."],
        "response_b_text": ["Short."],
        "response_a_turns": [[""]],
        "response_b_turns": [[""]]
    })

    df_test_ba = pd.DataFrame({
        "prompt_text": ["Test prompt"],
        "response_a_text": ["Short."],
        "response_b_text": ["Longer and detailed answer."],
        "response_a_turns": [[""]],
        "response_b_turns": [[""]]
    })

    X_ab = extractor.transform(df_test_ab)
    X_ba = extractor.transform(df_test_ba)

    p_ab = ens.predict_proba(X_ab)[0]
    p_ba = ens.predict_proba(X_ba)[0]

    assert np.isclose(p_ab[0], p_ba[1], atol=1e-5)
    assert np.isclose(p_ab[1], p_ba[0], atol=1e-5)
    assert np.isclose(p_ab[2], p_ba[2], atol=1e-5)

