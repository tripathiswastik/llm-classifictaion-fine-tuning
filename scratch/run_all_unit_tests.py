import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

for p in (BASE_DIR, SRC_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_metric
import test_calibration
import test_submission_format
import test_gpu_model
import test_symmetry
import test_edge_cases_and_anomalies

print("=" * 60)
print("RUNNING FULL 4-PHASE GOVERNANCE TEST SUITE")
print("=" * 60)

# 1. Metric Tests
test_metric.test_perfect_predictions()
test_metric.test_uniform_baseline()
test_metric.test_eps_clipping_handles_zeros()
test_metric.test_evaluate_breakdown()
print("  [PASS] test_metric.py (4/4)")

# 2. Calibration Tests
test_calibration.test_temperature_calibrator_anti_symmetry()
test_calibration.test_temperature_calibrator_monotonic_loss()
test_calibration.test_dirichlet_calibrator_anti_symmetry()
test_calibration.test_dirichlet_calibrator_monotonic_loss()
test_calibration.test_margin_coupled_dirichlet_calibrator_anti_symmetry()
test_calibration.test_margin_coupled_dirichlet_calibrator_monotonic_loss()
print("  [PASS] test_calibration.py (6/6)")

# 3. Submission Format Tests
test_submission_format.test_submission_schema_and_simplex()
print("  [PASS] test_submission_format.py (1/1)")

# 4. GPU Model Tests
test_gpu_model.test_pytorch_gpu_model_symmetry_and_simplex()
test_gpu_model.test_tri_ensemble_model()
print("  [PASS] test_gpu_model.py (2/2)")

# 5. Full Symmetry & Feature Extractor Tests (Dual LSA 158 features)
test_symmetry.test_anti_symmetry_property()
test_symmetry.test_ensemble_anti_symmetry_property()
print("  [PASS] test_symmetry.py (2/2)")

# 6. Edge Case & Anomaly Invariant Tests (182 features & extreme boundaries)
test_edge_cases_and_anomalies.test_empty_and_whitespace_responses()
test_edge_cases_and_anomalies.test_exact_identical_response_anomaly()
test_edge_cases_and_anomalies.test_extreme_length_outlier()
test_edge_cases_and_anomalies.test_multilingual_and_unicode_edge_cases()
test_edge_cases_and_anomalies.test_multi_turn_edge_cases()
test_edge_cases_and_anomalies.test_out_of_vocabulary_lsa_zero_norm()
test_edge_cases_and_anomalies.test_full_pipeline_anti_symmetry_and_simplex_on_edge_cases()
print("  [PASS] test_edge_cases_and_anomalies.py (7/7)")

print("\n" + "=" * 60)
print(">>> ALL 22/22 UNIT & EDGE CASE TESTS PASSED WITH 100% INVARIANT CONFORMANCE <<<")
print("=" * 60)
