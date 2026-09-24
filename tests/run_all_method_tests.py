"""Run the synthetic tests required before accepting real MOVER output."""

from test_build_mover_features_v4 import main as test_features
from test_fit_primary_model_v4 import main as test_model
from test_harmonize_features_v4 import main as test_harmonize
from test_validate_mover_export_v2 import main as test_validator


if __name__ == "__main__":
    test_validator()
    test_features()
    test_harmonize()
    test_model()
    print("PASS: all v4 synthetic methodology tests")
