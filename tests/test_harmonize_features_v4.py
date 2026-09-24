"""Synthetic schema test for two-center harmonization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from harmonize_features_v4 import COMMON_COLUMNS, harmonize


def common_row() -> dict:
    return {
        "age": 60, "sex": "Female", "asa": 3, "anesthesia_minutes": 180,
        "surgical_category": "abdominal", "common_surgical_support": True,
        "baseline_creatinine": 0.9, "baseline_egfr": 75, "icu_flag": "N",
        "map_observed_minutes": 175, "map_total_minutes": 180, "map_coverage": 0.972,
        "total_low_minutes": 20, "episode_count": 4, "longest_episode_minutes": 5,
        "excess5_minutes": 0, "excess10_minutes": 0,
        "auc_below_65_mmhg_minutes": 100, "mean_depth_below_65_mmhg": 5,
        "any_hypotension": 1, "episode_durations_json": "[5,5,5,5]",
        "outcome_observed_48h": True, "postop_creatinine_count_48h": 2,
        "first_postop_creatinine_hours": 8, "early_aki_48h": False,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="harmonize_test_") as temp:
        root = Path(temp)
        mover = common_row() | {
            "export_case": 1, "export_patient": 1,
            "hypertension_flag": True, "diabetes_flag": False,
        }
        vital = common_row() | {
            "vital_case": 2, "vital_patient": 2,
            "preop_htn": False, "preop_dm": True,
        }
        mover_path = root / "mover.csv"
        vital_path = root / "vital.csv"
        output = root / "combined"
        pd.DataFrame([mover]).to_csv(mover_path, index=False)
        pd.DataFrame([vital]).to_csv(vital_path, index=False)
        report = harmonize(mover_path, vital_path, output)
        combined = pd.read_csv(output / "COMBINED_FEATURES_V4.csv")
        assert combined.columns.tolist() == COMMON_COLUMNS
        assert set(combined.center) == {"MOVER", "VitalDB"}
        assert combined.analysis_case.is_unique
        assert report["consistency"]["episode_json_parse_failures"] == 0
        print("PASS: synthetic two-center harmonization")


if __name__ == "__main__":
    main()
