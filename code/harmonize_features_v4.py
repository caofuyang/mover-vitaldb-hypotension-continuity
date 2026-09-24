"""Create the locked two-center feature table without fitting an outcome model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_COLUMNS = [
    "center", "analysis_case", "analysis_patient", "age", "sex", "asa",
    "anesthesia_minutes", "surgical_category", "common_surgical_support",
    "hypertension", "diabetes", "baseline_creatinine", "baseline_egfr",
    "icu_flag", "map_observed_minutes", "map_total_minutes", "map_coverage",
    "total_low_minutes", "episode_count", "longest_episode_minutes",
    "excess5_minutes", "excess10_minutes", "auc_below_65_mmhg_minutes",
    "mean_depth_below_65_mmhg", "any_hypotension", "episode_durations_json",
    "outcome_observed_48h", "postop_creatinine_count_48h",
    "first_postop_creatinine_hours", "early_aki_48h",
]


def bool_series(value: pd.Series) -> pd.Series:
    return value.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def nullable_binary(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    text = value.astype(str).str.lower()
    numeric = numeric.mask(numeric.isna() & text.isin(["true", "yes", "y"]), 1)
    numeric = numeric.mask(numeric.isna() & text.isin(["false", "no", "n"]), 0)
    return numeric


def normalized_yes_no(value: pd.Series) -> pd.Series:
    text = value.fillna("Unknown").astype(str).str.strip().str.lower()
    return text.map(
        lambda item: "Yes" if item in {"yes", "y", "true", "1"}
        else ("No" if item in {"no", "n", "false", "0"} else "Unknown")
    )


def prepare_mover(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["center"] = "MOVER"
    data["analysis_case"] = "M_" + data.export_case.astype(str)
    data["analysis_patient"] = "M_" + data.export_patient.astype(str)
    data["hypertension"] = bool_series(data.hypertension_flag)
    data["diabetes"] = bool_series(data.diabetes_flag)
    return data


def prepare_vital(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["center"] = "VitalDB"
    data["analysis_case"] = "V_" + data.vital_case.astype(str)
    data["analysis_patient"] = "V_" + data.vital_patient.astype(str)
    data["hypertension"] = bool_series(data.preop_htn)
    data["diabetes"] = bool_series(data.preop_dm)
    return data


def harmonize(mover_csv: Path, vital_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    mover = prepare_mover(mover_csv)
    vital = prepare_vital(vital_csv)
    missing = {
        "MOVER": sorted(set(COMMON_COLUMNS) - set(mover.columns)),
        "VitalDB": sorted(set(COMMON_COLUMNS) - set(vital.columns)),
    }
    if any(missing.values()):
        raise ValueError("Missing harmonized columns: " + json.dumps(missing))
    combined = pd.concat([mover[COMMON_COLUMNS], vital[COMMON_COLUMNS]], ignore_index=True)
    combined["outcome_observed_48h"] = bool_series(combined.outcome_observed_48h)
    combined["common_surgical_support"] = bool_series(combined.common_surgical_support)
    combined["any_hypotension"] = pd.to_numeric(combined.any_hypotension, errors="coerce").fillna(0).astype(int)
    combined["early_aki_48h"] = nullable_binary(combined.early_aki_48h)
    combined["icu_flag"] = normalized_yes_no(combined.icu_flag)

    consistency = {
        "duplicate_case_ids": int(combined.analysis_case.duplicated().sum()),
        "duplicate_patient_ids_within_center": int(combined.analysis_patient.duplicated().sum()),
        "negative_total_low": int(pd.to_numeric(combined.total_low_minutes, errors="coerce").lt(0).sum()),
        "excess5_gt_total": int((pd.to_numeric(combined.excess5_minutes, errors="coerce") > pd.to_numeric(combined.total_low_minutes, errors="coerce")).sum()),
        "episode_json_parse_failures": 0,
    }
    for value in combined.episode_durations_json:
        try:
            episodes = json.loads(value)
            if not isinstance(episodes, list) or any((not isinstance(x, int)) or x <= 0 for x in episodes):
                consistency["episode_json_parse_failures"] += 1
        except Exception:
            consistency["episode_json_parse_failures"] += 1
    if any(consistency.values()):
        raise RuntimeError("Harmonization consistency failure: " + json.dumps(consistency))

    center_summary = []
    for center, group in combined.groupby("center"):
        observed = group[group.outcome_observed_48h]
        center_summary.append(
            {
                "center": center,
                "parent_rows": int(len(group)),
                "outcome_observed_rows": int(len(observed)),
                "aki_events": int(observed.early_aki_48h.fillna(0).sum()),
                "common_support_rows": int(group.common_surgical_support.sum()),
                "cases_longest_episode_gte_20": int(pd.to_numeric(group.longest_episode_minutes, errors="coerce").ge(20).sum()),
            }
        )
    report = {
        "purpose": "Two-center schema harmonization; no outcome-association model",
        "schema_version": "v4",
        "common_columns": COMMON_COLUMNS,
        "center_summary": center_summary,
        "consistency": consistency,
    }
    combined.to_csv(out_dir / "COMBINED_FEATURES_V4.csv", index=False)
    (out_dir / "HARMONIZATION_AUDIT_V4.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mover", type=Path, required=True)
    parser.add_argument("--vitaldb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(harmonize(args.mover, args.vitaldb, args.out_dir), indent=2, ensure_ascii=False))
