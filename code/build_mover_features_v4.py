"""Build prespecified MOVER analysis features from an accepted v2 export.

The script creates a one-operation-per-patient cohort before requiring a
postoperative creatinine value, derives 60-second hypotension episodes, applies
the locked creatinine and kidney-function rules, and records outcome
observability. It does not fit an exposure-outcome association model.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from surgical_category_audit import classify, egfr_2021, load_rules
from validate_mover_export_v2 import validate_export


COMMON_SUPPORT = {
    "abdominal",
    "thoracic",
    "vascular",
    "genitourinary_gynecologic",
    "superficial_other",
}


def resolve_creatinine(group: pd.DataFrame) -> pd.DataFrame:
    """Apply the locked same-time duplicate rule without outcome direction."""
    rows = []
    for time_seconds, values in group.groupby("time_seconds"):
        x = pd.to_numeric(values.creatinine, errors="coerce")
        x = x[np.isfinite(x) & x.gt(0) & x.ne(9999999)]
        if x.empty:
            continue
        spread = float(x.max() - x.min())
        if spread <= 0.1 + 1e-10:
            rows.append((float(time_seconds), float(x.median()), len(x), spread))
    return pd.DataFrame(rows, columns=["time_seconds", "creatinine", "source_rows", "spread"]).sort_values("time_seconds")


def episode_features(case_map: pd.DataFrame, anesthesia_minutes: float) -> dict:
    n = int(np.ceil(anesthesia_minutes))
    grid = np.full(n, np.nan)
    time = pd.to_numeric(case_map.time_seconds, errors="coerce").to_numpy(float)
    value = pd.to_numeric(case_map.map_value, errors="coerce").to_numpy(float)
    keep = np.isfinite(time) & np.isfinite(value) & (time >= 0) & (time < anesthesia_minutes * 60) & (value >= 20) & (value <= 160)
    bins = np.floor(time[keep] / 60).astype(int)
    if len(bins):
        medians = pd.Series(value[keep]).groupby(bins).median()
        valid_index = medians.index.to_numpy(int)
        valid_index = valid_index[(valid_index >= 0) & (valid_index < n)]
        grid[valid_index] = medians.loc[valid_index].to_numpy(float)
    observed = np.isfinite(grid)
    low = observed & (grid < 65)
    transitions = np.diff(np.r_[False, low, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    episodes = (stops - starts).astype(int).tolist()
    total = int(low.sum())
    depth_auc = float(np.where(low, 65 - grid, 0).sum())
    return {
        "map_observed_minutes": int(observed.sum()),
        "map_total_minutes": n,
        "map_coverage": float(observed.mean()) if n else np.nan,
        "total_low_minutes": total,
        "episode_count": len(episodes),
        "longest_episode_minutes": max(episodes) if episodes else 0,
        "excess5_minutes": int(sum(max(x - 5, 0) for x in episodes)),
        "excess10_minutes": int(sum(max(x - 10, 0) for x in episodes)),
        "auc_below_65_mmhg_minutes": depth_auc,
        "mean_depth_below_65_mmhg": depth_auc / total if total else 0.0,
        "any_hypotension": int(total > 0),
        "episode_durations_json": json.dumps(episodes, separators=(",", ":")),
    }


def build_features(
    archive: Path,
    out_dir: Path,
    minimum_paired_operations: int = 100,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    validation_path = out_dir / "MOVER_EXPORT_VALIDATION.json"
    validation = validate_export(
        archive,
        validation_path,
        minimum_paired_operations=minimum_paired_operations,
    )
    if validation["overall_status"] == "fail":
        raise RuntimeError(f"MOVER export failed validation; see {validation_path}")

    with zipfile.ZipFile(archive) as handle, tempfile.TemporaryDirectory(prefix="mover_features_") as temp:
        root = Path(temp)
        frames = {}
        for filename, key in [
            ("cohort_parent.csv", "cohort"),
            ("map.csv", "map"),
            ("creatinine.csv", "creatinine"),
        ]:
            member = next(x for x in handle.namelist() if Path(x).name == filename)
            path = root / filename
            path.write_bytes(handle.read(member))
            frames[key] = pd.read_csv(path)

    cohort = frames["cohort"].copy()
    map_data = frames["map"]
    creatinine = frames["creatinine"]
    rules = load_rules()
    cohort["surgical_category"] = [classify(x, "mover", rules) for x in cohort.procedure_name]

    audit = {
        "purpose": "Prespecified feature construction; no outcome-association model",
        "source_archive": archive.name,
        "flow": [{"stage": "exported_parent", "operations": int(len(cohort)), "patients": int(cohort.export_patient.nunique())}],
        "warnings": [],
    }
    eligible = cohort[
        ~cohort.cardiac_keyword_flag.astype(bool)
        & ~cohort.transplant_keyword_flag.astype(bool)
        & ~cohort.esrd_history_flag.astype(bool)
        & ~cohort.surgical_category.isin(["exclude_cardiac", "exclude_transplant", "unmapped"])
    ].copy()
    audit["flow"].append({"stage": "noncardiac_nontransplant_no_esrd_history", "operations": int(len(eligible)), "patients": int(eligible.export_patient.nunique())})

    cr_groups = {case: resolve_creatinine(group) for case, group in creatinine.groupby("export_case")}
    baseline_values = []
    for row in eligible.itertuples():
        lab = cr_groups.get(row.export_case, pd.DataFrame())
        if lab.empty:
            baseline_values.append(np.nan)
            continue
        baseline = lab[(lab.time_seconds >= -30 * 86400) & (lab.time_seconds < 0)]
        baseline_values.append(float(baseline.iloc[-1].creatinine) if len(baseline) else np.nan)
    eligible["baseline_creatinine"] = baseline_values
    female = eligible.sex.astype(str).str.lower().str.startswith("f")
    eligible["baseline_egfr"] = [
        egfr_2021(cr, age, is_female) if np.isfinite(cr) and np.isfinite(age) else np.nan
        for cr, age, is_female in zip(eligible.baseline_creatinine, eligible.age, female)
    ]
    eligible = eligible[
        eligible.baseline_creatinine.lt(4)
        & eligible.baseline_egfr.ge(15)
    ].copy()
    audit["flow"].append({"stage": "valid_baseline_kidney_function", "operations": int(len(eligible)), "patients": int(eligible.export_patient.nunique())})

    # Select one operation before examining postoperative-creatinine availability.
    eligible = eligible.sort_values(["export_patient", "patient_operation_order", "export_case"])
    selected = eligible.drop_duplicates("export_patient", keep="first").copy()
    audit["flow"].append({"stage": "one_prespecified_operation_per_patient", "operations": int(len(selected)), "patients": int(selected.export_patient.nunique())})

    map_groups = dict(tuple(map_data.groupby("export_case")))
    exposure_rows = []
    outcome_observed = []
    early_aki = []
    postop_count = []
    first_postop_hours = []
    outcome_window_hours = []
    for row in selected.itertuples():
        features = episode_features(map_groups[row.export_case], float(row.anesthesia_minutes))
        features["export_case"] = row.export_case
        exposure_rows.append(features)

        lab = cr_groups.get(row.export_case, pd.DataFrame())
        end_seconds = float(row.anesthesia_minutes) * 60
        endpoint = end_seconds + 48 * 3600
        if np.isfinite(row.next_operation_hours):
            endpoint = min(endpoint, float(row.next_operation_hours) * 3600)
        window_hours = max(0.0, (endpoint - end_seconds) / 3600)
        outcome_window_hours.append(window_hours)
        post = lab[(lab.time_seconds >= end_seconds) & (lab.time_seconds <= endpoint)] if not lab.empty and endpoint >= end_seconds else pd.DataFrame()
        postop_count.append(int(len(post)))
        if len(post):
            peak = float(post.creatinine.max())
            base = float(row.baseline_creatinine)
            outcome_observed.append(True)
            early_aki.append(bool((peak - base >= 0.3 - 1e-10) or (peak / base >= 1.5 - 1e-10)))
            first_postop_hours.append(float((post.time_seconds.min() - end_seconds) / 3600))
        else:
            outcome_observed.append(False)
            early_aki.append(pd.NA)
            first_postop_hours.append(np.nan)

    exposure = pd.DataFrame(exposure_rows)
    selected = selected.merge(exposure, on="export_case", validate="one_to_one")
    selected["outcome_observed_48h"] = outcome_observed
    selected["early_aki_48h"] = pd.array(early_aki, dtype="boolean")
    selected["postop_creatinine_count_48h"] = postop_count
    selected["first_postop_creatinine_hours"] = first_postop_hours
    selected["effective_outcome_window_hours"] = outcome_window_hours
    selected["common_surgical_support"] = selected.surgical_category.isin(COMMON_SUPPORT)

    audit["flow"].append({
        "stage": "outcome_observed_48h",
        "operations": int(selected.outcome_observed_48h.sum()),
        "patients": int(selected.loc[selected.outcome_observed_48h, "export_patient"].nunique()),
        "aki_events": int(selected.early_aki_48h.fillna(False).sum()),
    })
    audit["exposure_support"] = {
        "cases": int(len(selected)),
        "cases_any_hypotension": int(selected.any_hypotension.sum()),
        "cases_longest_episode_gte_10": int(selected.longest_episode_minutes.ge(10).sum()),
        "cases_longest_episode_gte_20": int(selected.longest_episode_minutes.ge(20).sum()),
        "cases_longest_episode_gte_60": int(selected.longest_episode_minutes.ge(60).sum()),
    }
    audit["common_support"] = {
        "operations": int(selected.common_surgical_support.sum()),
        "observed_outcomes": int(selected.loc[selected.common_surgical_support, "outcome_observed_48h"].sum()),
        "aki_events": int(selected.loc[selected.common_surgical_support, "early_aki_48h"].fillna(False).sum()),
    }

    keep = [
        "export_case", "export_patient", "age", "sex", "asa", "anesthesia_minutes",
        "surgical_category", "common_surgical_support", "hypertension_flag", "diabetes_flag",
        "baseline_creatinine", "baseline_egfr", "icu_flag", "patient_class_group",
        "map_observed_minutes", "map_total_minutes", "map_coverage", "total_low_minutes",
        "episode_count", "longest_episode_minutes", "excess5_minutes", "excess10_minutes",
        "auc_below_65_mmhg_minutes", "mean_depth_below_65_mmhg", "any_hypotension",
        "episode_durations_json", "next_operation_hours", "effective_outcome_window_hours",
        "outcome_observed_48h", "postop_creatinine_count_48h", "first_postop_creatinine_hours",
        "early_aki_48h",
    ]
    selected[keep].to_csv(out_dir / "MOVER_FEATURES_V4.csv", index=False)
    (out_dir / "MOVER_FEATURE_AUDIT_V4.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_features(args.archive, args.out_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
