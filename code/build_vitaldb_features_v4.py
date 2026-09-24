"""Build the harmonized VitalDB v4 feature cohort without outcome preselection.

The script uses the locally archived VitalDB case, laboratory, track-manifest,
and raw arterial-MAP files. One operation per patient is selected by a fixed
hash before postoperative-creatinine availability is examined. No association
model is fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_mover_features_v4 import COMMON_SUPPORT, episode_features
from surgical_category_audit import classify, egfr_2021, load_rules


SELECTION_SEED = "20260923"


def resolve_labs(group: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for time_seconds, values in group.groupby("dt"):
        x = pd.to_numeric(values.result, errors="coerce")
        x = x[np.isfinite(x) & x.gt(0) & x.ne(9999999)]
        if x.empty:
            continue
        spread = float(x.max() - x.min())
        if spread <= 0.1 + 1e-10:
            rows.append((float(time_seconds), float(x.median()), len(x), spread))
    return pd.DataFrame(rows, columns=["time_seconds", "creatinine", "source_rows", "spread"]).sort_values("time_seconds")


def selection_key(subject: object, case: object) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{subject}|{case}".encode()).hexdigest()


def build_features(
    cases_csv: Path,
    labs_csv: Path,
    manifest_json: Path,
    raw_map_zip: Path,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    cases = pd.read_csv(cases_csv)
    labs = pd.read_csv(labs_csv)
    manifest_data = json.loads(manifest_json.read_text(encoding="utf-8"))
    manifest = pd.DataFrame(manifest_data["results"])
    manifest["caseid"] = pd.to_numeric(manifest.caseid, errors="coerce").astype("Int64")
    manifest["priority"] = manifest.tname.map({"Solar8000/ART_MBP": 0, "EV1000/ART_MBP": 1}).fillna(99)
    manifest = manifest.sort_values(["caseid", "priority", "tid"]).drop_duplicates("caseid")

    age = pd.to_numeric(cases.age, errors="coerce")
    cases["procedure_text"] = (
        cases.department.fillna("").astype(str)
        + " | " + cases.optype.fillna("").astype(str)
        + " | " + cases.opname.fillna("").astype(str)
    )
    rules = load_rules()
    cases["surgical_category"] = [classify(x, "vitaldb", rules) for x in cases.procedure_text]
    eligible = cases[
        age.ge(18)
        & cases.ane_type.eq("General")
        & pd.to_numeric(cases.asa, errors="coerce").between(1, 4)
        & cases.anestart.notna()
        & cases.aneend.notna()
        & (cases.aneend > cases.anestart)
        & ~cases.surgical_category.isin(["exclude_transplant", "unmapped"])
        & cases.caseid.isin(manifest.caseid.dropna().astype(int))
    ].copy()

    audit = {
        "purpose": "Prespecified VitalDB feature construction; no outcome-association model",
        "selection_seed": SELECTION_SEED,
        "flow": [
            {"stage": "all_cases", "operations": int(len(cases)), "patients": int(cases.subjectid.nunique())},
            {"stage": "demographic_anesthesia_surgery_and_map_track_eligible", "operations": int(len(eligible)), "patients": int(eligible.subjectid.nunique())},
        ],
        "raw_map_manifest": {
            "complete": manifest_data.get("complete"),
            "successful_tracks": manifest_data.get("successful_tracks"),
            "failed_tracks": manifest_data.get("failed_tracks"),
        },
        "warnings": [
            "VitalDB lacks reliable cross-operation calendar ordering; one operation per patient is selected by a fixed outcome-blind hash.",
            "The postoperative window cannot be censored at a subsequent operation in VitalDB.",
        ],
    }

    selected_manifest = manifest.set_index("caseid")
    names_by_basename = {}
    exposure_rows = []
    with zipfile.ZipFile(raw_map_zip) as handle:
        names_by_basename = {Path(x).name: x for x in handle.namelist() if not x.endswith("/")}
        for row in eligible.itertuples():
            track = selected_manifest.loc[int(row.caseid)]
            filename = f"{int(row.caseid)}_{track.tid}.csv"
            member = names_by_basename.get(filename)
            if member is None:
                continue
            raw = pd.read_csv(handle.open(member))
            if raw.shape[1] < 2:
                continue
            time = pd.to_numeric(raw.iloc[:, 0], errors="coerce") - float(row.anestart)
            value = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
            case_map = pd.DataFrame({"time_seconds": time, "map_value": value})
            features = episode_features(case_map, (float(row.aneend) - float(row.anestart)) / 60)
            features["caseid"] = int(row.caseid)
            features["map_track"] = track.tname
            exposure_rows.append(features)

    exposure = pd.DataFrame(exposure_rows)
    eligible = eligible.merge(exposure, on="caseid", how="inner", validate="one_to_one")
    eligible = eligible[
        eligible.map_coverage.ge(0.70)
        & eligible.map_observed_minutes.ge(30)
    ].copy()
    audit["flow"].append({"stage": "map_quality_parent", "operations": int(len(eligible)), "patients": int(eligible.subjectid.nunique())})

    cr = labs[labs.name.astype(str).str.lower().eq("cr")].copy()
    cr_groups = {case: resolve_labs(group) for case, group in cr.groupby("caseid")}
    baseline_values = []
    for row in eligible.itertuples():
        lab = cr_groups.get(row.caseid, pd.DataFrame())
        if lab.empty:
            baseline_values.append(np.nan)
            continue
        baseline = lab[
            (lab.time_seconds >= float(row.anestart) - 30 * 86400)
            & (lab.time_seconds < float(row.anestart))
        ]
        baseline_values.append(float(baseline.iloc[-1].creatinine) if len(baseline) else np.nan)
    eligible["baseline_creatinine"] = baseline_values
    female = eligible.sex.astype(str).str.lower().str.startswith("f")
    eligible["baseline_egfr"] = [
        egfr_2021(value, float(a), is_female) if np.isfinite(value) and np.isfinite(a) else np.nan
        for value, a, is_female in zip(eligible.baseline_creatinine, pd.to_numeric(eligible.age, errors="coerce"), female)
    ]
    eligible = eligible[
        eligible.baseline_creatinine.lt(4)
        & eligible.baseline_egfr.ge(15)
    ].copy()
    audit["flow"].append({"stage": "valid_baseline_kidney_function", "operations": int(len(eligible)), "patients": int(eligible.subjectid.nunique())})

    # Fixed hash selection is independent of MAP values and postoperative AKI.
    eligible["selection_key"] = [selection_key(s, c) for s, c in zip(eligible.subjectid, eligible.caseid)]
    eligible = eligible.sort_values(["subjectid", "selection_key", "caseid"])
    selected = eligible.drop_duplicates("subjectid", keep="first").copy()
    audit["flow"].append({"stage": "one_hash_selected_operation_per_patient", "operations": int(len(selected)), "patients": int(selected.subjectid.nunique())})

    observed = []
    outcomes = []
    post_counts = []
    first_hours = []
    for row in selected.itertuples():
        lab = cr_groups.get(row.caseid, pd.DataFrame())
        start = float(row.aneend)
        stop = start + 48 * 3600
        post = lab[(lab.time_seconds >= start) & (lab.time_seconds <= stop)] if not lab.empty else pd.DataFrame()
        post_counts.append(int(len(post)))
        if len(post):
            peak = float(post.creatinine.max())
            base = float(row.baseline_creatinine)
            observed.append(True)
            outcomes.append(bool((peak - base >= 0.3 - 1e-10) or (peak / base >= 1.5 - 1e-10)))
            first_hours.append(float((post.time_seconds.min() - start) / 3600))
        else:
            observed.append(False)
            outcomes.append(pd.NA)
            first_hours.append(np.nan)

    selected["outcome_observed_48h"] = observed
    selected["early_aki_48h"] = pd.array(outcomes, dtype="boolean")
    selected["postop_creatinine_count_48h"] = post_counts
    selected["first_postop_creatinine_hours"] = first_hours
    selected["common_surgical_support"] = selected.surgical_category.isin(COMMON_SUPPORT)
    selected["anesthesia_minutes"] = (selected.aneend - selected.anestart) / 60
    selected["icu_flag"] = np.where(pd.to_numeric(selected.icu_days, errors="coerce").fillna(0).gt(0), "Y", "N")
    selected["patient_class_group"] = "VitalDB"

    audit["flow"].append({
        "stage": "outcome_observed_48h",
        "operations": int(selected.outcome_observed_48h.sum()),
        "patients": int(selected.loc[selected.outcome_observed_48h, "subjectid"].nunique()),
        "aki_events": int(selected.early_aki_48h.fillna(False).sum()),
    })
    audit["exposure_support"] = {
        "cases": int(len(selected)),
        "cases_any_hypotension": int(selected.any_hypotension.sum()),
        "cases_longest_episode_gte_10": int(selected.longest_episode_minutes.ge(10).sum()),
        "cases_longest_episode_gte_20": int(selected.longest_episode_minutes.ge(20).sum()),
        "cases_longest_episode_gte_60": int(selected.longest_episode_minutes.ge(60).sum()),
    }

    selected = selected.rename(columns={"caseid": "vital_case", "subjectid": "vital_patient"})
    keep = [
        "vital_case", "vital_patient", "age", "sex", "asa", "anesthesia_minutes",
        "surgical_category", "common_surgical_support", "preop_htn", "preop_dm", "emop", "bmi",
        "baseline_creatinine", "baseline_egfr", "icu_flag", "patient_class_group",
        "map_track", "map_observed_minutes", "map_total_minutes", "map_coverage",
        "total_low_minutes", "episode_count", "longest_episode_minutes", "excess5_minutes",
        "excess10_minutes", "auc_below_65_mmhg_minutes", "mean_depth_below_65_mmhg",
        "any_hypotension", "episode_durations_json", "outcome_observed_48h",
        "postop_creatinine_count_48h", "first_postop_creatinine_hours", "early_aki_48h",
    ]
    selected[keep].to_csv(out_dir / "VITALDB_FEATURES_V4.csv", index=False)
    (out_dir / "VITALDB_FEATURE_AUDIT_V4.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--labs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--map-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_features(args.cases, args.labs, args.manifest, args.map_zip, args.out_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
