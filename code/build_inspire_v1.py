"""Build the INSPIRE perioperative cohort for the v2.0 resolution analyses.

Post hoc and exploratory: this script is NOT part of the locked analysis
contract in config/ANALYSIS_CONTRACT_v4.yaml.

INSPIRE charts arterial pressure roughly every 5 minutes rather than
continuously, so 1-minute bins are not interpretable there. Each reading is
carried forward to the next (zero-order hold) and hypotension burden is the
time covered by readings below 65 mmHg.

Usage
-----
python code/build_inspire_v1.py \
    --inspire-dir <INSPIRE 1.4.2 directory> \
    --vitaldb-cases <VitalDB clinical_data.csv> \
    --out-dir <output directory>

The VitalDB case list is used only to remove operations that also appear in
the VitalDB open dataset, so the same procedure cannot enter two cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_mover_features_v4 import COMMON_SUPPORT
from surgical_category_audit import egfr_2021

SELECTION_SEED = "20260923"

# ICD-10-PCS body system (single character at index 1) -> harmonized v4 category
PCS_CARDIAC = {"2"}
PCS_THORACIC = {"B"}
PCS_VASCULAR = {"3", "4", "5", "6"}
PCS_GU_GYN = {"T", "U", "V"}
PCS_ABDOMINAL = {"D", "F"}
PCS_SUPERFICIAL = {"G", "H", "J", "K", "L", "M"}
PCS_OUT_OF_SUPPORT = {"0", "1", "8", "9", "C", "N", "P", "Q", "R", "S", "W", "X", "Y"}


def classify_pcs(code, cpb_present: bool) -> str:
    """Map an ICD-10-PCS code to the locked v4 surgical taxonomy."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "unmapped"
    c = str(code).strip().upper()
    if len(c) < 3:
        return "unmapped"
    if c[0] == "1":                      # obstetrics section
        return "genitourinary_gynecologic"
    body = c[1]
    if c[2] == "Y":                      # transplantation root operation
        return "exclude_transplant"
    if body in PCS_CARDIAC or cpb_present:
        return "exclude_cardiac"
    if body in PCS_GU_GYN:
        return "genitourinary_gynecologic"
    if body in PCS_THORACIC:
        return "thoracic"
    if body in PCS_VASCULAR:
        return "vascular"
    if body in PCS_ABDOMINAL:
        return "abdominal"
    if body in PCS_SUPERFICIAL:
        return "superficial_other"
    if body in PCS_OUT_OF_SUPPORT:
        return "out_of_support"
    return "unmapped"


def hold_features(times: np.ndarray, values: np.ndarray, anstart: float, anend: float):
    """Zero-order hold: each reading covers the interval up to the next reading."""
    order = np.argsort(times)
    t, v = times[order], values[order]
    keep = (t >= anstart) & (t <= anend)
    t, v = t[keep], v[keep]
    if len(t) == 0:
        return None
    ends = np.append(t[1:], anend)
    dur = ends - t
    low = v < 65.0
    total_low = float(dur[low].sum())
    episodes, i, lm = [], 0, low.astype(int)
    while i < len(lm):
        if lm[i]:
            j = i
            while j + 1 < len(lm) and lm[j + 1]:
                j += 1
            episodes.append(float(ends[j] - t[i]))
            i = j + 1
        else:
            i += 1
    depth = float((dur[low] * (65.0 - v[low])).sum())
    return {
        "map_observed_minutes": int(anend - anstart),
        "map_total_minutes": int(anend - anstart),
        "map_coverage": 1.0,
        "total_low_minutes": total_low,
        "episode_count": len(episodes),
        "longest_episode_minutes": max(episodes) if episodes else 0.0,
        "excess5_minutes": float(sum(max(x - 5, 0) for x in episodes)),
        "excess10_minutes": float(sum(max(x - 10, 0) for x in episodes)),
        "auc_below_65_mmhg_minutes": depth,
        "mean_depth_below_65_mmhg": (depth / total_low if total_low else 0.0),
        "any_hypotension": int(total_low > 0),
        "episode_durations_json": json.dumps([round(x, 1) for x in episodes]),
        "first_reading_gap": float(t[0] - anstart),
        "last_reading_gap": float(anend - t[-1]),
        "n_readings": int(len(t)),
    }


def read_filtered(csv_gz: Path, item: str, columns, chunksize: int = 5_000_000) -> pd.DataFrame:
    """Stream a large gzipped INSPIRE table and keep a single item_name."""
    parts = []
    for chunk in pd.read_csv(csv_gz, chunksize=chunksize):
        sel = chunk[chunk.item_name == item]
        if len(sel):
            parts.append(sel[columns])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def build(inspire_dir: Path, vitaldb_cases: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = pd.read_csv(inspire_dir / "operations.csv.gz")
    ops["age"] = pd.to_numeric(ops.age, errors="coerce")
    ops["asa"] = pd.to_numeric(ops.asa, errors="coerce")
    ops["cpb"] = ops.cpbon_time.notna()
    ops["surgical_category"] = [classify_pcs(c, b) for c, b in zip(ops.icd10_pcs, ops.cpb)]

    vb = pd.read_csv(vitaldb_cases)
    overlap_ids = set(pd.to_numeric(vb.caseid, errors="coerce").dropna().astype(int))
    ops["overlap_vitaldb"] = pd.to_numeric(ops.case_id, errors="coerce").isin(overlap_ids)

    flow = [{"stage": "all_operations", "operations": int(len(ops))}]
    eligible = ops[
        ops.age.ge(18)
        & ops.asa.between(1, 4)
        & ops.antype.eq("General")
        & ops.anstart_time.notna()
        & ops.anend_time.notna()
        & (ops.anend_time > ops.anstart_time)
        & ~ops.surgical_category.isin(["exclude_transplant", "exclude_cardiac", "unmapped"])
        & ~ops.overlap_vitaldb
    ].copy()
    flow.append({"stage": "eligible_demographics_anesthesia_surgery", "operations": int(len(eligible))})

    art = read_filtered(inspire_dir / "vitals.csv.gz", "art_mbp", ["op_id", "chart_time", "value"])
    art = art[art.op_id.isin(set(eligible.op_id))]
    by_op = dict(tuple(art.groupby("op_id")))
    flow.append({"stage": "operations_with_art_mbp", "operations": int(art.op_id.nunique())})

    rows = []
    for r in eligible.itertuples():
        g = by_op.get(r.op_id)
        if g is None:
            continue
        feat = hold_features(
            pd.to_numeric(g.chart_time, errors="coerce").to_numpy(float),
            pd.to_numeric(g.value, errors="coerce").to_numpy(float),
            float(r.anstart_time), float(r.anend_time),
        )
        if feat is None:
            continue
        feat["op_id"] = r.op_id
        rows.append(feat)
    eligible = eligible.merge(pd.DataFrame(rows), on="op_id", how="inner", validate="one_to_one")

    eligible["anesthesia_minutes"] = eligible.anend_time - eligible.anstart_time
    eligible = eligible[
        (eligible.first_reading_gap <= 15)
        & (eligible.last_reading_gap <= 15)
        & eligible.anesthesia_minutes.ge(30)
    ].copy()
    flow.append({"stage": "map_quality_hold", "operations": int(len(eligible))})

    cr = read_filtered(inspire_dir / "labs.csv.gz", "creatinine", ["subject_id", "chart_time", "value"])
    cr["value"] = pd.to_numeric(cr.value, errors="coerce")
    cr = cr[np.isfinite(cr.value) & cr.value.gt(0) & cr.value.ne(9999999)]
    agg = cr.groupby(["subject_id", "chart_time"]).value.agg(["median", "min", "max"]).reset_index()
    agg = agg[(agg["max"] - agg["min"]) <= 0.1 + 1e-10].sort_values(["subject_id", "chart_time"])
    labs = {s: (g.chart_time.to_numpy(float), g["median"].to_numpy(float))
            for s, g in agg.groupby("subject_id")}

    base, observed, aki, counts, first_hours = [], [], [], [], []
    for r in eligible.itertuples():
        a0, a1 = float(r.anstart_time), float(r.anend_time)
        got = labs.get(r.subject_id)
        if got is None:
            base.append(np.nan); observed.append(False); aki.append(pd.NA)
            counts.append(0); first_hours.append(np.nan)
            continue
        tm, cre = got
        pre = (tm >= a0 - 30 * 1440) & (tm < a0)
        b = float(cre[pre][-1]) if pre.any() else np.nan
        base.append(b)
        post = (tm >= a1) & (tm <= a1 + 48 * 60)
        counts.append(int(post.sum()))
        if post.any() and np.isfinite(b):
            peak = float(cre[post].max())
            observed.append(True)
            aki.append(bool((peak - b >= 0.3 - 1e-10) or (peak / b >= 1.5 - 1e-10)))
            first_hours.append(float(tm[post].min() - a1) / 60)
        else:
            observed.append(False); aki.append(pd.NA); first_hours.append(np.nan)

    eligible["baseline_creatinine"] = base
    eligible["outcome_observed_48h"] = observed
    eligible["early_aki_48h"] = pd.array(aki, dtype="boolean")
    eligible["postop_creatinine_count_48h"] = counts
    eligible["first_postop_creatinine_hours"] = first_hours

    female = eligible.sex.astype(str).str.upper().str.startswith("F")
    eligible["baseline_egfr"] = [
        egfr_2021(c, a, f) if np.isfinite(c) and np.isfinite(a) else np.nan
        for c, a, f in zip(eligible.baseline_creatinine, eligible.age, female)
    ]
    eligible = eligible[eligible.baseline_creatinine.lt(4) & eligible.baseline_egfr.ge(15)].copy()
    flow.append({"stage": "valid_baseline_kidney_function", "operations": int(len(eligible))})

    eligible["selection_key"] = [
        hashlib.sha256(f"{SELECTION_SEED}|{s}|{o}".encode()).hexdigest()
        for s, o in zip(eligible.subject_id, eligible.op_id)
    ]
    eligible = (eligible.sort_values(["subject_id", "selection_key", "op_id"])
                .drop_duplicates("subject_id", keep="first").copy())
    flow.append({"stage": "one_operation_per_patient", "operations": int(len(eligible))})
    flow.append({
        "stage": "outcome_observed_48h",
        "operations": int(eligible.outcome_observed_48h.sum()),
        "aki_events": int(eligible.loc[eligible.outcome_observed_48h, "early_aki_48h"].fillna(False).sum()),
    })

    eligible["common_surgical_support"] = eligible.surgical_category.isin(COMMON_SUPPORT)
    eligible["icu_flag"] = np.where(eligible.icuin_time.notna(), "Y", "N")
    eligible["patient_class_group"] = "INSPIRE"
    eligible = eligible.rename(columns={"op_id": "inspire_case", "subject_id": "inspire_patient"})

    keep = [
        "inspire_case", "inspire_patient", "age", "sex", "asa", "anesthesia_minutes",
        "surgical_category", "common_surgical_support", "baseline_creatinine", "baseline_egfr",
        "icu_flag", "patient_class_group", "map_observed_minutes", "map_total_minutes",
        "map_coverage", "total_low_minutes", "episode_count", "longest_episode_minutes",
        "excess5_minutes", "excess10_minutes", "auc_below_65_mmhg_minutes",
        "mean_depth_below_65_mmhg", "any_hypotension", "episode_durations_json",
        "outcome_observed_48h", "postop_creatinine_count_48h", "first_postop_creatinine_hours",
        "early_aki_48h",
    ]
    eligible[keep].to_csv(out_dir / "INSPIRE_FEATURES_HOLD.csv", index=False)

    audit = {
        "note": "Post hoc / exploratory INSPIRE cohort. Not part of the locked analysis contract.",
        "flow": flow,
        "surgical_category_counts": eligible.surgical_category.value_counts().to_dict(),
        "exposure_support": {
            "cases": int(len(eligible)),
            "any_hypotension": int(eligible.any_hypotension.sum()),
            "longest_ge_10": int(eligible.longest_episode_minutes.ge(10).sum()),
            "longest_ge_20": int(eligible.longest_episode_minutes.ge(20).sum()),
        },
    }
    (out_dir / "INSPIRE_FEATURE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspire-dir", type=Path, required=True,
                   help="INSPIRE 1.4.2 directory containing operations.csv.gz, vitals.csv.gz, labs.csv.gz")
    p.add_argument("--vitaldb-cases", type=Path, required=True,
                   help="VitalDB clinical_data.csv, used only to remove overlapping operations")
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.inspire_dir, a.vitaldb_cases, a.out_dir), indent=2, ensure_ascii=False))
