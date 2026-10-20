"""Apply the locked v4 surgical taxonomy and audit common-support feasibility.

This is a design-stage audit only. It does not fit an exposure-outcome model.
Rules are applied in ascending priority and must be frozen before association
modeling. Any unmapped procedures require blinded adjudication using procedure
text only, without AKI status or MAP exposure.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
RULE_FILE = HERE / "SURGICAL_CATEGORY_RULES_v4.csv"


def load_rules() -> pd.DataFrame:
    rules = pd.read_csv(RULE_FILE).sort_values("priority")
    rules["compiled"] = [re.compile(x, re.I) for x in rules.pattern]
    return rules


def classify(text: object, scope: str, rules: pd.DataFrame) -> str:
    value = "" if pd.isna(text) else str(text)
    for row in rules.itertuples():
        if row.scope not in {"both", scope}:
            continue
        if row.compiled.search(value):
            return row.category
    return "unmapped"


def egfr_2021(creatinine: float, age: float, female: bool) -> float:
    k = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    return 142 * min(creatinine / k, 1) ** alpha * max(creatinine / k, 1) ** -1.2 * 0.9938 ** age * (1.012 if female else 1)


def vitaldb(rules: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_csv(ROOT / "corrected_submission_work_v2/code/feasibility_cohort.csv")
    map_features = pd.read_csv(ROOT / "corrected_submission_work_v2/code/map_features.csv")
    data = cohort.merge(map_features, on="caseid")
    age = pd.to_numeric(data.age.replace(">89", 90), errors="coerce")
    female = data.sex.astype(str).str.upper().str.startswith("F")
    data["baseline_egfr"] = [
        egfr_2021(cr, a, f) if np.isfinite(cr) and np.isfinite(a) else np.nan
        for cr, a, f in zip(data.baseline_cr, age, female)
    ]
    eligible = (
        age.ge(18)
        & data.ane_type.eq("General")
        & data.asa.between(1, 4)
        & data.map_raw_coverage.ge(0.70)
        & data.map_valid_minutes.ge(30)
        & data.baseline_cr.lt(4)
        & data.baseline_egfr.ge(15)
        & data.postop_cr_count_48h.gt(0)
    )
    data = data.loc[eligible].copy()
    data["procedure_text"] = (
        data.department.fillna("") + " | " + data.optype.fillna("") + " | " + data.opname.fillna("")
    )
    data["category"] = [classify(x, "vitaldb", rules) for x in data.procedure_text]
    delta = data.max_postop_cr_48h - data.baseline_cr
    ratio = data.max_postop_cr_48h / data.baseline_cr
    data["early_aki"] = (delta >= 0.3 - 1e-10) | (ratio >= 1.5 - 1e-10)
    return data[["caseid", "subjectid", "procedure_text", "category", "early_aki"]]


def mover(rules: pd.DataFrame) -> pd.DataFrame:
    archive = ROOT / "upload/14ab08c2-7368-43f8-8381-2623d99d9b98.zip"
    with zipfile.ZipFile(archive) as handle:
        cohort = pd.read_csv(handle.open("cohort.csv"))
        labs = pd.read_csv(handle.open("creatinine.csv"))
    features = pd.read_csv(ROOT / "harmonized_analysis/mover_features.csv").rename(columns={"case": "export_case"})
    data = cohort.merge(features[["export_case", "occupancy", "valid_minutes"]], on="export_case")
    data = data[
        data.occupancy.ge(0.70)
        & data.valid_minutes.ge(30)
    ].copy()
    data["category"] = [classify(x, "mover", rules) for x in data.procedure_name]
    lab_groups = dict(tuple(labs.groupby("export_case")))
    baseline = []
    early_aki = []
    assessable = []
    for row in data.itertuples():
        lab = lab_groups.get(row.export_case, pd.DataFrame())
        if lab.empty:
            baseline.append(np.nan)
            early_aki.append(False)
            assessable.append(False)
            continue
        lab = lab[np.isfinite(lab.lab_value) & lab.lab_value.gt(0) & lab.lab_value.ne(9999999)].copy()
        # Resolve same-time duplicates without using outcome direction.
        resolved = []
        for time_seconds, group in lab.groupby("time_seconds"):
            if group.lab_value.max() - group.lab_value.min() <= 0.1 + 1e-10:
                resolved.append((time_seconds, float(group.lab_value.median())))
        lab = pd.DataFrame(resolved, columns=["time_seconds", "lab_value"]).sort_values("time_seconds")
        end = row.anesthesia_minutes * 60
        pre = lab[(lab.time_seconds < 0) & (lab.time_seconds >= -30 * 86400)]
        post = lab[(lab.time_seconds >= end) & (lab.time_seconds <= end + 48 * 3600)]
        if pre.empty or post.empty:
            baseline.append(np.nan)
            early_aki.append(False)
            assessable.append(False)
            continue
        base = float(pre.iloc[-1].lab_value)
        peak = float(post.lab_value.max())
        baseline.append(base)
        early_aki.append((peak - base >= 0.3 - 1e-10) or (peak / base >= 1.5 - 1e-10))
        assessable.append(True)
    data["baseline_creatinine"] = baseline
    data["early_aki"] = early_aki
    data["outcome_assessable"] = assessable
    female = data.sex.astype(str).str.lower().str.startswith("f")
    data["baseline_egfr"] = [
        egfr_2021(cr, a, f) if np.isfinite(cr) and np.isfinite(a) else np.nan
        for cr, a, f in zip(data.baseline_creatinine, data.age, female)
    ]
    data = data[
        data.outcome_assessable
        & data.baseline_creatinine.lt(4)
        & data.baseline_egfr.ge(15)
    ].copy()
    data = data.sort_values(["export_patient", "start_day_relative_to_first_selected", "export_case"])
    data = data.drop_duplicates("export_patient", keep="first")
    return data[["export_case", "export_patient", "procedure_name", "category", "early_aki"]]


def summary(data: pd.DataFrame, dataset: str) -> list[dict]:
    rows = []
    for category, group in data.groupby("category", dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "category": category,
                "operations": int(len(group)),
                "patients": int(group.iloc[:, 1].nunique()),
                "early_aki_events": int(group.early_aki.sum()),
                "early_aki_percent": round(100 * group.early_aki.mean(), 2),
            }
        )
    return rows


if __name__ == "__main__":
    locked = load_rules()
    vital = vitaldb(locked)
    mover_data = mover(locked)
    combined = pd.DataFrame(summary(vital, "VitalDB") + summary(mover_data, "MOVER"))
    combined.to_csv(HERE / "SURGICAL_CATEGORY_AUDIT.csv", index=False)
    common = {"abdominal", "thoracic", "vascular", "genitourinary_gynecologic", "superficial_other"}
    result = {
        "purpose": "Design-stage surgical-category and common-support audit; no exposure-outcome model",
        "rules_file": RULE_FILE.name,
        "primary_common_support_categories": sorted(common),
        "summary": combined.to_dict("records"),
        "common_support_totals": {},
        "unmapped_examples": {},
        "interpretation_warning": "MOVER counts are provisional because the current export preselected postoperative creatinine availability.",
    }
    for name, data, text_col in [
        ("VitalDB", vital, "procedure_text"),
        ("MOVER", mover_data, "procedure_name"),
    ]:
        selected = data[data.category.isin(common)]
        result["common_support_totals"][name] = {
            "operations": int(len(selected)),
            "patients": int(selected.iloc[:, 1].nunique()),
            "early_aki_events": int(selected.early_aki.sum()),
        }
        result["unmapped_examples"][name] = (
            data.loc[data.category.eq("unmapped"), text_col].value_counts().head(50).index.tolist()
        )
    (HERE / "SURGICAL_CATEGORY_AUDIT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    vital[["caseid", "category"]].to_csv(HERE / "VITALDB_SURGICAL_CATEGORY_ASSIGNMENTS.csv", index=False)
    mover_data[["export_case", "category"]].to_csv(HERE / "MOVER_SURGICAL_CATEGORY_ASSIGNMENTS_PROVISIONAL.csv", index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
