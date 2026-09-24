"""Create descriptive tables and an outcome-observation audit without refitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTINUOUS = [
    "age", "asa", "baseline_creatinine", "baseline_egfr", "anesthesia_minutes",
    "map_coverage", "total_low_minutes", "episode_count", "longest_episode_minutes",
    "excess5_minutes", "auc_below_65_mmhg_minutes", "mean_depth_below_65_mmhg",
]
CATEGORICAL = ["sex", "surgical_category", "hypertension", "diabetes", "icu_flag"]


def observed_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def continuous_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "median": float(values.median()) if len(values) else None,
        "q1": float(values.quantile(0.25)) if len(values) else None,
        "q3": float(values.quantile(0.75)) if len(values) else None,
    }


def smd_continuous(a: pd.Series, b: pd.Series) -> float | None:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / denominator) if denominator > 0 else 0.0


def smd_binary(a: pd.Series, b: pd.Series, level: str) -> float:
    pa = float(a.astype(str).eq(level).mean())
    pb = float(b.astype(str).eq(level).mean())
    denominator = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / denominator) if denominator > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(args.input_csv)
    data["outcome_observed"] = observed_flag(data.outcome_observed_48h)
    sex = data.sex.fillna("Unknown").astype(str).str.strip().str.lower()
    data["sex"] = sex.map(
        lambda item: "Female" if item.startswith("f")
        else ("Male" if item.startswith("m") else "Unknown")
    )

    table_rows = []
    groups = [("Overall", data)] + [(center, group) for center, group in data.groupby("center")]
    for group_name, group in groups:
        for variable in CONTINUOUS:
            table_rows.append({"group": group_name, "variable": variable, "level": "", **continuous_summary(group[variable])})
        for variable in CATEGORICAL:
            text = group[variable].fillna("Unknown").astype(str)
            for level, count in text.value_counts(dropna=False).items():
                table_rows.append({
                    "group": group_name, "variable": variable, "level": str(level),
                    "n": int(count), "percent": float(100 * count / len(group)),
                })
        observed = group[group.outcome_observed]
        table_rows.append({
            "group": group_name, "variable": "early_aki_48h", "level": "event_among_observed",
            "n": int(pd.to_numeric(observed.early_aki_48h, errors="coerce").sum()),
            "percent": float(100 * pd.to_numeric(observed.early_aki_48h, errors="coerce").mean()),
        })
    pd.DataFrame(table_rows).to_csv(args.out_dir / "TABLE1_DESCRIPTIVE_LONG.csv", index=False)

    # Manuscript Table 1 describes the outcome-observed analytic cohort, not the
    # parent cohort, so emit that table separately under an explicit name.
    observed_data = data[data.outcome_observed].copy()
    observed_rows = []
    observed_groups = [("Overall", observed_data)] + [
        (center, group) for center, group in observed_data.groupby("center")
    ]
    for group_name, group in observed_groups:
        for variable in CONTINUOUS:
            observed_rows.append({
                "group": group_name, "variable": variable, "level": "",
                **continuous_summary(group[variable]),
            })
        for variable in CATEGORICAL:
            text_values = group[variable].fillna("Unknown").astype(str)
            for level, count in text_values.value_counts(dropna=False).items():
                observed_rows.append({
                    "group": group_name, "variable": variable, "level": str(level),
                    "n": int(count), "percent": float(100 * count / len(group)),
                })
        observed_rows.append({
            "group": group_name, "variable": "creatinine_defined_aki_48h", "level": "event",
            "n": int(pd.to_numeric(group.early_aki_48h, errors="coerce").sum()),
            "percent": float(100 * pd.to_numeric(group.early_aki_48h, errors="coerce").mean()),
        })
    pd.DataFrame(observed_rows).to_csv(
        args.out_dir / "TABLE1_DESCRIPTIVE_OBSERVED_COHORT.csv", index=False)

    audit_rows = []
    for center_name, group in groups:
        yes = group[group.outcome_observed]
        no = group[~group.outcome_observed]
        for variable in CONTINUOUS:
            audit_rows.append({
                "group": center_name, "variable": variable, "level": "continuous",
                "observed_n": int(pd.to_numeric(yes[variable], errors="coerce").notna().sum()),
                "missing_n": int(pd.to_numeric(no[variable], errors="coerce").notna().sum()),
                "observed_mean_or_percent": float(pd.to_numeric(yes[variable], errors="coerce").mean()),
                "missing_mean_or_percent": float(pd.to_numeric(no[variable], errors="coerce").mean()),
                "standardized_mean_difference": smd_continuous(yes[variable], no[variable]),
            })
        for variable in CATEGORICAL:
            yes_text = yes[variable].fillna("Unknown").astype(str)
            no_text = no[variable].fillna("Unknown").astype(str)
            for level in sorted(set(yes_text) | set(no_text)):
                audit_rows.append({
                    "group": center_name, "variable": variable, "level": level,
                    "observed_n": int(yes_text.eq(level).sum()), "missing_n": int(no_text.eq(level).sum()),
                    "observed_mean_or_percent": float(100 * yes_text.eq(level).mean()),
                    "missing_mean_or_percent": float(100 * no_text.eq(level).mean()),
                    "standardized_mean_difference": smd_binary(yes_text, no_text, level),
                })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.out_dir / "OUTCOME_OBSERVATION_AUDIT.csv", index=False)
    result = {
        "purpose": "Descriptive and postoperative-creatinine observation audit; no new association model",
        "parent_rows": int(len(data)),
        "outcome_observed_rows": int(data.outcome_observed.sum()),
        "outcome_missing_rows": int((~data.outcome_observed).sum()),
        "maximum_absolute_smd": float(audit.standardized_mean_difference.abs().max()),
        "variables_with_absolute_smd_gte_0_2": audit.loc[
            audit.standardized_mean_difference.abs().ge(0.2),
            ["group", "variable", "level", "standardized_mean_difference"],
        ].to_dict("records"),
    }
    (args.out_dir / "DESCRIPTIVE_OBSERVATION_AUDIT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
