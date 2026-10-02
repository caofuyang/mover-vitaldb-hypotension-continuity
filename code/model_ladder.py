"""Model ladder: cumulative-duration coefficient under progressive adjustment.

Post hoc and exploratory: this script is NOT part of the locked analysis
contract in config/ANALYSIS_CONTRACT_v4.yaml.

The cumulative-duration coefficient (odds ratio per 10 minutes with mean
arterial pressure below 65 mmHg) is estimated in each cohort under a ladder of
logistic models, with and without the near-collinear episode-pattern term.

Usage
-----
# pooled waveform cohorts (has a centre column)
python code/model_ladder.py --cohort mov1min \
    --input expected_results/../../COMBINED_FEATURES_V4.csv \
    --low-column total_low_minutes --center --out model_ladder_1min.json

# resampled 5-minute cohort
python code/model_ladder.py --cohort ds5min \
    --input downsampled_5min_features.csv \
    --low-column ds_total --center --out model_ladder_5min.json

# INSPIRE charted cohort (single centre)
python code/model_ladder.py --cohort inspire \
    --input INSPIRE_FEATURES_HOLD.csv \
    --low-column total_low_minutes --out model_ladder_inspire.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_COVARIATES = "age10 + C(sex_group) + C(asa_group) + egfr10 + C(surgical_category)"


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def prepare(frame: pd.DataFrame, low_column: str, has_center: bool) -> pd.DataFrame:
    d = frame[as_bool(frame.outcome_observed_48h)].copy()
    d["age10"] = (pd.to_numeric(d.age, errors="coerce") - 60) / 10
    d["egfr10"] = (pd.to_numeric(d.baseline_egfr, errors="coerce").clip(upper=120) - 60) / 10
    d["anesthesia_hours"] = pd.to_numeric(d.anesthesia_minutes, errors="coerce") / 60
    d["total_low10"] = pd.to_numeric(d[low_column], errors="coerce") / 10
    d["sex_group"] = (d.sex.astype(str).str.lower().str[0]
                      .map({"m": "Male", "f": "Female"}).fillna("Unknown"))
    d["asa_group"] = pd.to_numeric(d.asa, errors="coerce").round().astype("Int64").astype(str)
    d["surgical_category"] = d.surgical_category.fillna("unmapped").astype(str)
    d["early_aki_48h"] = pd.to_numeric(d.early_aki_48h, errors="coerce")
    if has_center:
        d["center"] = d.center.astype(str)
    return d


def fit(formula: str, data: pd.DataFrame):
    return smf.glm(formula, data=data, family=sm.families.Binomial(),
                   missing="drop").fit(maxiter=100, cov_type="HC0")


def orci(result, term: str):
    if term not in result.params.index:
        return None
    b, se = float(result.params[term]), float(result.bse[term])
    return {"OR": round(float(np.exp(b)), 3),
            "lo": round(float(np.exp(b - 1.96 * se)), 3),
            "hi": round(float(np.exp(b + 1.96 * se)), 3),
            "p": round(float(result.pvalues[term]), 4)}


def run(d: pd.DataFrame, has_center: bool) -> dict:
    c = "C(center) + " if has_center else ""
    steps = [
        ("M0_duration", "total_low10"),
        ("M1_plus_depth", "total_low10 + mean_depth5"),
        ("M2_plus_covariates", c + BASE_COVARIATES + " + total_low10"),
        ("M3_plus_depth_covariates", c + BASE_COVARIATES + " + total_low10 + mean_depth5"),
        ("M4_plus_anesthesia", c + BASE_COVARIATES + " + anesthesia_hours + total_low10"),
        ("M5_plus_anesthesia_depth", c + BASE_COVARIATES + " + anesthesia_hours + total_low10 + mean_depth5"),
        ("M6_plus_pattern_term", c + BASE_COVARIATES + " + anesthesia_hours + total_low10 + excess5_10 + mean_depth5"),
    ]
    ladder = []
    for name, rhs in steps:
        r = fit("early_aki_48h ~ " + rhs, d)
        ladder.append({"step": name, "n": int(r.nobs),
                       "total_low10": orci(r, "total_low10"),
                       "mean_depth5": orci(r, "mean_depth5"),
                       "anesthesia_hours": orci(r, "anesthesia_hours"),
                       "excess5_10": orci(r, "excess5_10")})
    return {
        "note": "Post hoc / exploratory model ladder. Not part of the locked analysis contract.",
        "n": int(len(d)),
        "events": int(d.early_aki_48h.sum()),
        "corr_total_vs_excess5": round(
            pd.to_numeric(d.total_low10, errors="coerce").corr(
                pd.to_numeric(d.excess5_10, errors="coerce")), 3),
        "ladder": ladder,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--low-column", required=True)
    p.add_argument("--depth-column", default="mean_depth_below_65_mmhg")
    p.add_argument("--excess5-column", default="excess5_minutes")
    p.add_argument("--center", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    frame = pd.read_csv(a.input)
    frame["mean_depth5"] = pd.to_numeric(frame[a.depth_column], errors="coerce") / 5
    frame["excess5_10"] = pd.to_numeric(frame[a.excess5_column], errors="coerce") / 10
    result = run(prepare(frame, a.low_column, a.center), a.center)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
