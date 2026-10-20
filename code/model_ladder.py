"""Model ladder v2 (2026-10-18).

Changes from the v2.0.1 script
------------------------------
1. The ladder is presented in a single coherent order: duration alone; plus
   centre (pooled cohorts only); plus demographic, renal and surgical
   covariates; plus anaesthesia duration; plus mean depth; plus the
   episode-pattern term. The earlier script added depth at step 1 and then
   dropped it again at the anaesthesia step, so the printed ladder did not
   match a monotone sequence of adjustments.
2. The episode-pattern contrast is reported both per 10 minutes (the fitted
   coefficient) and for the prespecified 15-minute scenario (one 20-minute
   episode versus four 5-minute episodes). The 15-minute odds ratio is
   exp(1.5 * beta) with the same transformation applied to the confidence
   limits, rather than converting a per-10-minute odds ratio by raising it to
   the power 1.5 (the two agree only to first order).
3. Centre-specific pattern contrasts are computed with the same full model
   within each centre.

Post hoc / exploratory; not part of the locked analysis contract.
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


def prepare(frame: pd.DataFrame, low_column: str, depth_column: str,
            excess5_column: str, has_center: bool) -> pd.DataFrame:
    d = frame[as_bool(frame.outcome_observed_48h)].copy()
    d["age10"] = (pd.to_numeric(d.age, errors="coerce") - 60) / 10
    d["egfr10"] = (pd.to_numeric(d.baseline_egfr, errors="coerce").clip(upper=120) - 60) / 10
    d["anesthesia_hours"] = pd.to_numeric(d.anesthesia_minutes, errors="coerce") / 60
    d["total_low10"] = pd.to_numeric(d[low_column], errors="coerce") / 10
    d["mean_depth5"] = pd.to_numeric(d[depth_column], errors="coerce") / 5
    d["excess5_10"] = pd.to_numeric(d[excess5_column], errors="coerce") / 10
    d["sex_group"] = (d.sex.astype(str).str.lower().str[0]
                      .map({"m": "Male", "f": "Female"}).fillna("Unknown"))
    d["asa_group"] = pd.to_numeric(d.asa, errors="coerce").round().astype("Int64").astype(str)
    d["surgical_category"] = d.surgical_category.fillna("unmapped").astype(str)
    d["early_aki_48h"] = d.early_aki_48h.map(
        {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0}).astype(int)
    if has_center:
        d["center"] = d.center.astype(str)
    return d


def fit(formula: str, data: pd.DataFrame):
    return smf.glm(formula, data=data, family=sm.families.Binomial(),
                   missing="drop").fit(maxiter=100, cov_type="HC0")


def orci(result, term: str, scale: float = 1.0):
    if term not in result.params.index:
        return None
    b, se = float(result.params[term]) * scale, float(result.bse[term]) * scale
    return {"OR": round(float(np.exp(b)), 3),
            "lo": round(float(np.exp(b - 1.96 * se)), 3),
            "hi": round(float(np.exp(b + 1.96 * se)), 3),
            "p": round(float(result.pvalues[term]), 4)}


def ladder(d: pd.DataFrame, has_center: bool) -> dict:
    c = "C(center) + " if has_center else ""
    cov = c + BASE_COVARIATES
    steps = [
        ("Duration alone", "total_low10"),
        ("+ center", c + "total_low10"),
        ("+ demographic and surgical covariates", cov + " + total_low10"),
        ("+ anesthesia duration", cov + " + anesthesia_hours + total_low10"),
        ("+ mean depth", cov + " + anesthesia_hours + total_low10 + mean_depth5"),
        ("+ episode-pattern term",
         cov + " + anesthesia_hours + total_low10 + mean_depth5 + excess5_10"),
    ]
    out = []
    for name, rhs in steps:
        if name == "+ center" and not has_center:
            out.append({"step": name, "n": int(len(d)), "total_low10": None})
            continue
        r = fit("early_aki_48h ~ " + rhs, d)
        out.append({"step": name, "n": int(r.nobs),
                    "total_low10": orci(r, "total_low10"),
                    "mean_depth5": orci(r, "mean_depth5"),
                    "anesthesia_hours": orci(r, "anesthesia_hours"),
                    "excess5_10": orci(r, "excess5_10")})
    # Likelihood-ratio test for the pattern term in the full model.
    full_rhs = cov + " + anesthesia_hours + total_low10 + mean_depth5 + excess5_10"
    reduced_rhs = cov + " + anesthesia_hours + total_low10 + mean_depth5"
    full = fit("early_aki_48h ~ " + full_rhs, d)
    reduced = fit("early_aki_48h ~ " + reduced_rhs, d)
    lr = 2 * (full.llf - reduced.llf)
    from scipy import stats
    lr_p = float(stats.chi2.sf(lr, 1))
    pattern10 = orci(full, "excess5_10")
    pattern15 = orci(full, "excess5_10", scale=1.5)
    return {"ladder": out,
            "pattern_per10": pattern10,
            "pattern_15min_scenario": pattern15,
            "pattern_lr": {"stat": round(float(lr), 3), "df": 1, "p": round(lr_p, 3)},
            "n": int(full.nobs), "events": int(d.early_aki_48h.sum()),
            "corr_total_vs_excess5": round(
                pd.to_numeric(d.total_low10, errors="coerce").corr(
                    pd.to_numeric(d.excess5_10, errors="coerce")), 3)}


def centre_pattern(d: pd.DataFrame) -> dict:
    """Full model within each centre; pattern contrast for the 15-minute scenario."""
    out = {}
    for centre, g in d.groupby("center"):
        rhs = (BASE_COVARIATES + " + anesthesia_hours + total_low10 + mean_depth5 + excess5_10")
        r = fit("early_aki_48h ~ " + rhs, g)
        out[str(centre)] = {"n": int(r.nobs), "events": int(g.early_aki_48h.sum()),
                            "pattern_15min_scenario": orci(r, "excess5_10", scale=1.5)}
    return out


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
    d = prepare(frame, a.low_column, a.depth_column, a.excess5_column, a.center)
    result = {"note": "Post hoc / exploratory. Not part of the locked analysis contract.",
              "cohort": a.input.name, **ladder(d, a.center)}
    if a.center:
        result["pattern_by_centre"] = centre_pattern(d)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
