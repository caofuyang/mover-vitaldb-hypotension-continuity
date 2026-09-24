"""Prespecified secondary analyses from the locked contract that the manuscript
did not report. Nothing here changes the locked primary analysis.

Contract entries addressed:
  two_hinge_piecewise_linear_5_and_10_minutes
  simple_sustained_fraction_benchmark
  episode_count_and_duration_variability_benchmark
  all_operations_patient_clustered   (checked, not applicable by design)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, "/tmp/pkg/code")
import fit_primary_model_v4 as fp  # noqa: E402

SEED = 20260923
SIMULATIONS = 2000
CONTRAST_UNIT = 0.25  # sustained fraction is reported per 25 percentage points


def scenario_from_term(result, analysis, term, unit, formula):
    """Standardized risk contrast for an arbitrary single pattern term."""
    lo = analysis.copy()
    hi = analysis.copy()
    lo[term] = 0.0
    hi[term] = unit
    import patsy
    rhs = formula.split("~", 1)[1]
    xl = patsy.dmatrix(rhs, lo, return_type="dataframe").reindex(
        columns=result.params.index, fill_value=0).to_numpy(float)
    xh = patsy.dmatrix(rhs, hi, return_type="dataframe").reindex(
        columns=result.params.index, fill_value=0).to_numpy(float)
    beta = np.asarray(result.params, float)
    se = float(result.bse[term])
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(beta, np.asarray(result.cov_params()), size=SIMULATIONS)
    pl, ph = [], []
    for start in range(0, SIMULATIONS, 100):
        block = draws[start:start + 100]
        pl.extend(expit(xl @ block.T).mean(axis=0).tolist())
        ph.extend(expit(xh @ block.T).mean(axis=0).tolist())
    pl, ph = np.asarray(pl), np.asarray(ph)
    return {
        "term": term,
        "unit": unit,
        "odds_ratio": float(np.exp(unit * beta[list(result.params.index).index(term)])),
        "odds_ratio_95ci": [
            float(np.exp(unit * (beta[list(result.params.index).index(term)] - 1.959963984540054 * se))),
            float(np.exp(unit * (beta[list(result.params.index).index(term)] + 1.959963984540054 * se))),
        ],
        "risk_at_zero": float(expit(xl @ beta).mean()),
        "risk_at_unit": float(expit(xh @ beta).mean()),
        "risk_difference": float(expit(xh @ beta).mean() - expit(xl @ beta).mean()),
        "risk_difference_95ci": [float(x) for x in np.quantile(ph - pl, [0.025, 0.975])],
    }


raw = pd.read_csv("/tmp/pkg_run/combined_features/COMBINED_FEATURES_V4.csv")
data = fp.prepare(raw)
covariates = ["age10", "egfr10", "anesthesia_hours", "total_low10", "mean_depth5"]
data = data[data[covariates].notna().all(axis=1)].copy()
observed = data[data.outcome_observed_48h & data.early_aki_48h.notna()].copy()

runs = [json.loads(v) for v in observed.episode_durations_json]
total = pd.to_numeric(observed.total_low_minutes, errors="coerce").to_numpy(float)
longest = pd.to_numeric(observed.longest_episode_minutes, errors="coerce").to_numpy(float)
count = pd.to_numeric(observed.episode_count, errors="coerce").to_numpy(float)
excess5 = pd.to_numeric(observed.excess5_minutes, errors="coerce").to_numpy(float)
excess10 = pd.to_numeric(observed.excess10_minutes, errors="coerce").to_numpy(float)

out: dict = {"cohort": {"n": int(len(observed)), "events": int(observed.early_aki_48h.sum())}}

# one-operation-per-patient design: clustering must be vacuous
out["all_operations_patient_clustered"] = {
    "cases": int(observed.analysis_case.nunique()),
    "unique_patients": int(observed.analysis_patient.nunique()),
    "note": "One operation per patient was retained by design, so a patient-clustered "
            "variance is identical to the primary robust covariance and adds nothing.",
}

# two-hinge piecewise linear (5 and 10 minutes), both in the same model
observed = observed.copy()
observed["excess10_10"] = excess10 / 10.0
two_hinge_formula = (
    "early_aki_48h ~ C(center) + age10 + C(sex_group) + C(asa_group) + egfr10 + "
    "C(surgical_category) + anesthesia_hours + total_low10 + excess5_10 + excess10_10 + mean_depth5"
)
two = fp.fit_glm(two_hinge_formula, observed)
out["two_hinge_piecewise_linear_5_and_10_minutes"] = {
    "converged": bool(two.converged),
    "n": int(two.nobs),
    "excess5_per_10_min": {
        "odds_ratio": float(np.exp(two.params["excess5_10"])),
        "odds_ratio_95ci": [float(np.exp(two.params["excess5_10"] - 1.959963984540054 * two.bse["excess5_10"])),
                            float(np.exp(two.params["excess5_10"] + 1.959963984540054 * two.bse["excess5_10"]))],
        "p": float(two.pvalues["excess5_10"]),
    },
    "excess10_per_10_min": {
        "odds_ratio": float(np.exp(two.params["excess10_10"])),
        "odds_ratio_95ci": [float(np.exp(two.params["excess10_10"] - 1.959963984540054 * two.bse["excess10_10"])),
                            float(np.exp(two.params["excess10_10"] + 1.959963984540054 * two.bse["excess10_10"]))],
        "p": float(two.pvalues["excess10_10"]),
    },
}

# simple sustained-fraction scalar benchmark (longest episode / total hypotensive time)
with np.errstate(invalid="ignore", divide="ignore"):
    fraction = np.where(total > 0, longest / np.where(total > 0, total, np.nan), 0.0)
observed["sustained_fraction"] = np.nan_to_num(fraction, nan=0.0)
sf_formula = (
    "early_aki_48h ~ C(center) + age10 + C(sex_group) + C(asa_group) + egfr10 + "
    "C(surgical_category) + anesthesia_hours + total_low10 + sustained_fraction + mean_depth5"
)
sf_result = fp.fit_glm(sf_formula, observed)
out["simple_sustained_fraction_benchmark"] = scenario_from_term(
    sf_result, observed, "sustained_fraction", CONTRAST_UNIT, sf_formula)
out["simple_sustained_fraction_benchmark"]["definition"] = (
    "longest hypotensive episode divided by total hypotensive minutes; contrast is 0 versus 0.25")

# episode count and duration variability benchmarks
observed["episode_count_10"] = count / 10.0
spread = np.array([
    float(np.std(run, ddof=0)) if len(run) > 1 else 0.0 for run in runs
])
observed["episode_spread_10"] = spread / 10.0
count_formula = (
    "early_aki_48h ~ C(center) + age10 + C(sex_group) + C(asa_group) + egfr10 + "
    "C(surgical_category) + anesthesia_hours + total_low10 + episode_count_10 + mean_depth5"
)
count_result = fp.fit_glm(count_formula, observed)
out["episode_count_benchmark"] = scenario_from_term(
    count_result, observed, "episode_count_10", 1.0, count_formula)
out["episode_count_benchmark"]["definition"] = "10 additional hypotensive episodes"

spread_formula = (
    "early_aki_48h ~ C(center) + age10 + C(sex_group) + C(asa_group) + egfr10 + "
    "C(surgical_category) + anesthesia_hours + total_low10 + episode_spread_10 + mean_depth5"
)
spread_result = fp.fit_glm(spread_formula, observed)
out["episode_variability_benchmark"] = scenario_from_term(
    spread_result, observed, "episode_spread_10", 1.0, spread_formula)
out["episode_variability_benchmark"]["definition"] = (
    "10-minute increase in the within-patient standard deviation of episode durations")

Path("/tmp/work/SECONDARY_ANALYSES.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
