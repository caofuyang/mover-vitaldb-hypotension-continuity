"""Standardized 48-hour AKI risk with 95% intervals for every reported scenario.

Uses the same estimator, random seed, and coefficient-simulation size as the
locked primary analysis, so the risk-difference intervals reproduce the archived
values exactly while each of the two component risks also receives an interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_primary_model_v4 as fp  # noqa: E402

SEED = 20260923
SIMULATIONS = 2000


def scenario(result, analysis: pd.DataFrame, formula: str) -> dict:
    reference = analysis.copy()
    comparison = analysis.copy()
    for frame in (reference, comparison):
        frame["total_low10"] = 2.0
        frame["mean_depth5"] = 1.0
    reference["excess5_10"] = 0.0
    comparison["excess5_10"] = 1.5
    rhs = formula.split("~", 1)[1]
    x_ref = patsy.dmatrix(rhs, reference, return_type="dataframe").reindex(
        columns=result.params.index, fill_value=0).to_numpy(float)
    x_cmp = patsy.dmatrix(rhs, comparison, return_type="dataframe").reindex(
        columns=result.params.index, fill_value=0).to_numpy(float)
    beta = np.asarray(result.params, float)
    index = list(result.params.index).index("excess5_10")
    standard_error = float(result.bse["excess5_10"])
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(beta, np.asarray(result.cov_params()), size=SIMULATIONS)
    reference_draws, comparison_draws = [], []
    for start in range(0, SIMULATIONS, 100):
        block = draws[start:start + 100]
        reference_draws.extend(expit(x_ref @ block.T).mean(axis=0).tolist())
        comparison_draws.extend(expit(x_cmp @ block.T).mean(axis=0).tolist())
    reference_draws = np.asarray(reference_draws)
    comparison_draws = np.asarray(comparison_draws)
    return {
        "n": int(len(analysis)),
        "events": int(analysis.early_aki_48h.sum()),
        "odds_ratio": float(np.exp(1.5 * beta[index])),
        "odds_ratio_95ci": [
            float(np.exp(1.5 * (beta[index] - 1.959963984540054 * standard_error))),
            float(np.exp(1.5 * (beta[index] + 1.959963984540054 * standard_error))),
        ],
        "four_by_five_risk": float(expit(x_ref @ beta).mean()),
        "four_by_five_risk_95ci": [float(x) for x in np.quantile(reference_draws, [0.025, 0.975])],
        "one_by_twenty_risk": float(expit(x_cmp @ beta).mean()),
        "one_by_twenty_risk_95ci": [float(x) for x in np.quantile(comparison_draws, [0.025, 0.975])],
        "risk_difference": float(expit(x_cmp @ beta).mean() - expit(x_ref @ beta).mean()),
        "risk_difference_95ci": [float(x) for x in np.quantile(comparison_draws - reference_draws, [0.025, 0.975])],
        "risk_ratio": float(expit(x_cmp @ beta).mean() / expit(x_ref @ beta).mean()),
        "risk_ratio_95ci": [float(x) for x in np.quantile(comparison_draws / reference_draws, [0.025, 0.975])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = fp.prepare(pd.read_csv(args.input_csv))
    covariates = ["age10", "egfr10", "anesthesia_hours", "total_low10", "excess5_10", "mean_depth5"]
    data = data[data[covariates].notna().all(axis=1)].copy()
    observed = data[data.outcome_observed_48h & data.early_aki_48h.notna()].copy()

    primary_formula = "early_aki_48h ~ " + " + ".join(fp.PRIMARY_TERMS)
    no_center_formula = "early_aki_48h ~ " + " + ".join(
        term for term in fp.PRIMARY_TERMS if term != "C(center)")
    weights, _ = fp.observation_weights(data)
    observed_weights = weights[
        data.outcome_observed_48h.to_numpy() & data.early_aki_48h.notna().to_numpy()]

    report = {
        "comparison": "one 20-minute episode versus four 5-minute episodes, "
                      "total hypotension duration fixed at 20 minutes and mean depth at 5 mmHg",
        "pooled_primary": scenario(fp.fit_glm(primary_formula, observed), observed, primary_formula),
        "pooled_ipow": scenario(fp.fit_glm(primary_formula, observed, weights=observed_weights),
                                observed, primary_formula),
    }
    for center, group in observed.groupby("center"):
        report[f"center_{center}"] = scenario(
            fp.fit_glm(no_center_formula, group), group, no_center_formula)
    common = observed[observed.common_surgical_support.astype(bool)].copy()
    report["pooled_common_surgical_support"] = scenario(
        fp.fit_glm(primary_formula, common), common, primary_formula)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    answered = {k: v for k, v in report.items() if k != "comparison"}
    print(json.dumps({"status": "completed", "scenarios": len(answered),
                      "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
