"""Fit prespecified exposure-definition sensitivities after the locked primary model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import chi2

from fit_primary_model_v4 import prepare


SEED = 20260923
SIMULATIONS = 2000
BASE_TERMS = [
    "C(center)", "age10", "C(sex_group)", "C(asa_group)", "egfr10",
    "C(surgical_category)", "anesthesia_hours", "sens_total10", "sens_excess10", "sens_depth5",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def fit(formula: str, data: pd.DataFrame):
    return smf.glm(formula, data=data, family=sm.families.Binomial()).fit(maxiter=100, cov_type="HC0")


def scenario(result, data: pd.DataFrame, excess_difference_units: float) -> dict:
    reference = data.copy()
    comparison = data.copy()
    for frame in [reference, comparison]:
        frame["sens_total10"] = 2.0
        frame["sens_depth5"] = 1.0
    reference["sens_excess10"] = 0.0
    comparison["sens_excess10"] = excess_difference_units
    rhs = result.model.formula.split("~", 1)[1]
    x_ref = patsy.dmatrix(rhs, reference, return_type="dataframe").reindex(columns=result.params.index, fill_value=0).to_numpy(float)
    x_cmp = patsy.dmatrix(rhs, comparison, return_type="dataframe").reindex(columns=result.params.index, fill_value=0).to_numpy(float)
    beta = np.asarray(result.params)
    ref = expit(x_ref @ beta)
    cmp = expit(x_cmp @ beta)
    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(beta, np.asarray(result.cov_params()), size=SIMULATIONS)
    draw_ref = expit(x_ref @ draws.T).mean(axis=0)
    draw_cmp = expit(x_cmp @ draws.T).mean(axis=0)
    log_or = excess_difference_units * float(result.params["sens_excess10"])
    log_se = excess_difference_units * float(result.bse["sens_excess10"])
    return {
        "odds_ratio": float(np.exp(log_or)),
        "odds_ratio_95ci": [float(np.exp(log_or - 1.959963984540054 * log_se)), float(np.exp(log_or + 1.959963984540054 * log_se))],
        "standardized_reference_risk": float(ref.mean()),
        "standardized_comparison_risk": float(cmp.mean()),
        "standardized_risk_difference": float((cmp - ref).mean()),
        "standardized_risk_difference_95ci": [float(x) for x in np.quantile(draw_cmp - draw_ref, [0.025, 0.975])],
        "standardized_risk_ratio": float(cmp.mean() / ref.mean()),
        "standardized_risk_ratio_95ci": [float(x) for x in np.quantile(draw_cmp / draw_ref, [0.025, 0.975])],
    }


def run_variant(
    base: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    name: str,
    prefix: str,
    excess_field: str,
    excess_difference_units: float,
    center: str | None = None,
    minimum_coverage: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    columns = [
        "analysis_case", f"{prefix}_total_low_minutes", f"{prefix}_{excess_field}",
        f"{prefix}_mean_depth_below_threshold_mmhg", f"{prefix}_map_coverage",
    ]
    data = base.merge(sensitivity[columns], on="analysis_case", validate="one_to_one")
    if center:
        data = data[data.center.eq(center)].copy()
    if minimum_coverage is not None:
        data = data[pd.to_numeric(data[f"{prefix}_map_coverage"], errors="coerce").ge(minimum_coverage)].copy()
    data["sens_total10"] = pd.to_numeric(data[f"{prefix}_total_low_minutes"], errors="coerce") / 10
    data["sens_excess10"] = pd.to_numeric(data[f"{prefix}_{excess_field}"], errors="coerce") / 10
    data["sens_depth5"] = pd.to_numeric(data[f"{prefix}_mean_depth_below_threshold_mmhg"], errors="coerce") / 5
    required = ["age10", "egfr10", "anesthesia_hours", "sens_total10", "sens_excess10", "sens_depth5"]
    data = data[data[required].notna().all(axis=1) & data.outcome_observed_48h & data.early_aki_48h.notna()].copy()
    terms = [term for term in BASE_TERMS if not (center and term == "C(center)")]
    full_formula = "early_aki_48h ~ " + " + ".join(terms)
    reduced_formula = "early_aki_48h ~ " + " + ".join(term for term in terms if term != "sens_excess10")
    full = fit(full_formula, data)
    reduced = fit(reduced_formula, data)
    lr = max(0.0, 2 * (full.llf - reduced.llf))
    convergence = {
        "converged": bool(full.converged),
        "maximum_absolute_coefficient": float(np.abs(full.params).max()),
        "separation_trigger": bool((not full.converged) or np.abs(full.params).max() > 10),
        "parameters": int(len(full.params)),
    }
    result = {
        "name": name,
        "prefix": prefix,
        "population": center or "pooled",
        "minimum_map_coverage": minimum_coverage,
        "rows": int(len(data)),
        "aki_events": int(data.early_aki_48h.sum()),
        "formula": full_formula,
        "excess_field": excess_field,
        "contrast_excess_difference_units_of_10_minutes": excess_difference_units,
        "likelihood_ratio_test": {"statistic": float(lr), "df": 1, "p_value": float(chi2.sf(lr, 1))},
        "model": convergence,
        "scenario": scenario(full, data, excess_difference_units),
    }
    coefficients = pd.DataFrame({
        "model": name,
        "term": full.params.index,
        "coefficient": full.params.values,
        "standard_error": full.bse.values,
        "odds_ratio_per_model_unit": np.exp(full.params.values),
        "ci_low": np.exp(full.params.values - 1.959963984540054 * full.bse.values),
        "ci_high": np.exp(full.params.values + 1.959963984540054 * full.bse.values),
        "p_value": full.pvalues.values,
    })
    return result, coefficients


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--lock-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    base = prepare(pd.read_csv(args.combined))
    sensitivity = pd.read_csv(args.sensitivity)
    specifications = [
        dict(name="threshold_60", prefix="t60_60s", excess_field="excess5_minutes", excess_difference_units=1.5),
        dict(name="threshold_70", prefix="t70_60s", excess_field="excess5_minutes", excess_difference_units=1.5),
        dict(name="bridge_one_missing_minute", prefix="t65_60s_bridge1missing", excess_field="excess5_minutes", excess_difference_units=1.5),
        dict(name="hinge_10_minutes", prefix="t65_60s", excess_field="excess10_minutes", excess_difference_units=1.0),
        dict(name="vitaldb_10_second", prefix="t65_10s", excess_field="excess5_minutes", excess_difference_units=1.5, center="VitalDB", minimum_coverage=0.70),
    ]
    results = []
    coefficients = []
    for specification in specifications:
        result, coefficient = run_variant(base, sensitivity, **specification)
        results.append(result)
        coefficients.append(coefficient)
    report = {
        "purpose": "Prespecified sensitivity association models after the locked primary analysis",
        "causal_claim": False,
        "input_sha256": {
            "combined_features": sha256(args.combined),
            "sensitivity_features": sha256(args.sensitivity),
            "analysis_lock_manifest": sha256(args.lock_manifest),
        },
        "results": results,
    }
    pd.concat(coefficients, ignore_index=True).to_csv(args.out_dir / "SENSITIVITY_MODEL_COEFFICIENTS_V4_1.csv", index=False)
    (args.out_dir / "SENSITIVITY_MODEL_REPORT_V4_1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
