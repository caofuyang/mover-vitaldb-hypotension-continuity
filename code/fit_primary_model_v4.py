"""Fit the prespecified 1-df multicenter duration-pattern model.

This program is intended for use only after the v4 contract is finally hashed.
It fits the pooled complete-outcome model, an IPOW sensitivity model, identical
site-specific models, and the prespecified center interaction. It does not
perform automated variable, threshold, or knot selection.
"""

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
import yaml


SEED = 20260923
SIMULATIONS = 2000
BOOTSTRAP_REPETITIONS = 500
PRIMARY_TERMS = [
    "C(center)",
    "age10",
    "C(sex_group)",
    "C(asa_group)",
    "egfr10",
    "C(surgical_category)",
    "anesthesia_hours",
    "total_low10",
    "excess5_10",
    "mean_depth5",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_analysis_lock(input_csv: Path, contract_path: Path, manifest_path: Path) -> dict:
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if contract.get("status") != "locked_ready_for_final_v4_1_model_after_documented_technical_amendment":
        raise RuntimeError("Analysis contract is not locked")
    if not manifest.get("all_hard_gates_passed"):
        raise RuntimeError("Pre-model hard gates did not all pass")
    expected = manifest.get("sha256", {})
    observed = {
        "analysis_contract": sha256(contract_path),
        "combined_features": sha256(input_csv),
        "primary_model_script": sha256(Path(__file__)),
    }
    mismatched = [name for name, value in observed.items() if expected.get(name) != value]
    if mismatched:
        raise RuntimeError("Analysis lock hash mismatch: " + ", ".join(mismatched))
    return {
        "contract_sha256": observed["analysis_contract"],
        "combined_features_sha256": observed["combined_features"],
        "manifest_sha256": sha256(manifest_path),
    }


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    required = {
        "center", "age", "sex", "asa", "baseline_egfr", "surgical_category",
        "anesthesia_minutes", "total_low_minutes", "excess5_minutes",
        "mean_depth_below_65_mmhg", "outcome_observed_48h", "early_aki_48h",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    data["center"] = data.center.astype(str)
    data["sex_group"] = data.sex.fillna("Unknown").astype(str).str.lower().map(
        lambda x: "Female" if x.startswith("f") else ("Male" if x.startswith("m") else "Unknown")
    )
    data["asa_group"] = pd.to_numeric(data.asa, errors="coerce").round().astype("Int64").astype(str)
    data["surgical_category"] = data.surgical_category.fillna("unmapped").astype(str)
    data["age10"] = (pd.to_numeric(data.age, errors="coerce") - 60) / 10
    data["egfr10"] = (pd.to_numeric(data.baseline_egfr, errors="coerce").clip(upper=120) - 60) / 10
    data["anesthesia_hours"] = pd.to_numeric(data.anesthesia_minutes, errors="coerce") / 60
    data["total_low10"] = pd.to_numeric(data.total_low_minutes, errors="coerce") / 10
    data["excess5_10"] = pd.to_numeric(data.excess5_minutes, errors="coerce") / 10
    data["mean_depth5"] = pd.to_numeric(data.mean_depth_below_65_mmhg, errors="coerce") / 5
    data["outcome_observed_48h"] = data.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])
    outcome_numeric = pd.to_numeric(data.early_aki_48h, errors="coerce")
    outcome_text = data.early_aki_48h.astype(str).str.lower()
    outcome_numeric = outcome_numeric.mask(outcome_numeric.isna() & outcome_text.isin(["true", "yes", "y"]), 1)
    outcome_numeric = outcome_numeric.mask(outcome_numeric.isna() & outcome_text.isin(["false", "no", "n"]), 0)
    data["early_aki_48h"] = outcome_numeric
    return data


def fit_glm(formula: str, data: pd.DataFrame, weights: np.ndarray | None = None):
    model = smf.glm(
        formula,
        data=data,
        family=sm.families.Binomial(),
        freq_weights=weights,
        missing="drop",
    )
    result = model.fit(maxiter=100, cov_type="HC0")
    return result


def convergence_summary(result) -> dict:
    coefficients = np.asarray(result.params, float)
    return {
        "converged": bool(result.converged),
        "maximum_absolute_coefficient": float(np.max(np.abs(coefficients))),
        "separation_trigger": bool((not result.converged) or np.max(np.abs(coefficients)) > 10),
        "nobs": int(result.nobs),
        "parameters": int(len(coefficients)),
    }


def scenario_summary(result, analysis: pd.DataFrame) -> dict:
    reference = analysis.copy()
    comparison = analysis.copy()
    for frame in [reference, comparison]:
        frame["total_low10"] = 2.0
        frame["mean_depth5"] = 1.0
    reference["excess5_10"] = 0.0
    comparison["excess5_10"] = 1.5
    rhs = result.model.formula.split("~", 1)[1]
    x_ref_frame = patsy.dmatrix(rhs, reference, return_type="dataframe")
    x_cmp_frame = patsy.dmatrix(rhs, comparison, return_type="dataframe")
    x_ref = x_ref_frame.reindex(columns=result.params.index, fill_value=0).to_numpy(float)
    x_cmp = x_cmp_frame.reindex(columns=result.params.index, fill_value=0).to_numpy(float)
    beta = np.asarray(result.params)
    p_ref = expit(x_ref @ beta)
    p_cmp = expit(x_cmp @ beta)
    rd = float(np.mean(p_cmp - p_ref))
    risk_ref = float(np.mean(p_ref))
    risk_cmp = float(np.mean(p_cmp))

    rng = np.random.default_rng(SEED)
    draws = rng.multivariate_normal(beta, np.asarray(result.cov_params()), size=SIMULATIONS)
    # Chunk to avoid a large cases x simulations temporary matrix.
    draw_rd = []
    draw_rr = []
    for start in range(0, SIMULATIONS, 100):
        block = draws[start : start + 100]
        ref = expit(x_ref @ block.T).mean(axis=0)
        cmp = expit(x_cmp @ block.T).mean(axis=0)
        draw_rd.extend((cmp - ref).tolist())
        draw_rr.extend((cmp / ref).tolist())
    term = "excess5_10"
    scenario_or = float(np.exp(1.5 * result.params[term]))
    scenario_log_se = float(1.5 * result.bse[term])
    return {
        "comparison": "one_20_minute_episode_vs_four_5_minute_episodes",
        "fixed_total_low_minutes": 20,
        "fixed_mean_depth_below_65_mmhg": 5,
        "odds_ratio": scenario_or,
        "odds_ratio_95ci": [
            float(np.exp(np.log(scenario_or) - 1.959963984540054 * scenario_log_se)),
            float(np.exp(np.log(scenario_or) + 1.959963984540054 * scenario_log_se)),
        ],
        "standardized_reference_risk": risk_ref,
        "standardized_comparison_risk": risk_cmp,
        "standardized_risk_difference": rd,
        "standardized_risk_difference_95ci": [float(x) for x in np.quantile(draw_rd, [0.025, 0.975])],
        "standardized_risk_ratio": risk_cmp / risk_ref,
        "standardized_risk_ratio_95ci": [float(x) for x in np.quantile(draw_rr, [0.025, 0.975])],
        "uncertainty_method": "multivariate-normal coefficient simulation using robust covariance",
    }


def observation_weights(data: pd.DataFrame) -> tuple[np.ndarray | None, dict]:
    working = data.copy()
    working["observed_numeric"] = working.outcome_observed_48h.astype(int)
    terms = [
        "C(center)", "age10", "C(sex_group)", "C(asa_group)", "egfr10",
        "C(surgical_category)", "anesthesia_hours", "total_low10", "mean_depth5",
    ]
    if "icu_flag" in working.columns:
        working["icu_group"] = working.icu_flag.fillna("Unknown").astype(str)
        terms.append("C(icu_group)")
    formula = "observed_numeric ~ " + " + ".join(terms)
    try:
        result = fit_glm(formula, working)
        probability = np.clip(np.asarray(result.predict(working)), 0.05, 0.95)
        prevalence_by_center = working.groupby("center").observed_numeric.transform("mean").to_numpy()
        raw = prevalence_by_center / probability
        observed_raw = raw[working.outcome_observed_48h.to_numpy()]
        lo, hi = np.quantile(observed_raw, [0.01, 0.99])
        truncated = np.clip(raw, lo, hi)
        diagnostics = {
            "status": "available",
            "formula": formula,
            "probability_clip": [0.05, 0.95],
            "weight_truncation_quantiles": [0.01, 0.99],
            "observed_weight_min_median_max_before_truncation": [
                float(np.min(observed_raw)), float(np.median(observed_raw)), float(np.max(observed_raw))
            ],
            "observed_weight_min_median_max_after_truncation": [
                float(np.min(truncated[working.outcome_observed_48h])),
                float(np.median(truncated[working.outcome_observed_48h])),
                float(np.max(truncated[working.outcome_observed_48h])),
            ],
            "extreme_weight_warning": bool(np.max(truncated[working.outcome_observed_48h]) > 20),
            "model": convergence_summary(result),
        }
        return truncated, diagnostics
    except Exception as exc:
        return None, {"status": "unavailable", "error": str(exc)}


def coefficient_frame(result, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": label,
            "term": result.params.index,
            "coefficient": result.params.values,
            "standard_error": result.bse.values,
            "odds_ratio": np.exp(result.params.values),
            "ci_low": np.exp(result.params.values - 1.959963984540054 * result.bse.values),
            "ci_high": np.exp(result.params.values + 1.959963984540054 * result.bse.values),
            "p_value": result.pvalues.values,
        }
    )


def bootstrap_primary(result, analysis: pd.DataFrame) -> dict:
    x = np.asarray(result.model.exog, float)
    y = np.asarray(result.model.endog, float)
    names = list(result.model.exog_names)
    excess_index = names.index("excess5_10")
    center = analysis.center.to_numpy()
    strata = [np.flatnonzero(center == value) for value in sorted(np.unique(center))]
    rng = np.random.default_rng(SEED)
    estimates = []
    failures = 0
    start_beta = np.asarray(result.params, float)
    for _ in range(BOOTSTRAP_REPETITIONS):
        sample = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        xb = x[sample]
        yb = y[sample]
        beta = start_beta.copy()
        success = False
        for _ in range(75):
            probability = expit(xb @ beta)
            weight = np.clip(probability * (1 - probability), 1e-8, None)
            hessian = xb.T @ (weight[:, None] * xb)
            score = xb.T @ (yb - probability)
            try:
                step = np.linalg.solve(hessian, score)
            except np.linalg.LinAlgError:
                break
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                success = True
                break
        if success and np.all(np.isfinite(beta)) and np.max(np.abs(beta)) <= 10:
            estimates.append(float(np.exp(1.5 * beta[excess_index])))
        else:
            failures += 1
    return {
        "method": "nonparametric patient bootstrap stratified by center",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "valid_fits": len(estimates),
        "failed_fits": failures,
        "instability_flag": len(estimates) < 450,
        "scenario_or_median": float(np.median(estimates)) if estimates else None,
        "scenario_or_percentile_95ci": [float(x) for x in np.quantile(estimates, [0.025, 0.975])] if estimates else None,
        "proportion_scenario_or_above_one": float(np.mean(np.asarray(estimates) > 1)) if estimates else None,
    }


def run_analysis(input_csv: Path, out_dir: Path, lock_audit: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)
    data = prepare(pd.read_csv(input_csv))
    covariate_cols = ["age10", "egfr10", "anesthesia_hours", "total_low10", "excess5_10", "mean_depth5"]
    complete_covariates = data[covariate_cols].notna().all(axis=1)
    data = data[complete_covariates].copy()
    observed = data[data.outcome_observed_48h & data.early_aki_48h.notna()].copy()
    if observed.early_aki_48h.sum() < 20:
        raise RuntimeError("Fewer than 20 observed AKI events; formal model not allowed")

    primary_formula = "early_aki_48h ~ " + " + ".join(PRIMARY_TERMS)
    comparator_terms = [x for x in PRIMARY_TERMS if x != "excess5_10"]
    comparator_formula = "early_aki_48h ~ " + " + ".join(comparator_terms)
    primary = fit_glm(primary_formula, observed)
    if convergence_summary(primary)["separation_trigger"]:
        raise RuntimeError("Primary model triggered the prespecified penalized/bias-reduced fallback")
    comparator = fit_glm(comparator_formula, observed)
    lr = max(0.0, 2 * (primary.llf - comparator.llf))
    lr_p = float(chi2.sf(lr, 1))

    weights, weight_audit = observation_weights(data)
    weighted = None
    if weights is not None:
        observed_weights = weights[data.outcome_observed_48h.to_numpy() & data.early_aki_48h.notna().to_numpy()]
        weighted = fit_glm(primary_formula, observed, weights=observed_weights)

    interaction_formula = primary_formula + " + excess5_10:C(center)"
    interaction = fit_glm(interaction_formula, observed)

    coefficients = [coefficient_frame(primary, "pooled_primary")]
    models = {
        "pooled_primary": convergence_summary(primary),
        "pooled_ipow": None,
        "center_specific": {},
        "center_interaction": convergence_summary(interaction),
    }
    scenarios = {"pooled_primary": scenario_summary(primary, observed)}
    bootstrap = bootstrap_primary(primary, observed)
    if weighted is not None:
        coefficients.append(coefficient_frame(weighted, "pooled_ipow"))
        models["pooled_ipow"] = convergence_summary(weighted)
        scenarios["pooled_ipow"] = scenario_summary(weighted, observed)

    for center, group in observed.groupby("center"):
        if group.early_aki_48h.sum() < 20:
            models["center_specific"][center] = {"status": "not_fitted", "reason": "fewer_than_20_events"}
            continue
        terms = [x for x in PRIMARY_TERMS if x != "C(center)"]
        try:
            result = fit_glm("early_aki_48h ~ " + " + ".join(terms), group)
            models["center_specific"][center] = convergence_summary(result)
            coefficients.append(coefficient_frame(result, f"center_{center}"))
            scenarios[f"center_{center}"] = scenario_summary(result, group)
        except Exception as exc:
            models["center_specific"][center] = {"status": "failed", "error": str(exc)}

    common_support = observed[observed.common_surgical_support.astype(bool)].copy()
    common_model = fit_glm(primary_formula, common_support)
    models["pooled_common_surgical_support"] = convergence_summary(common_model)
    coefficients.append(coefficient_frame(common_model, "pooled_common_surgical_support"))
    scenarios["pooled_common_surgical_support"] = scenario_summary(common_model, common_support)

    coefficients.append(coefficient_frame(interaction, "center_interaction"))
    pd.concat(coefficients, ignore_index=True).to_csv(out_dir / "PRIMARY_MODEL_COEFFICIENTS.csv", index=False)
    report = {
        "status": "completed",
        "analysis_contract": "v4 prespecified one-degree-of-freedom duration-pattern model",
        "analysis_lock": lock_audit or {"status": "synthetic_test_bypass"},
        "causal_claim": False,
        "input_rows_after_covariate_completeness": int(len(data)),
        "observed_outcome_rows": int(len(observed)),
        "aki_events": int(observed.early_aki_48h.sum()),
        "center_counts": observed.groupby("center").early_aki_48h.agg(["size", "sum"]).reset_index().to_dict("records"),
        "primary_formula": primary_formula,
        "comparator_formula": comparator_formula,
        "likelihood_ratio_test_excess5": {"statistic": lr, "df": 1, "p_value": lr_p},
        "models": models,
        "observation_weighting": weight_audit,
        "scenarios": scenarios,
        "bootstrap_stability": bootstrap,
        "warnings": [
            "This is an observational association model, not a causal effect estimate.",
            "Formal manuscript reporting requires bootstrap stability analysis and final contract hash verification.",
        ],
    }
    (out_dir / "PRIMARY_MODEL_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--lock-manifest", type=Path, required=True)
    args = parser.parse_args()
    lock = verify_analysis_lock(args.input_csv, args.contract, args.lock_manifest)
    print(json.dumps(run_analysis(args.input_csv, args.out_dir, lock), indent=2, ensure_ascii=False))
