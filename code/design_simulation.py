"""Design-stage operating-characteristic simulation for the 1-df primary test.

Uses the observed episode-duration designs but generates synthetic outcomes.
No observed exposure-outcome association is fitted. The target parameter is the
additional log-odds contribution per 10 minutes of episode time beyond a
prespecified 5-minute hinge, conditional on total hypotension duration.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
REPETITIONS = 2000
SEED = 20260923


def observed_design(path: Path, center: str) -> tuple[list[list[int]], float, int]:
    """Read exposure vectors and the marginal event rate without fitting an association."""
    frame = pd.read_csv(path)
    observed = frame[
        frame.outcome_observed_48h.astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    if center:
        observed = observed[observed.center.eq(center)].copy()
    outcomes = pd.to_numeric(observed.early_aki_48h, errors="coerce")
    if outcomes.isna().any() or observed.empty:
        raise ValueError(f"Missing or empty observed outcomes for {center or path.name}")
    runs = [json.loads(value) for value in observed.episode_durations_json]
    if any(not isinstance(value, list) for value in runs):
        raise ValueError(f"Invalid episode vector for {center or path.name}")
    return runs, float(outcomes.mean()), int(outcomes.sum())


def calibrate_intercept(linear_without_intercept: np.ndarray, target_rate: float) -> float:
    lo, hi = -20.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if expit(mid + linear_without_intercept).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    beta = np.zeros(x.shape[1])
    beta[0] = np.log((y.mean() + 0.5 / len(y)) / (1 - y.mean() + 0.5 / len(y)))
    for _ in range(50):
        p = expit(x @ beta)
        w = np.clip(p * (1 - p), 1e-8, None)
        hessian = x.T @ (w[:, None] * x)
        score = x.T @ (y - p)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return None
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            try:
                covariance = np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                return None
            return beta, covariance
    return None


def audit_design(name: str, runs: list[list[int]], event_rate: float, rng: np.random.Generator) -> list[dict]:
    total = np.asarray([sum(x) for x in runs], float)
    excess5 = np.asarray([sum(max(z - 5, 0) for z in x) for x in runs], float)
    x = np.column_stack([np.ones(len(runs)), total / 10.0, excess5 / 10.0])
    # Modest total-duration association; this is a design assumption, not an estimate.
    beta_total = np.log(1.10)
    output = []
    for scenario_or in [1.0, 1.3, 1.5, 2.0]:
        # A 1x20 versus 4x5 contrast differs by 15 excess minutes = 1.5 units.
        beta_excess = np.log(scenario_or) / 1.5
        intercept = calibrate_intercept(x[:, 1] * beta_total + x[:, 2] * beta_excess, event_rate)
        probability = expit(intercept + x[:, 1] * beta_total + x[:, 2] * beta_excess)
        significant = []
        estimates = []
        failures = 0
        for _ in range(REPETITIONS):
            y = rng.binomial(1, probability)
            fitted = logistic_fit(x, y)
            if fitted is None:
                failures += 1
                continue
            beta, covariance = fitted
            se = np.sqrt(covariance[2, 2])
            z = beta[2] / se
            significant.append(abs(z) > 1.959963984540054)
            estimates.append(float(np.exp(1.5 * beta[2])))
        output.append(
            {
                "dataset": name,
                "n": len(runs),
                "target_event_rate": event_rate,
                "scenario_or_1x20_vs_4x5": scenario_or,
                "valid_repetitions": len(significant),
                "fit_failures": failures,
                "two_sided_rejection_probability": float(np.mean(significant)),
                "median_estimated_scenario_or": float(np.median(estimates)),
                "estimated_or_q025_q975": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
            }
        )
    output.append(
        {
            "dataset": name,
            "design_correlation_total_vs_excess5": float(np.corrcoef(total, excess5)[0, 1]),
            "cases_with_any_excess_beyond_5_minutes": int((excess5 > 0).sum()),
            "excess5_q50_q75_q90_q95": [float(x) for x in np.quantile(excess5, [0.5, 0.75, 0.9, 0.95])],
        }
    )
    return output


def pooled_audit(
    mover: list[list[int]], vital: list[list[int]], mover_rate: float, vital_rate: float,
    rng: np.random.Generator
) -> list[dict]:
    runs = mover + vital
    center = np.r_[np.zeros(len(mover)), np.ones(len(vital))]
    total = np.asarray([sum(x) for x in runs], float)
    excess5 = np.asarray([sum(max(z - 5, 0) for z in x) for x in runs], float)
    x = np.column_stack([np.ones(len(runs)), center, total / 10.0, excess5 / 10.0])
    beta_total = np.log(1.10)
    output = []
    for scenario_or in [1.0, 1.3, 1.5, 2.0]:
        beta_excess = np.log(scenario_or) / 1.5
        common_lp = x[:, 2] * beta_total + x[:, 3] * beta_excess
        intercept_mover = calibrate_intercept(common_lp[center == 0], mover_rate)
        intercept_vital = calibrate_intercept(common_lp[center == 1], vital_rate)
        eta = intercept_mover + center * (intercept_vital - intercept_mover) + common_lp
        probability = expit(eta)
        significant = []
        estimates = []
        failures = 0
        for _ in range(REPETITIONS):
            y = rng.binomial(1, probability)
            fitted = logistic_fit(x, y)
            if fitted is None:
                failures += 1
                continue
            beta, covariance = fitted
            se = np.sqrt(covariance[3, 3])
            significant.append(abs(beta[3] / se) > 1.959963984540054)
            estimates.append(float(np.exp(1.5 * beta[3])))
        output.append(
            {
                "dataset": "Pooled_with_center_intercept",
                "n": len(runs),
                "scenario_or_1x20_vs_4x5": scenario_or,
                "valid_repetitions": len(significant),
                "fit_failures": failures,
                "two_sided_rejection_probability": float(np.mean(significant)),
                "median_estimated_scenario_or": float(np.median(estimates)),
                "estimated_or_q025_q975": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
            }
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "DESIGN_SIMULATION_RESULTS.json")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    REPETITIONS = args.repetitions
    generator = np.random.default_rng(SEED)
    results = {
        "purpose": "Prospective design simulation using observed exposure patterns and synthetic outcomes only",
        "primary_model": "logit(AKI) = intercept + beta_total*(total_minutes/10) + beta_excess*(sum(max(episode_minutes-5,0))/10)",
        "primary_test": "two-sided Wald test of beta_excess=0",
        "repetitions": REPETITIONS,
        "assumed_total_duration_or_per_10_minutes": 1.10,
        "limitations": [
            "Covariates and postoperative-creatinine observation weighting are not included in this first operating-characteristic screen.",
            "The observed center-specific AKI rates calibrate only the synthetic outcome intercepts; no observed exposure-outcome association is fitted.",
            "Power values are design guidance, not study results.",
        ],
        "results": [],
    }
    mover_design, mover_rate, mover_events = observed_design(args.combined, "MOVER")
    vital_design, vital_rate, vital_events = observed_design(args.combined, "VitalDB")
    results["design_source"] = str(args.combined)
    results["observed_margins_used_only_for_calibration"] = {
        "MOVER": {"n": len(mover_design), "events": mover_events, "event_rate": mover_rate},
        "VitalDB": {"n": len(vital_design), "events": vital_events, "event_rate": vital_rate},
    }
    results["results"] += audit_design("MOVER", mover_design, mover_rate, generator)
    results["results"] += audit_design("VitalDB", vital_design, vital_rate, generator)
    results["results"] += pooled_audit(mover_design, vital_design, mover_rate, vital_rate, generator)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
