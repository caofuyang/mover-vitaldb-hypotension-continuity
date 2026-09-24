"""Synthetic-outcome test for the locked primary model implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from fit_primary_model_v4 import run_analysis


def main() -> None:
    rng = np.random.default_rng(20260923)
    n = 6000
    center = rng.choice(["MOVER", "VitalDB"], n, p=[0.4, 0.6])
    age = np.clip(rng.normal(62, 13, n), 18, 95)
    sex = rng.choice(["Male", "Female"], n)
    asa = rng.choice([1, 2, 3, 4], n, p=[0.08, 0.35, 0.47, 0.10])
    egfr = np.clip(rng.normal(78, 24, n), 15, 140)
    category = rng.choice(["abdominal", "thoracic", "vascular", "genitourinary_gynecologic"], n)
    anesthesia = np.clip(rng.gamma(4, 45, n), 30, 700)
    total = rng.poisson(18, n)
    # Broad conditional variation in fragmentation at a fixed total duration.
    excess = np.minimum(total, np.rint(total * rng.uniform(0, 0.8, n)).astype(int))
    depth = np.where(total > 0, np.clip(rng.normal(6, 2, n), 1, 15), 0)
    eta = (
        -3.0
        + 0.8 * (center == "MOVER")
        + 0.12 * ((age - 60) / 10)
        + 0.08 * (total / 10)
        + np.log(1.45) / 1.5 * (excess / 10)
        + 0.12 * (depth / 5)
    )
    outcome = rng.binomial(1, expit(eta))
    observe_eta = 1.8 - 0.7 * (center == "VitalDB") + 0.15 * (asa - 2) + 0.001 * anesthesia
    observed = rng.binomial(1, expit(observe_eta)).astype(bool)
    frame = pd.DataFrame(
        {
            "center": center,
            "age": age,
            "sex": sex,
            "asa": asa,
            "baseline_egfr": egfr,
            "surgical_category": category,
            "common_surgical_support": True,
            "anesthesia_minutes": anesthesia,
            "total_low_minutes": total,
            "excess5_minutes": excess,
            "mean_depth_below_65_mmhg": depth,
            "outcome_observed_48h": observed,
            "early_aki_48h": np.where(observed, outcome, np.nan),
            "icu_flag": rng.choice(["Y", "N"], n),
        }
    )
    with tempfile.TemporaryDirectory(prefix="primary_model_test_") as temp:
        root = Path(temp)
        source = root / "synthetic.csv"
        output = root / "results"
        frame.to_csv(source, index=False)
        report = run_analysis(source, output)
        estimated = report["scenarios"]["pooled_primary"]["odds_ratio"]
        print("Synthetic estimated scenario OR:", estimated)
        assert report["status"] == "completed"
        assert 1.15 < estimated < 1.80
        assert (output / "PRIMARY_MODEL_COEFFICIENTS.csv").exists()
        assert report["observation_weighting"]["status"] == "available"
        print("PASS: synthetic primary model; scenario OR", round(estimated, 3))


if __name__ == "__main__":
    main()
