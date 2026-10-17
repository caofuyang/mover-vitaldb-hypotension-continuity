"""Absolute-risk figure: standardised 48-hour AKI risk against cumulative minutes
of mean arterial pressure below 65 mmHg, in the waveform cohort and in INSPIRE,
with 95% confidence bands from the model covariance.

Exploratory. Example:

    python code/make_figure_absolute_risk.py \
        --waveform-csv COMBINED_FEATURES_V4.csv \
        --inspire-csv INSPIRE_FEATURES_HOLD.csv \
        --out-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix

BASE = "age10 + C(sex_g) + C(asa_g) + egfr10 + C(surg)"
RNG = np.random.default_rng(11)


def prep(path: Path, low: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])].copy()
    d["y"] = d.early_aki_48h.map({True: 1, False: 0, "True": 1, "False": 0}).astype(int)
    d["x"] = pd.to_numeric(d[low], errors="coerce") / 10
    d["age10"] = (d.age - 60) / 10
    d["egfr10"] = (d.baseline_egfr.clip(upper=120) - 60) / 10
    d["sex_g"] = (d.sex.astype(str).str.lower().str[0]
                  .map({"m": "Male", "f": "Female"}).fillna("Unknown"))
    d["asa_g"] = d.asa.round().astype(int).astype(str)
    d["surg"] = d.surgical_category.fillna("unmapped").astype(str)
    if "center" in d.columns:
        d["ctr"] = d.center.astype(str)
    return d


def standardised_curve(d: pd.DataFrame, rhs: str, grid: np.ndarray,
                       n_standardise: int = 800, draws: int = 600):
    """Mean standardised risk at each grid value, with a 95% band from the
    multivariate normal approximation to the coefficient estimates."""
    Xfull = dmatrix(rhs, d, return_type="dataframe", NA_action="drop")
    # patsy's cr() basis sums to the intercept, so one basis column is redundant.
    # Dropping it leaves the same model space and a full-rank design.
    redundant = [c for c in Xfull.columns if c.startswith("cr(x,")][0]
    X = Xfull.drop(columns=[redundant])
    y = d.loc[X.index, "y"]
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC0")
    sub = d.loc[X.index].sample(min(n_standardise, len(X)), random_state=3)
    betas = RNG.multivariate_normal(m.params.values, m.cov_params().values, draws)
    risk = np.zeros((draws, len(grid)))
    for j, g in enumerate(grid):
        s = sub.copy()
        s["x"] = g / 10
        Xg = pd.DataFrame(build_design_matrices([Xfull.design_info], s)[0],
                          columns=Xfull.columns).drop(columns=[redundant])
        eta = Xg.values @ betas.T
        risk[:, j] = (1 / (1 + np.exp(-eta))).mean(axis=0)
    return risk.mean(axis=0), np.percentile(risk, 2.5, axis=0), np.percentile(risk, 97.5, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waveform-csv", type=Path, required=True)
    ap.add_argument("--inspire-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    wave = prep(args.waveform_csv, "total_low_minutes")
    insp = prep(args.inspire_csv, "total_low_minutes")
    grid = np.arange(0, 121, 2)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for d, rhs, label, colour in [
        (wave, BASE + " + C(ctr) + cr(x, df=4)", "MOVER + VitalDB (waveform)", "#1f4e79"),
        (insp, BASE + " + cr(x, df=4)", "INSPIRE (charted ~5-min)", "#b3541e"),
    ]:
        mid, lo, hi = standardised_curve(d, rhs, grid)
        ax.fill_between(grid, lo * 100, hi * 100, color=colour, alpha=0.15, lw=0)
        ax.plot(grid, mid * 100, color=colour, lw=2, label=label)

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 18)
    ax.set_xlabel("Cumulative minutes with mean arterial pressure below 65 mmHg", fontsize=9)
    ax.set_ylabel("Standardised 48-hour AKI risk (%)", fontsize=9)
    ax.set_title("Absolute risk by cumulative hypotension", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02,
             "Shaded areas are 95% confidence bands. Adjusted for age, sex, ASA status, baseline eGFR "
             "and surgical category; restricted cubic splines (4 df). Exploratory.",
             ha="center", fontsize=7, color="#555555")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 600), ("pdf", None), ("svg", None), ("eps", None)]:
        fig.savefig(args.out_dir / f"Fig_absolute_risk.{ext}", dpi=dpi, bbox_inches="tight")
    print("written to", args.out_dir)


if __name__ == "__main__":
    main()
