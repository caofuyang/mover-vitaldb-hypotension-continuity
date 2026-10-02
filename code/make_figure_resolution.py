"""Figure 1: cumulative hypotension duration and AKI, by measurement resolution.

Post hoc and exploratory. The figure is regenerated from the cohort feature
tables rather than from hard-coded numbers.

Usage
-----
python code/make_figure_resolution.py \
    --waveform-csv COMBINED_FEATURES_V4.csv \
    --resampled-csv downsampled_5min_features.csv \
    --inspire-csv INSPIRE_FEATURES_HOLD.csv \
    --out-dir figures

The resampled file must contain the column `ds_total`; the other two must
contain `total_low_minutes`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COVARIATES = "age10 + C(sex_group) + C(asa_group) + egfr10 + C(surgical_category)"
WAVEFORM, CHARTED = "#1f4e79", "#b3541e"


def prepare(frame: pd.DataFrame, low_column: str) -> pd.DataFrame:
    d = frame[frame.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])].copy()
    d["age10"] = (pd.to_numeric(d.age, errors="coerce") - 60) / 10
    d["egfr10"] = (pd.to_numeric(d.baseline_egfr, errors="coerce").clip(upper=120) - 60) / 10
    d["anesthesia_hours"] = pd.to_numeric(d.anesthesia_minutes, errors="coerce") / 60
    d["total_low10"] = pd.to_numeric(d[low_column], errors="coerce") / 10
    d["sex_group"] = (d.sex.astype(str).str.lower().str[0]
                      .map({"m": "Male", "f": "Female"}).fillna("Unknown"))
    d["asa_group"] = pd.to_numeric(d.asa, errors="coerce").round().astype("Int64").astype(str)
    d["surgical_category"] = d.surgical_category.fillna("unmapped").astype(str)
    d["early_aki_48h"] = pd.to_numeric(d.early_aki_48h, errors="coerce")
    if "center" in d.columns:
        d["center"] = d.center.astype(str)
    return d


def estimate(rhs: str, data: pd.DataFrame):
    r = smf.glm("early_aki_48h ~ " + rhs, data=data, family=sm.families.Binomial(),
                missing="drop").fit(maxiter=100, cov_type="HC0")
    b, se = float(r.params["total_low10"]), float(r.bse["total_low10"])
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)


def forest(ax, rows, title, colors):
    y = np.arange(len(rows))[::-1]
    for yi, (label, orv, lo, hi), colour in zip(y, rows, colors):
        ax.plot([lo, hi], [yi, yi], color=colour, lw=1.6, solid_capstyle="round")
        ax.plot([orv], [yi], "o", color=colour, ms=6.5)
        ax.text(1.30, yi, f"{orv:.3f} ({lo:.3f}-{hi:.3f})", va="center", ha="left",
                fontsize=7.6, color="#222222")
    ax.axvline(1.0, color="#888888", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.78, 1.28)
    ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax.set_xticklabels(["0.8", "0.9", "1.0", "1.1", "1.2"], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("Odds ratio per 10 min below 65 mmHg", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main(args) -> None:
    wave = prepare(pd.read_csv(args.waveform_csv), "total_low_minutes")
    res = prepare(pd.read_csv(args.resampled_csv), "ds_total")
    insp = prepare(pd.read_csv(args.inspire_csv), "total_low_minutes")

    panel_a = [
        ("Waveform, 1-minute\n(MOVER + VitalDB)",
         *estimate(f"{COVARIATES} + C(center) + total_low10", wave)),
        ("Resampled to 5-minute\n(same patients)",
         *estimate(f"{COVARIATES} + C(center) + total_low10", res)),
        ("Charted ~5-minute\n(INSPIRE)",
         *estimate(f"{COVARIATES} + total_low10", insp)),
    ]
    panel_b = []
    for label, d in [("1-minute", wave), ("5-minute", res)]:
        for centre in sorted(d.center.unique()):
            panel_b.append((f"{centre}\n({label})",
                            *estimate("total_low10", d[d.center == centre])))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1, 1]})
    forest(axes[0], panel_a, "A  Pooled cohort, adjusted", [WAVEFORM, CHARTED, CHARTED])
    forest(axes[1], panel_b, "B  Each centre, unadjusted", [WAVEFORM] * 2 + [CHARTED] * 2)
    fig.text(0.5, -0.015,
             "Filled circles are point estimates with 95% confidence intervals. "
             "Adjustment in panel A: age, sex, ASA status, baseline eGFR and surgical category.",
             ha="center", fontsize=7, color="#555555")
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 600), ("pdf", None), ("svg", None), ("eps", None)]:
        fig.savefig(args.out_dir / f"Fig_resolution_duration_aki.{ext}", dpi=dpi, bbox_inches="tight")

    print("Panel A (adjusted OR, 95% CI):")
    for label, o, l, h in panel_a:
        print(f"  {label.replace(chr(10), ' '):42s} {o:.3f} ({l:.3f}-{h:.3f})")
    print("Panel B (crude OR, 95% CI):")
    for label, o, l, h in panel_b:
        print(f"  {label.replace(chr(10), ' '):42s} {o:.3f} ({l:.3f}-{h:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--waveform-csv", type=Path, required=True)
    p.add_argument("--resampled-csv", type=Path, required=True)
    p.add_argument("--inspire-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    main(p.parse_args())
