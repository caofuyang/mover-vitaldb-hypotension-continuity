"""Post hoc clinical-value analyses: shape of the exposure-outcome relation,
absolute risk by band, and a centre-level sensitivity analysis.

All outputs are exploratory. Example:

    python code/analysis_clinical.py \
        --waveform-csv COMBINED_FEATURES_V4.csv \
        --resampled-csv downsampled_5min_features.csv \
        --inspire-csv INSPIRE_FEATURES_HOLD.csv \
        --out clinical_analyses.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = "age10 + C(sex_g) + C(asa_g) + egfr10 + C(surg)"


def prep(path: Path, low: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])].copy()
    d["y"] = d.early_aki_48h.map({True: 1, False: 0, "True": 1, "False": 0}).astype(int)
    d["x"] = pd.to_numeric(d[low], errors="coerce") / 10
    d["age10"] = (d.age - 60) / 10
    d["egfr10"] = (d.baseline_egfr.clip(upper=120) - 60) / 10
    d["an_h"] = d.anesthesia_minutes / 60
    d["sex_g"] = (d.sex.astype(str).str.lower().str[0]
                  .map({"m": "Male", "f": "Female"}).fillna("Unknown"))
    d["asa_g"] = d.asa.round().astype(int).astype(str)
    d["surg"] = d.surgical_category.fillna("unmapped").astype(str)
    if "center" in d.columns:
        d["ctr"] = d.center.astype(str)
    return d


def fit(formula: str, data: pd.DataFrame):
    return smf.glm(formula, data=data, family=sm.families.Binomial(),
                   missing="drop").fit(cov_type="HC0")


def risk_at(model, template: pd.DataFrame, values):
    rows = []
    for v in values:
        d = template.copy()
        d["x"] = v / 10
        rows.append(float(model.predict(d).mean()))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--waveform-csv", type=Path, required=True)
    p.add_argument("--resampled-csv", type=Path, required=True)
    p.add_argument("--inspire-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("clinical_analyses.json"))
    a = p.parse_args()

    wave = prep(a.waveform_csv, "total_low_minutes")
    insp = prep(a.inspire_csv, "total_low_minutes")

    cohorts = {
        "pooled_waveform": (wave, BASE + " + C(ctr) + cr(x, df=4)"),
        "inspire": (insp, BASE + " + cr(x, df=4)"),
    }
    grid = [0, 10, 20, 30, 45, 60, 90, 120]
    out = {"grid_minutes": grid, "curves": {}}

    for name, (d, rhs) in cohorts.items():
        m_spline = fit("y ~ " + rhs, d)
        m_linear = fit("y ~ " + rhs.replace(" + cr(x, df=4)", " + x"), d)
        out["curves"][name] = {
            "n": int(m_spline.nobs),
            "events": int(d.y.sum()),
            "risk_spline": [round(r, 4) for r in risk_at(m_spline, d, grid)],
            "risk_linear": [round(r, 4) for r in risk_at(m_linear, d, grid)],
        }
        print(f"{name}: n={int(m_spline.nobs)} events={int(d.y.sum())}")

    bands = [(-0.1, 0.1, "0"), (0.1, 15, "1-15"), (15, 30, "16-30"),
             (30, 60, "31-60"), (60, 1e6, ">60")]
    band_out = {}
    for name, d in [("pooled_waveform", wave), ("inspire", insp)]:
        d = d.copy()
        d["band"] = pd.cut(d.x * 10, [b[0] for b in bands] + [1e6],
                           labels=[b[2] for b in bands])
        g = d.groupby("band", observed=True).agg(n=("y", "size"), events=("y", "sum"),
                                                 crude=("y", "mean"))
        band_out[name] = {str(k): {"n": int(v.n), "events": int(v.events),
                                   "crude": round(float(v.crude), 4)}
                          for k, v in g.iterrows()}

    sens = {}
    for label, d, rhs in [
        ("VitalDB only, crude", wave[wave.ctr == "VitalDB"], "y ~ x"),
        ("VitalDB only, adjusted", wave[wave.ctr == "VitalDB"], "y ~ " + BASE + " + x"),
        ("MOVER only, crude", wave[wave.ctr == "MOVER"], "y ~ x"),
        ("MOVER only, adjusted", wave[wave.ctr == "MOVER"], "y ~ " + BASE + " + x"),
    ]:
        m = fit(rhs, d)
        sens[label] = {"n": int(m.nobs), "OR": round(float(np.exp(m.params.x)), 3),
                       "lo": round(float(np.exp(m.params.x - 1.96 * m.bse.x)), 3),
                       "hi": round(float(np.exp(m.params.x + 1.96 * m.bse.x)), 3),
                       "p": round(float(m.pvalues.x), 4)}
        print(f"{label}: {sens[label]}")

    out["bands"] = band_out
    out["sensitivity"] = sens
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
