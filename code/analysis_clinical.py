"""Clinical-value analyses v2 (2026-10-18): shape, absolute risk by band, centre sensitivity.

Fixes relative to the v2.0.1 script
-----------------------------------
1. One spline specification, used by both the numbers reported in the text and
   the figure: restricted cubic spline with internal knots at 30, 60 and 90
   minutes and boundary knots at the lowest and highest observed duration (four
   basis columns after the redundant, collinear column is removed). The old
   analysis_clinical.py used patsy's default ``cr(x, df=4)`` without dropping
   the redundant column, which put knots at the minimum, two interior points and
   the maximum, and the figure script used a different, sample-based
   standardisation, so figure and text could not agree.
2. Standardised risks are averaged over the whole analytic cohort, not a random
   sample of 800 patients.
3. The point estimate is the standardised prediction at the fitted coefficients;
   the 400 draws from the sampling distribution (HC0 covariance) are used only
   for the interval. Averaging the draws would estimate a different quantity
   under a non-linear link.
4. Bands, spline and centre sensitivity are produced in one pass from the same
   prepared data, for all three cohorts.

Exploratory. Example:
    python analysis_clinical_v2.py --waveform-csv COMBINED_FEATURES_V4.csv \
        --resampled-csv downsampled_5min_features.csv \
        --inspire-csv INSPIRE_FEATURES_V2.csv --out clinical_v2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix

COVARIATES = "age10 + C(sex_group) + C(asa_group) + egfr10 + C(surgical_category)"
SPLINE = "cr(x, knots=[3, 6, 9])"
RNG = np.random.default_rng(20261018)
DRAWS = 400


def prep(path: Path, low: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])].copy()
    d["y"] = d.early_aki_48h.map({True: 1, False: 0, "True": 1, "False": 0,
                                  "true": 1, "false": 0}).astype(int)
    d["x"] = pd.to_numeric(d[low], errors="coerce") / 10.0
    d["age10"] = (pd.to_numeric(d.age, errors="coerce") - 60) / 10
    d["egfr10"] = (pd.to_numeric(d.baseline_egfr, errors="coerce").clip(upper=120) - 60) / 10
    d["sex_group"] = (d.sex.astype(str).str.lower().str[0]
                      .map({"m": "Male", "f": "Female"}).fillna("Unknown"))
    d["asa_group"] = pd.to_numeric(d.asa, errors="coerce").round().astype("Int64").astype(str)
    d["surgical_category"] = d.surgical_category.fillna("unmapped").astype(str)
    if "center" in d.columns:
        d["center"] = d.center.astype(str)
    return d.dropna(subset=["y", "x"]).reset_index(drop=True)


def fit_design(d: pd.DataFrame, rhs: str):
    Xfull = dmatrix(rhs, d, return_type="dataframe", NA_action="drop")
    redundant = [c for c in Xfull.columns if c.startswith("cr(x,")]
    drop = [redundant[0]] if redundant else []
    X = Xfull.drop(columns=drop)
    y = d.loc[X.index, "y"]
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC0")
    return model, Xfull, drop


def standardised(model, Xfull, drop, template: pd.DataFrame, values) -> dict:
    """Marginal standardisation over the whole cohort, with a Monte-Carlo CI."""
    betas = RNG.multivariate_normal(model.params.values, model.cov_params().values, DRAWS)
    out = []
    for v in values:
        s = template.copy()
        s["x"] = v / 10.0
        Xg = build_design_matrices([Xfull.design_info], s, return_type="dataframe")[0]
        Xg = Xg[model.params.index]
        beta_hat = model.params.loc[Xg.columns].to_numpy(float)
        point = float((1 / (1 + np.exp(-(Xg.values @ beta_hat)))).mean())
        eta, chunks = None, []
        for start in range(0, DRAWS, 50):
            chunk = betas[start:start + 50]
            p = 1 / (1 + np.exp(-(Xg.values @ chunk.T)))
            chunks.append(p.mean(axis=0))
        risk = np.concatenate(chunks)
        out.append({"value": v, "mean": point,
                    "lo": float(np.percentile(risk, 2.5)),
                    "hi": float(np.percentile(risk, 97.5))})
    return out


def bands(d: pd.DataFrame, has_center: bool) -> dict:
    edges = [-0.01, 0.1, 15, 30, 60, 1e9]
    labels = ["0", "1-15", "16-30", "31-60", ">60"]
    d = d.copy()
    d["band"] = pd.cut(d.x * 10, edges, labels=labels)
    rhs = "C(band) + " + COVARIATES + (" + C(center)" if has_center else "")
    Xfull = dmatrix(rhs, d, return_type="dataframe", NA_action="drop")
    model = sm.GLM(d.loc[Xfull.index, "y"], Xfull,
                   family=sm.families.Binomial()).fit(cov_type="HC0")
    betas = RNG.multivariate_normal(model.params.values, model.cov_params().values, DRAWS)
    result = {}
    crude = d.groupby("band", observed=True).y.agg(["size", "sum", "mean"])
    for lab in labels:
        s = d.copy()
        s["band"] = lab
        Xg = build_design_matrices([Xfull.design_info], s, return_type="dataframe")[0]
        Xg = Xg[model.params.index]
        beta_hat = model.params.loc[Xg.columns].to_numpy(float)
        point = float((1 / (1 + np.exp(-(Xg.values @ beta_hat)))).mean())
        risks = []
        for start in range(0, DRAWS, 50):
            chunk = betas[start:start + 50]
            risks.append((1 / (1 + np.exp(-(Xg.values @ chunk.T)))).mean(axis=0))
        risk = np.concatenate(risks)
        result[lab] = {
            "n": int(crude.loc[lab, "size"]), "events": int(crude.loc[lab, "sum"]),
            "crude": round(float(crude.loc[lab, "mean"]), 4),
            "standardised": round(point, 4),
            "lo": round(float(np.percentile(risk, 2.5)), 4),
            "hi": round(float(np.percentile(risk, 97.5)), 4)}
    return result


def curve(d: pd.DataFrame, has_center: bool, grid) -> dict:
    rhs = COVARIATES + (" + C(center)" if has_center else "") + " + " + SPLINE
    model, Xfull, drop = fit_design(d, rhs)
    points = standardised(model, Xfull, drop, d, list(grid))
    lin_rhs = COVARIATES + (" + C(center)" if has_center else "") + " + x"
    lin_model, lin_X, _ = fit_design(d, lin_rhs)
    lin_points = standardised(lin_model, lin_X, [], d, [120])
    return {"n": int(model.nobs), "events": int(d.y.sum()),
            "grid": list(grid),
            "risk": [round(p["mean"], 4) for p in points],
            "lo": [round(p["lo"], 4) for p in points],
            "hi": [round(p["hi"], 4) for p in points],
            "linear_at_120": round(lin_points[0]["mean"], 4)}


def centre_sensitivity(d: pd.DataFrame) -> dict:
    out = {}
    for centre, g in d.groupby("center"):
        for label, rhs in [("crude", "x"), ("adjusted", COVARIATES + " + x")]:
            X = dmatrix(rhs, g, return_type="dataframe", NA_action="drop")
            m = sm.GLM(g.loc[X.index, "y"], X, family=sm.families.Binomial()).fit(cov_type="HC0")
            out[f"{centre} only, {label}"] = {
                "n": int(m.nobs), "OR": round(float(np.exp(m.params["x"])), 3),
                "lo": round(float(np.exp(m.params["x"] - 1.96 * m.bse["x"])), 3),
                "hi": round(float(np.exp(m.params["x"] + 1.96 * m.bse["x"])), 3),
                "p": round(float(m.pvalues["x"]), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waveform-csv", type=Path, required=True)
    ap.add_argument("--resampled-csv", type=Path, required=True)
    ap.add_argument("--inspire-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    wave = prep(a.waveform_csv, "total_low_minutes")
    resampled = prep(a.resampled_csv, "ds_total")
    insp = prep(a.inspire_csv, "total_low_minutes")

    grid = sorted(set(range(0, 121, 2)) | {45, 60, 90, 120})
    key = [0, 45, 60, 90, 120]
    out = {"spline_spec": {"type": "restricted cubic spline (patsy cr)",
                           "internal_knots_minutes": [30, 60, 90],
                           "boundary_knots": "minimum and maximum observed cumulative duration in each cohort",
                           "basis_columns_after_dropping_redundant": 4,
                           "standardisation": "marginal, averaged over the whole cohort",
                           "draws": DRAWS},
           "key_minutes": key, "curves": {}, "bands": {}, "sensitivity": {}}

    for name, d, has_center in [("pooled_waveform_1min", wave, True),
                                ("pooled_resampled_5min", resampled, True),
                                ("inspire", insp, False)]:
        c = curve(d, has_center, grid)
        out["curves"][name] = {
            "n": c["n"], "events": c["events"],
            "risk_at_key": [round(c["risk"][grid.index(k)], 4) for k in key],
            "lo_at_key": [round(c["lo"][grid.index(k)], 4) for k in key],
            "hi_at_key": [round(c["hi"][grid.index(k)], 4) for k in key],
            "linear_at_120": c["linear_at_120"],
            "grid": c["grid"], "risk": c["risk"], "lo": c["lo"], "hi": c["hi"]}
        out["bands"][name] = bands(d, has_center)
        print(name, "n", c["n"], "events", c["events"],
              "risk@key", out["curves"][name]["risk_at_key"])

    out["sensitivity"] = centre_sensitivity(wave)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
