"""The two remaining prespecified secondary analyses from the locked contract.

  seven_day_ratio_component
  next_surgery_censoring

Both are computed from the same raw sources as the locked pipeline. The 48-hour
outcome is recomputed first as a self-check: it must reproduce the archived
200 events, which validates the time conventions used here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPORT = Path("/sessions/zen-beautiful-bohr/mnt/caofu/Downloads/"
              "Methodology_Upgrade_v4_MOVER_v5_1_20260923/methodology_upgrade_v4/"
              "MOVER_episode_export_v5_1_20260923_155532")
VITAL = Path("/sessions/zen-beautiful-bohr/mnt/e/physionet/vitaldb")
COMBINED = Path("/tmp/pkg_run/combined_features/COMBINED_FEATURES_V4.csv")

combined = pd.read_csv(COMBINED)


def resolve(group: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    """Same-timestamp duplicate rule: median when the spread is small."""
    rows = []
    for stamp, values in group.groupby(time_col):
        x = pd.to_numeric(values[value_col], errors="coerce")
        x = x[np.isfinite(x) & x.gt(0) & x.ne(9999999)]
        if x.empty:
            continue
        spread = float(x.max() - x.min())
        if spread <= 0.1 + 1e-10:            # locked rule: exclude conflicting timestamps
            rows.append((float(stamp), float(x.median())))
    return pd.DataFrame(rows, columns=["time_seconds", "creatinine"]).sort_values("time_seconds")


def aki(peak: float, base: float) -> tuple[bool, bool]:
    absolute = peak - base >= 0.3 - 1e-10
    ratio = peak / base >= 1.5 - 1e-10
    return (absolute or ratio), ratio


# ------------------------------------------------------------------ MOVER
mover = combined[combined.center.eq("MOVER")].copy()
mover["export_case"] = mover.analysis_case.str.replace("M_", "", regex=False).astype(int)
parent = pd.read_csv(EXPORT / "cohort_parent.csv")
parent = parent[["export_case", "next_operation_hours"]]
mover = mover.merge(parent, on="export_case", how="left")

creatinine = pd.read_csv(EXPORT / "creatinine.csv")
creatinine = creatinine[creatinine.creatinine.notna()]
creatinine["creatinine"] = pd.to_numeric(creatinine.creatinine, errors="coerce")
creatinine = creatinine[creatinine.creatinine.ne(9999999)]
labs_by_case = {case: resolve(group, "time_seconds", "creatinine")
                for case, group in creatinine.groupby("export_case")}

mover_out = []
for row in mover.itertuples():
    lab = labs_by_case.get(row.export_case, pd.DataFrame())
    end_seconds = float(row.anesthesia_minutes) * 60
    result = {"analysis_case": row.analysis_case, "baseline": row.baseline_creatinine,
              "end_seconds": end_seconds, "next_operation_hours": row.next_operation_hours}
    for label, hours in [("48h", 48), ("7d", 168)]:
        window_end = end_seconds + hours * 3600
        if np.isfinite(row.next_operation_hours):
            window_end = min(window_end, float(row.next_operation_hours) * 3600)
        post = lab[(lab.time_seconds >= end_seconds) & (lab.time_seconds <= window_end)] if len(lab) else pd.DataFrame()
        if len(post) and np.isfinite(row.baseline_creatinine):
            peak = float(post.creatinine.max())
            full, ratio = aki(peak, float(row.baseline_creatinine))
            result[f"observed_{label}"] = True
            result[f"aki_{label}"] = full
            result[f"ratio_{label}"] = ratio
        else:
            result[f"observed_{label}"] = False
            result[f"aki_{label}"] = None
            result[f"ratio_{label}"] = None
    mover_out.append(result)
mover_out = pd.DataFrame(mover_out)

# ---------------------------------------------------------------- VitalDB
vital = combined[combined.center.eq("VitalDB")].copy()
vital["caseid"] = vital.analysis_case.str.replace("V_", "", regex=False).astype(int)

clinical = pd.read_csv(VITAL / "clinical_data.csv",
                       usecols=["caseid", "subjectid", "anestart", "aneend"])
clinical["anestart"] = pd.to_numeric(clinical.anestart, errors="coerce")
clinical["aneend"] = pd.to_numeric(clinical.aneend, errors="coerce")
clinical = clinical.sort_values(["subjectid", "anestart"])
clinical["next_operation_start"] = clinical.groupby("subjectid").anestart.shift(-1)
vital = vital.merge(clinical[["caseid", "subjectid", "anestart", "aneend", "next_operation_start"]],
                    on="caseid", how="left")

labs = pd.read_csv(VITAL / "lab_data.csv")
labs = labs[labs.name.astype(str).str.lower().eq("cr")].copy()
labs["result"] = pd.to_numeric(labs.result, errors="coerce")
labs = labs[labs.result.notna() & labs.result.gt(0)]
vital_labs = {case: resolve(group, "dt", "result") for case, group in labs.groupby("caseid")}

vital_out = []
for row in vital.itertuples():
    lab = vital_labs.get(row.caseid, pd.DataFrame())
    end_seconds = float(row.aneend)
    result = {"analysis_case": row.analysis_case, "baseline": row.baseline_creatinine,
              "end_seconds": end_seconds, "next_operation_start": row.next_operation_start}
    for label, hours in [("48h", 48), ("7d", 168)]:
        # The locked VitalDB pipeline does not cap the window at the next operation;
        # only the MOVER branch does.
        window_end = end_seconds + hours * 3600
        post = lab[(lab.time_seconds >= end_seconds) & (lab.time_seconds <= window_end)] if len(lab) else pd.DataFrame()
        if len(post) and np.isfinite(row.baseline_creatinine):
            peak = float(post.creatinine.max())
            full, ratio = aki(peak, float(row.baseline_creatinine))
            result[f"observed_{label}"] = True
            result[f"aki_{label}"] = full
            result[f"ratio_{label}"] = ratio
        else:
            result[f"observed_{label}"] = False
            result[f"aki_{label}"] = None
            result[f"ratio_{label}"] = None
    vital_out.append(result)
vital_out = pd.DataFrame(vital_out)

raw = pd.concat([mover_out, vital_out], ignore_index=True)
raw = raw.merge(combined[["analysis_case", "center", "outcome_observed_48h", "early_aki_48h"]],
                on="analysis_case", how="left")

# ----------------------------------------------------------- self-check
check = raw[raw.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])].copy()
archived = int(pd.to_numeric(check.early_aki_48h, errors="coerce").sum())
recomputed = int(check.aki_48h.fillna(False).sum())
per_center = check.groupby("center").apply(
    lambda g: pd.Series({"archived": pd.to_numeric(g.early_aki_48h, errors="coerce").sum(),
                         "recomputed": g.aki_48h.fillna(False).sum()}), include_groups=False)

report = {
    "self_check_48h": {
        "archived_events": archived,
        "recomputed_events": recomputed,
        "match": bool(archived == recomputed),
        "by_center": per_center.to_dict("index"),
    },
    "seven_day_ratio_component": {
        "description": "7-day postoperative creatinine outcome, ratio criterion only "
                       "(peak/baseline >= 1.5); window is capped by the next operation in MOVER only",
        "observed_n": int(raw.observed_7d.fillna(False).sum()),
        "events_ratio_only": int(raw.ratio_7d.fillna(False).sum()),
        "events_full_7d": int(raw.aki_7d.fillna(False).sum()),
        "by_center": raw[raw.observed_7d.fillna(False)].groupby("center").apply(
            lambda g: pd.Series({"observed": len(g),
                                 "events_ratio_only": int(g.ratio_7d.fillna(False).sum()),
                                 "events_full_7d": int(g.aki_7d.fillna(False).sum())}),
            include_groups=False).to_dict("index"),
    },
    "next_surgery_censoring": {
        "description": "exclude operations whose next operation started before the end of the "
                       "48-hour outcome window",
        "mover_with_next_operation_within_48h": int(
            ((mover_out.end_seconds + 48 * 3600) > (mover_out.next_operation_hours.fillna(np.inf) * 3600)).sum()),
        "vitaldb_with_next_operation_within_48h": int(
            ((vital_out.end_seconds + 48 * 3600) > vital_out.next_operation_start.fillna(np.inf)).sum()),
        "note": "MOVER already caps the outcome window at the next operation, so the counts above "
                "describe how many operations the sensitivity analysis removes rather than a defect.",
    },
}

# --- actually fit the censoring sensitivity: drop operations whose next operation
#     started before the 48-hour outcome window closed -------------------------
import sys
sys.path.insert(0, "/tmp/pkg/code")
import fit_primary_model_v4 as fp  # noqa: E402
import patsy  # noqa: E402
from scipy.special import expit  # noqa: E402

censor = {}
for name, frame, start_col, next_col in [
    ("MOVER", mover_out, "end_seconds", "next_operation_hours"),
    ("VitalDB", vital_out, "end_seconds", "next_operation_start"),
]:
    if name == "MOVER":
        border = frame[next_col].fillna(np.inf) * 3600
    else:
        border = frame[next_col].fillna(np.inf)
    censor[name] = set(frame.loc[frame[start_col] + 48 * 3600 > border, "analysis_case"])

excluded = censor["MOVER"] | censor["VitalDB"]
analysis = fp.prepare(combined)
covariates = ["age10", "egfr10", "anesthesia_hours", "total_low10", "excess5_10", "mean_depth5"]
analysis = analysis[analysis[covariates].notna().all(axis=1)]
observed_rows = analysis[analysis.outcome_observed_48h & analysis.early_aki_48h.notna()].copy()
kept = observed_rows[~observed_rows.analysis_case.isin(excluded)].copy()

formula = "early_aki_48h ~ " + " + ".join(fp.PRIMARY_TERMS)
result = fp.fit_glm(formula, kept)
beta = float(result.params["excess5_10"]); se = float(result.bse["excess5_10"])
report["next_surgery_censoring"]["refit"] = {
    "n": int(len(kept)),
    "events": int(kept.early_aki_48h.sum()),
    "excluded_operations": int(len(excluded)),
    "scenario_odds_ratio": float(np.exp(1.5 * beta)),
    "scenario_odds_ratio_95ci": [float(np.exp(1.5 * (beta - 1.959963984540054 * se))),
                                 float(np.exp(1.5 * (beta + 1.959963984540054 * se)))],
    "wald_p": float(result.pvalues["excess5_10"]),
}

# --- refit the prespecified scenario against the 7-day outcome ----------------
seven = raw.merge(analysis[["analysis_case", "age10", "sex_group", "asa_group",
                            "egfr10", "surgical_category", "anesthesia_hours",
                            "total_low10", "excess5_10", "mean_depth5"]],
                  on="analysis_case", how="inner")
seven = seven[seven.observed_7d.astype(bool)]
seven_results = {}
for label, column in [("full_criterion", "aki_7d"), ("ratio_only", "ratio_7d")]:
    frame = seven.copy()
    frame["outcome_7d"] = frame[column].astype(float)
    formula_7d = formula.replace("early_aki_48h", "outcome_7d")
    fit = fp.fit_glm(formula_7d, frame)
    b = float(fit.params["excess5_10"]); s = float(fit.bse["excess5_10"])
    seven_results[label] = {
        "n": int(len(frame)),
        "events": int(frame.outcome_7d.sum()),
        "scenario_odds_ratio": float(np.exp(1.5 * b)),
        "scenario_odds_ratio_95ci": [float(np.exp(1.5 * (b - 1.959963984540054 * s))),
                                     float(np.exp(1.5 * (b + 1.959963984540054 * s)))],
        "wald_p": float(fit.pvalues["excess5_10"]),
    }
report["seven_day_ratio_component"]["refit"] = seven_results

# breakdown of the primary 48-hour outcome by KDIGO criterion
obs48 = raw[raw.outcome_observed_48h.astype(str).str.lower().isin(["true","1"])].copy()
report["primary_48h_outcome_by_criterion"] = {
    "events_total": int(obs48.aki_48h.fillna(False).sum()),
    "ratio_criterion": int(obs48.ratio_48h.fillna(False).sum()),
    "absolute_only": int((obs48.aki_48h.fillna(False) & ~obs48.ratio_48h.fillna(False)).sum()),
    "ratio_only": int((~obs48.aki_48h.fillna(False) & obs48.ratio_48h.fillna(False)).sum()),
    "both": int((obs48.aki_48h.fillna(False) & obs48.ratio_48h.fillna(False)).sum()),
    "note": "absolute_only meets the 0.3 mg/dl criterion without reaching 1.5x baseline",
}
print(json.dumps(report["primary_48h_outcome_by_criterion"], indent=2))
Path("/tmp/work/REMAINING_SECONDARY.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
