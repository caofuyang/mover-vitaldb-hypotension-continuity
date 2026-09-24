"""Recompute prespecified hypotension exposures while preserving the locked cohort."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def features_from_points(
    time_seconds: pd.Series,
    map_value: pd.Series,
    anesthesia_minutes: float,
    *,
    threshold: float,
    bin_seconds: int,
    bridge_one_missing_bin: bool = False,
) -> dict:
    n = int(np.ceil(anesthesia_minutes * 60 / bin_seconds))
    grid = np.full(n, np.nan)
    time = pd.to_numeric(time_seconds, errors="coerce").to_numpy(float)
    value = pd.to_numeric(map_value, errors="coerce").to_numpy(float)
    keep = (
        np.isfinite(time) & np.isfinite(value) & (time >= 0)
        & (time < anesthesia_minutes * 60) & (value >= 20) & (value <= 160)
    )
    bins = np.floor(time[keep] / bin_seconds).astype(int)
    if len(bins):
        medians = pd.Series(value[keep]).groupby(bins).median()
        index = medians.index.to_numpy(int)
        index = index[(index >= 0) & (index < n)]
        grid[index] = medians.loc[index].to_numpy(float)
    observed = np.isfinite(grid)
    low = observed & (grid < threshold)
    bridged = np.zeros(n, dtype=bool)
    if bridge_one_missing_bin and n >= 3:
        bridged[1:-1] = (~observed[1:-1]) & low[:-2] & low[2:]
        low = low | bridged
    working = grid.copy()
    if bridged.any():
        indices = np.flatnonzero(bridged)
        working[indices] = (grid[indices - 1] + grid[indices + 1]) / 2
    transitions = np.diff(np.r_[False, low, False].astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    episode_minutes = ((stops - starts) * bin_seconds / 60).tolist()
    total_minutes = float(low.sum() * bin_seconds / 60)
    depth_auc = float(np.where(low, threshold - working, 0).sum() * bin_seconds / 60)
    return {
        "map_observed_minutes": float(observed.sum() * bin_seconds / 60),
        "map_total_minutes": float(n * bin_seconds / 60),
        "map_coverage": float(observed.mean()) if n else np.nan,
        "total_low_minutes": total_minutes,
        "episode_count": int(len(episode_minutes)),
        "longest_episode_minutes": float(max(episode_minutes)) if episode_minutes else 0.0,
        "excess5_minutes": float(sum(max(value - 5, 0) for value in episode_minutes)),
        "excess10_minutes": float(sum(max(value - 10, 0) for value in episode_minutes)),
        "auc_below_threshold_mmhg_minutes": depth_auc,
        "mean_depth_below_threshold_mmhg": depth_auc / total_minutes if total_minutes else 0.0,
        "any_hypotension": int(total_minutes > 0),
        "episode_durations_json": json.dumps(episode_minutes, separators=(",", ":")),
        "bridged_missing_bins": int(bridged.sum()),
    }


def prefixed(features: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in features.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--mover-archive", type=Path, required=True)
    parser.add_argument("--vitaldb-cases", type=Path, required=True)
    parser.add_argument("--vitaldb-manifest", type=Path, required=True)
    parser.add_argument("--vitaldb-map-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    combined = pd.read_csv(args.combined)
    locked = combined.set_index("analysis_case")
    rows = []

    with zipfile.ZipFile(args.mover_archive) as handle, tempfile.TemporaryDirectory(prefix="mover_sensitivity_") as temp:
        member = next(name for name in handle.namelist() if Path(name).name == "map.csv")
        map_path = Path(temp) / "map.csv"
        map_path.write_bytes(handle.read(member))
        mover_map = pd.read_csv(map_path)
    mover_map["analysis_case"] = "M_" + mover_map.export_case.astype(str)
    mover_map = mover_map[mover_map.analysis_case.isin(locked.index)]
    mover_groups = dict(tuple(mover_map.groupby("analysis_case")))

    for analysis_case, cohort_row in locked[locked.center.eq("MOVER")].iterrows():
        points = mover_groups.get(analysis_case)
        if points is None:
            raise RuntimeError(f"Missing MOVER MAP for {analysis_case}")
        row = {"analysis_case": analysis_case, "center": "MOVER"}
        for threshold in [60, 65, 70]:
            result = features_from_points(
                points.time_seconds, points.map_value, float(cohort_row.anesthesia_minutes),
                threshold=threshold, bin_seconds=60,
            )
            row.update(prefixed(result, f"t{threshold}_60s"))
        bridge = features_from_points(
            points.time_seconds, points.map_value, float(cohort_row.anesthesia_minutes),
            threshold=65, bin_seconds=60, bridge_one_missing_bin=True,
        )
        row.update(prefixed(bridge, "t65_60s_bridge1missing"))
        rows.append(row)

    cases = pd.read_csv(args.vitaldb_cases).set_index("caseid")
    manifest = pd.DataFrame(json.loads(args.vitaldb_manifest.read_text(encoding="utf-8"))["results"])
    manifest["caseid"] = pd.to_numeric(manifest.caseid, errors="coerce").astype("Int64")
    manifest["priority"] = manifest.tname.map({"Solar8000/ART_MBP": 0, "EV1000/ART_MBP": 1}).fillna(99)
    manifest = manifest.sort_values(["caseid", "priority", "tid"]).drop_duplicates("caseid").set_index("caseid")
    with zipfile.ZipFile(args.vitaldb_map_zip) as handle:
        members = {Path(name).name: name for name in handle.namelist() if not name.endswith("/")}
        for analysis_case, cohort_row in locked[locked.center.eq("VitalDB")].iterrows():
            caseid = int(analysis_case.removeprefix("V_"))
            track = manifest.loc[caseid]
            member = members.get(f"{caseid}_{track.tid}.csv")
            if member is None:
                raise RuntimeError(f"Missing VitalDB MAP for {analysis_case}")
            raw = pd.read_csv(handle.open(member))
            relative_time = pd.to_numeric(raw.iloc[:, 0], errors="coerce") - float(cases.loc[caseid].anestart)
            map_value = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
            row = {"analysis_case": analysis_case, "center": "VitalDB"}
            for threshold in [60, 65, 70]:
                result = features_from_points(
                    relative_time, map_value, float(cohort_row.anesthesia_minutes),
                    threshold=threshold, bin_seconds=60,
                )
                row.update(prefixed(result, f"t{threshold}_60s"))
            bridge = features_from_points(
                relative_time, map_value, float(cohort_row.anesthesia_minutes),
                threshold=65, bin_seconds=60, bridge_one_missing_bin=True,
            )
            row.update(prefixed(bridge, "t65_60s_bridge1missing"))
            high_resolution = features_from_points(
                relative_time, map_value, float(cohort_row.anesthesia_minutes),
                threshold=65, bin_seconds=10,
            )
            row.update(prefixed(high_resolution, "t65_10s"))
            rows.append(row)

    sensitivity = pd.DataFrame(rows).sort_values("analysis_case")
    if len(sensitivity) != len(combined) or sensitivity.analysis_case.duplicated().any():
        raise RuntimeError("Sensitivity feature row/key mismatch")
    check = combined[["analysis_case", "total_low_minutes", "excess5_minutes", "excess10_minutes"]].merge(
        sensitivity, on="analysis_case", validate="one_to_one"
    )
    exact_checks = {
        "total_low_minutes": bool(np.allclose(check.total_low_minutes, check.t65_60s_total_low_minutes, atol=1e-10)),
        "excess5_minutes": bool(np.allclose(check.excess5_minutes, check.t65_60s_excess5_minutes, atol=1e-10)),
        "excess10_minutes": bool(np.allclose(check.excess10_minutes, check.t65_60s_excess10_minutes, atol=1e-10)),
    }
    if not all(exact_checks.values()):
        raise RuntimeError("Primary 65-mmHg recomputation does not reproduce locked features: " + json.dumps(exact_checks))

    audit = {
        "purpose": "Prespecified sensitivity exposure construction on the locked cohort; no association model",
        "rows": int(len(sensitivity)),
        "center_rows": {key: int(value) for key, value in sensitivity.center.value_counts().items()},
        "primary_reproduction_checks": exact_checks,
        "definitions": {
            "thresholds_mmhg": [60, 65, 70],
            "primary_bin_seconds": 60,
            "bridge_rule": "bridge exactly one missing 60-second bin only when immediately flanked by low bins",
            "vitaldb_high_resolution_bin_seconds": 10,
            "missing_bins_otherwise_break_episodes": True,
        },
        "support": {},
    }
    for prefix in ["t60_60s", "t65_60s", "t70_60s", "t65_60s_bridge1missing", "t65_10s"]:
        available = sensitivity[sensitivity[f"{prefix}_total_low_minutes"].notna()] if f"{prefix}_total_low_minutes" in sensitivity else pd.DataFrame()
        if available.empty:
            continue
        audit["support"][prefix] = {
            "cases": int(len(available)),
            "any_hypotension": int(available[f"{prefix}_any_hypotension"].sum()),
            "longest_episode_gte_20": int(available[f"{prefix}_longest_episode_minutes"].ge(20).sum()),
            "median_coverage": float(available[f"{prefix}_map_coverage"].median()),
        }
    sensitivity.to_csv(args.out_dir / "SENSITIVITY_FEATURES_V4_1.csv", index=False)
    (args.out_dir / "SENSITIVITY_FEATURE_AUDIT_V4_1.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
