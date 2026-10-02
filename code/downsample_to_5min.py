"""Resolution control: resample the MOVER/VitalDB pressure grid to 5-minute charting.

Post hoc and exploratory: this script is NOT part of the locked analysis
contract in config/ANALYSIS_CONTRACT_v4.yaml.

Patients, covariates and outcomes are held fixed; only the temporal resolution
of the exposure changes. The 1-minute mean arterial pressure grid used by the
locked pipeline is subsampled to every fifth minute and summarized with the same
zero-order-hold logic applied to INSPIRE, so the two can be compared directly.

Usage
-----
python code/downsample_to_5min.py \
    --mover-features <MOVER_FEATURES_V4.csv> \
    --mover-map <MOVER export map.csv> \
    --vitaldb-features <VITALDB_FEATURES_V4.csv> \
    --vitaldb-cases <VitalDB clinical_data.csv> \
    --vitaldb-map-dir <VitalDB map_raw_download_v1 directory> \
    --vitaldb-manifest <VitalDB download_report.json> \
    --out-dir <output directory>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEP_MINUTES = 5
VALID_MIN, VALID_MAX = 20.0, 160.0


def grid_1min(time_seconds, value, anesthesia_minutes: float) -> np.ndarray:
    """Rebuild the same 60-second median grid used by the locked pipeline."""
    n = int(np.ceil(anesthesia_minutes))
    grid = np.full(n, np.nan)
    t = np.asarray(time_seconds, float)
    v = np.asarray(value, float)
    keep = (np.isfinite(t) & np.isfinite(v) & (t >= 0)
            & (t < anesthesia_minutes * 60) & (v >= VALID_MIN) & (v <= VALID_MAX))
    bins = np.floor(t[keep] / 60).astype(int)
    if len(bins):
        med = pd.Series(v[keep]).groupby(bins).median()
        idx = med.index.to_numpy(int)
        idx = idx[(idx >= 0) & (idx < n)]
        grid[idx] = med.loc[idx].to_numpy(float)
    return grid


def downsample_hold(grid: np.ndarray, step: int = STEP_MINUTES):
    """Keep every `step`-th minute, then carry each value forward."""
    minutes = np.arange(0, len(grid), step)
    values = grid[minutes]
    ok = np.isfinite(values)
    minutes, values = minutes[ok], values[ok]
    if len(minutes) < 5:
        return None
    ends = np.append(minutes[1:], minutes[-1] + step)
    duration = (ends - minutes).astype(float)
    low = values < 65.0
    total_low = float(duration[low].sum())
    episodes, i, lm = [], 0, low.astype(int)
    while i < len(lm):
        if lm[i]:
            j = i
            while j + 1 < len(lm) and lm[j + 1]:
                j += 1
            episodes.append(float(ends[j] - minutes[i]))
            i = j + 1
        else:
            i += 1
    depth = float((duration[low] * (65.0 - values[low])).sum())
    return {
        "ds_total": round(total_low, 1),
        "ds_episodes": len(episodes),
        "ds_longest": (max(episodes) if episodes else 0.0),
        "ds_excess5": round(float(sum(max(x - 5, 0) for x in episodes)), 1),
        "ds_depth": (depth / total_low if total_low else 0.0),
        "ds_readings": int(len(minutes)),
    }


def mover_frame(features: Path, mov_map: Path) -> pd.DataFrame:
    mover = pd.read_csv(features)
    need = set(mover.export_case)
    mp = pd.read_csv(mov_map, usecols=["export_case", "time_seconds", "map_value"])
    mp = mp[mp.export_case.isin(need)]
    anesthesia = mover.set_index("export_case").anesthesia_minutes.to_dict()
    rows = []
    for case, g in mp.groupby("export_case"):
        am = float(anesthesia.get(case, np.nan))
        if not np.isfinite(am):
            continue
        feat = downsample_hold(grid_1min(g.time_seconds.values, g.map_value.values, am))
        if feat:
            feat["export_case"] = case
            rows.append(feat)
    out = mover.merge(pd.DataFrame(rows), on="export_case", how="inner")
    out["center"] = "MOVER"
    out["analysis_case"] = "M_" + out.export_case.astype(str)
    return out


def vitaldb_frame(features: Path, cases: Path, map_dir: Path, manifest: Path) -> pd.DataFrame:
    vd = pd.read_csv(features)
    cd = pd.read_csv(cases)
    anestart = {int(k): v for k, v in zip(pd.to_numeric(cd.caseid, errors="coerce").dropna(), cd.anestart)}

    rep = json.loads(Path(manifest).read_text(encoding="utf-8"))
    man = pd.DataFrame(rep["results"])
    man["caseid"] = pd.to_numeric(man.caseid, errors="coerce").astype("Int64")
    man = man.sort_values(["caseid", "tid"]).drop_duplicates("caseid").set_index("caseid")

    rows = []
    for r in vd.itertuples():
        cid = int(r.vital_case)
        if cid not in man.index:
            continue
        f = Path(map_dir) / f"{cid}_{man.loc[cid].tid}.csv"
        if not f.exists():
            continue
        raw = pd.read_csv(f)
        if raw.shape[1] < 2:
            continue
        t = pd.to_numeric(raw.iloc[:, 0], errors="coerce") - float(anestart.get(cid, np.nan))
        feat = downsample_hold(
            grid_1min(t.values, pd.to_numeric(raw.iloc[:, 1], errors="coerce").values,
                      float(r.anesthesia_minutes)))
        if feat:
            feat["vital_case"] = cid
            rows.append(feat)
    out = vd.merge(pd.DataFrame(rows), on="vital_case", how="inner")
    out["center"] = "VitalDB"
    out["analysis_case"] = "V_" + out.vital_case.astype(str)
    return out


KEEP = ["center", "analysis_case", "age", "sex", "asa", "anesthesia_minutes",
        "surgical_category", "baseline_egfr", "ds_total", "ds_episodes", "ds_longest",
        "ds_excess5", "ds_depth", "ds_readings", "outcome_observed_48h", "early_aki_48h"]


def build(args) -> pd.DataFrame:
    mover = mover_frame(args.mover_features, args.mover_map)
    vital = vitaldb_frame(args.vitaldb_features, args.vitaldb_cases,
                          args.vitaldb_map_dir, args.vitaldb_manifest)
    combined = pd.concat([mover[KEEP], vital[KEEP]], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_dir / "downsampled_5min_features.csv", index=False)
    print(f"MOVER {len(mover)}  VitalDB {len(vital)}  combined {len(combined)}")
    return combined


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mover-features", type=Path, required=True)
    p.add_argument("--mover-map", type=Path, required=True)
    p.add_argument("--vitaldb-features", type=Path, required=True)
    p.add_argument("--vitaldb-cases", type=Path, required=True)
    p.add_argument("--vitaldb-map-dir", type=Path, required=True)
    p.add_argument("--vitaldb-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    build(p.parse_args())
