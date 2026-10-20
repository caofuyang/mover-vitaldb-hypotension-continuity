"""INSPIRE carry-forward interval audit and sensitivity analysis (2026-10-18).

The primary INSPIRE exposure carries each charted reading forward to the next
reading, and the last reading to the end of anaesthesia, with no maximum
interval. This script

1. describes the distribution of intervals between consecutive valid readings
   (including the first and last interval relative to the anaesthesia window);
2. rebuilds the exposure with a maximum carry-forward interval of 15 or 30
   minutes, so that a sparse stretch cannot create an arbitrarily long
   hypotensive episode;
3. writes both feature tables so that the model ladder can be re-run.

python inspire_gap_sensitivity.py --features INSPIRE_FEATURES_V2.csv \
    --inspire-dir /path/to/INSPIRE --out-dir gaps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LO, HI, THRESHOLD = 20.0, 160.0, 65.0


def hold_features(times, values, anstart, anend, cap=None):
    t = np.asarray(times, float)
    v = np.asarray(values, float)
    ok = (np.isfinite(t) & np.isfinite(v) & (t >= anstart) & (t <= anend)
          & (v >= LO) & (v <= HI))
    t, v = t[ok], v[ok]
    if len(t) == 0:
        return None, []
    order = np.argsort(t)
    t, v = t[order], v[order]
    ends = np.append(t[1:], anend)
    gaps = ends - t
    if cap is not None:
        ends = t + np.minimum(gaps, cap)
        gaps_used = np.minimum(gaps, cap)
    else:
        gaps_used = gaps
    low = v < THRESHOLD
    total_low = float(gaps_used[low].sum())
    episodes, i, lm = [], 0, low.astype(int)
    while i < len(lm):
        if lm[i]:
            j = i
            # A capped hold leaves uncovered time between readings. Consecutive
            # low readings only form one episode when the next reading starts at
            # or before the end of the interval covered by the previous one.
            while (j + 1 < len(lm) and lm[j + 1]
                   and t[j + 1] <= ends[j] + 1e-9):
                j += 1
            episodes.append(float(ends[j] - t[i]))
            i = j + 1
        else:
            i += 1
    depth = float((gaps_used[low] * (THRESHOLD - v[low])).sum())
    feat = {
        "total_low_minutes": total_low,
        "episode_count": int(len(episodes)),
        "longest_episode_minutes": float(max(episodes)) if episodes else 0.0,
        "excess5_minutes": float(sum(max(x - 5.0, 0.0) for x in episodes)),
        "auc_below_65_mmhg_minutes": depth,
        "mean_depth_below_65_mmhg": (depth / total_low) if total_low > 0 else 0.0,
        "any_hypotension": int(total_low > 0),
        "n_readings": int(len(t)),
        "max_gap_minutes": float(gaps.max()),
        "share_time_in_gaps_over_15": float(gaps[gaps > 15].sum() / gaps.sum()) if gaps.sum() else 0.0,
    }
    return feat, gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--inspire-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(a.features)
    ids = set(pd.to_numeric(feat.inspire_case, errors="coerce").dropna().astype(int))
    print("cohort rows:", len(feat))

    parts = []
    for chunk in pd.read_csv(a.inspire_dir / "vitals.csv.gz",
                             usecols=["op_id", "item_name", "chart_time", "value"],
                             chunksize=3_000_000):
        sel = chunk[(chunk.item_name == "art_mbp") & chunk.op_id.isin(ids)]
        if len(sel):
            parts.append(sel[["op_id", "chart_time", "value"]])
    art = pd.concat(parts, ignore_index=True)
    print("art_mbp rows for cohort:", len(art))
    by_op = {k: (g.chart_time.to_numpy(float), g.value.to_numpy(float))
             for k, g in art.groupby("op_id")}

    all_gaps, rows_uncapped, rows_15, rows_30 = [], [], [], []
    for r in feat.itertuples():
        op = int(r.inspire_case)
        got = by_op.get(op)
        if got is None:
            rows_uncapped.append({}); rows_15.append({}); rows_30.append({})
            continue
        f0, gaps = hold_features(got[0], got[1], float(r.anstart_time), float(r.anend_time))
        f15, _ = hold_features(got[0], got[1], float(r.anstart_time), float(r.anend_time), cap=15)
        f30, _ = hold_features(got[0], got[1], float(r.anstart_time), float(r.anend_time), cap=30)
        rows_uncapped.append(f0 or {}); rows_15.append(f15 or {}); rows_30.append(f30 or {})
        if len(gaps):
            all_gaps.append(gaps)

    gaps = np.concatenate(all_gaps) if all_gaps else np.array([])
    stats = {
        "readings": int(len(art)),
        "intervals": int(len(gaps)),
        "interval_minutes": {
            "median": round(float(np.median(gaps)), 2),
            "p75": round(float(np.percentile(gaps, 75)), 2),
            "p90": round(float(np.percentile(gaps, 90)), 2),
            "p95": round(float(np.percentile(gaps, 95)), 2),
            "p99": round(float(np.percentile(gaps, 99)), 2),
            "max": round(float(gaps.max()), 1)},
        "intervals_over_15_min": int((gaps > 15).sum()),
        "intervals_over_30_min": int((gaps > 30).sum()),
        "share_of_intervals_at_most_10_min": round(float((gaps <= 10).mean()), 6),
        "share_of_intervals_over_15_min": round(float((gaps > 15).mean()), 8),
        "share_of_held_time_in_intervals_over_15_min": round(float(gaps[gaps > 15].sum() / gaps.sum()), 8),
        "share_of_held_time_in_intervals_over_30_min": round(float(gaps[gaps > 30].sum() / gaps.sum()), 8),
    }
    print(json.dumps(stats, indent=2))

    for label, rows in [("uncapped", rows_uncapped), ("cap15", rows_15), ("cap30", rows_30)]:
        new = pd.DataFrame(rows, index=feat.index)
        out = feat.copy()
        for c in ["total_low_minutes", "episode_count", "longest_episode_minutes",
                  "excess5_minutes", "auc_below_65_mmhg_minutes", "mean_depth_below_65_mmhg",
                  "any_hypotension", "n_readings", "max_gap_minutes", "share_time_in_gaps_over_15"]:
            out[c] = new[c] if c in new.columns else np.nan
        out.to_csv(a.out_dir / f"INSPIRE_FEATURES_{label}.csv", index=False)
        print(label, "mean total low", round(float(out.total_low_minutes.mean()), 2))

    (a.out_dir / "INSPIRE_GAP_AUDIT.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
