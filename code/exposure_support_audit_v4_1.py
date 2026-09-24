"""Exposure support for the prespecified 1x20-versus-4x5 contrast.

Reports how many patients and episodes actually occupy the duration patterns
that the locked contrast compares. This is a descripton of exposure only; no
outcome association is fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def observed_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def support(frame: pd.DataFrame) -> dict:
    runs = [json.loads(value) for value in frame.episode_durations_json]
    total = pd.to_numeric(frame.total_low_minutes, errors="coerce").to_numpy(float)
    excess = pd.to_numeric(frame.excess5_minutes, errors="coerce").to_numpy(float)
    longest = pd.to_numeric(frame.longest_episode_minutes, errors="coerce").to_numpy(float)
    episodes = pd.to_numeric(frame.episode_count, errors="coerce").to_numpy(float)
    return {
        "cases": int(len(frame)),
        "cases_with_any_hypotension": int((total > 0).sum()),
        "total_low_minutes_gte_20": int((total >= 20).sum()),
        "excess5_minutes_gte_15": int((excess >= 15).sum()),
        "longest_episode_gte_20": int((longest >= 20).sum()),
        "single_episode_of_20_minutes_or_more": int(((episodes == 1) & (longest >= 20)).sum()),
        "exactly_four_five_minute_episodes": int(sum(1 for run in runs if sorted(run) == [5, 5, 5, 5])),
        "total_15_to_25_minutes_with_zero_excess": int(
            ((total >= 15) & (total <= 25) & (excess == 0)).sum()),
        "excess5_minutes_quantiles_50_75_90_95": [
            float(np.percentile(excess, p)) for p in (50, 75, 90, 95)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    observed = data[observed_flag(data.outcome_observed_48h)].copy()
    report = {
        "purpose": "Exposure support for the locked one-degree-of-freedom duration-pattern contrast",
        "note": "Support describes the exposure only. The contrast itself is model-based.",
        "parent_cohort": {
            "overall": support(data),
            **{center: support(group) for center, group in data.groupby("center")},
        },
        "outcome_observed_cohort": {
            "overall": support(observed),
            **{center: support(group) for center, group in observed.groupby("center")},
        },
        "correlation_total_duration_and_pattern_term": {
            "pooled": float(np.corrcoef(
                pd.to_numeric(observed.total_low_minutes, errors="coerce"),
                pd.to_numeric(observed.excess5_minutes, errors="coerce"))[0, 1]),
            **{
                center: float(np.corrcoef(
                    pd.to_numeric(group.total_low_minutes, errors="coerce"),
                    pd.to_numeric(group.excess5_minutes, errors="coerce"))[0, 1])
                for center, group in observed.groupby("center")
            },
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
