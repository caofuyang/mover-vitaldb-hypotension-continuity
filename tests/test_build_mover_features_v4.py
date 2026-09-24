"""Synthetic test for the prespecified MOVER feature builder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from build_mover_features_v4 import build_features
from test_mover_reexport_v2 import main as run_export_test


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mover_feature_test_") as temp:
        root = Path(temp)
        archive = root / "synthetic_export.zip"
        output = root / "features"
        run_export_test(archive)
        audit = build_features(archive, output, minimum_paired_operations=2)
        features = pd.read_csv(output / "MOVER_FEATURES_V4.csv")
        assert len(features) == 2
        assert features.export_patient.nunique() == 2
        assert sorted(features.outcome_observed_48h.tolist()) == [False, True]
        observed = features[features.outcome_observed_48h].iloc[0]
        assert bool(observed.early_aki_48h)
        assert set(features.total_low_minutes) == {10}
        assert set(features.longest_episode_minutes) == {10}
        assert set(features.excess5_minutes) == {5}
        assert audit["flow"][-1]["aki_events"] == 1
        print("PASS: synthetic MOVER feature construction")


if __name__ == "__main__":
    main()
