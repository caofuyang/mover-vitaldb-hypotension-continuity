"""Command-line entry point for corrected paired-scale MOVER export v5.1."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from mover_reexport_v2 import build_export


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Local MOVER data root")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    destination = args.out or Path.cwd() / (
        "MOVER_episode_export_v5_1_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    raise SystemExit(0 if build_export(args.root, destination) else 1)
