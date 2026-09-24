"""Run the locked MOVER-VitalDB analysis from authorized local source exports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
CONTRACT = ROOT / "config" / "ANALYSIS_CONTRACT_v4.yaml"
RULES = ROOT / "config" / "SURGICAL_CATEGORY_RULES_v4.csv"


def run(script: str, *arguments: object, dry_run: bool = False) -> None:
    command = [sys.executable, str(CODE / script), *map(str, arguments)]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def required_path(value: object, label: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.example.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = settings["paths"]
    mover_archive = required_path(paths["mover_export_zip"], "MOVER export")
    vital_cases = required_path(paths["vitaldb_cases_csv"], "VitalDB cases")
    vital_labs = required_path(paths["vitaldb_labs_csv"], "VitalDB labs")
    vital_manifest = required_path(paths["vitaldb_manifest_json"], "VitalDB MAP manifest")
    vital_map = required_path(paths["vitaldb_map_zip"], "VitalDB MAP archive")
    output = Path(paths["output_directory"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output_directory; path already exists: {output}")
    if not args.dry_run:
        output.mkdir(parents=True)

    mover = output / "mover_features"
    vital = output / "vitaldb_features"
    combined = output / "combined_features"
    descriptive = output / "descriptive_results"
    sensitivity = output / "sensitivity_features"
    primary = output / "primary_model_results"
    sensitivity_results = output / "sensitivity_model_results"
    figures = output / "figures"
    simulation = output / "DESIGN_SIMULATION_RESULTS.json"
    preflight = output / "MODEL_PREFLIGHT_V4.json"
    lock = output / "ANALYSIS_LOCK_MANIFEST_V4_1.json"

    run("build_mover_features_v4.py", mover_archive, "--out-dir", mover, dry_run=args.dry_run)
    run(
        "build_vitaldb_features_v4.py", "--cases", vital_cases, "--labs", vital_labs,
        "--manifest", vital_manifest, "--map-zip", vital_map, "--out-dir", vital,
        dry_run=args.dry_run,
    )
    run(
        "harmonize_features_v4.py", "--mover", mover / "MOVER_FEATURES_V4.csv",
        "--vitaldb", vital / "VITALDB_FEATURES_V4.csv", "--out-dir", combined,
        dry_run=args.dry_run,
    )
    combined_csv = combined / "COMBINED_FEATURES_V4.csv"
    run("descriptive_observation_audit_v4.py", combined_csv, "--out-dir", descriptive, dry_run=args.dry_run)
    run("model_preflight_v4.py", combined_csv, "--out", preflight, dry_run=args.dry_run)
    run("design_simulation.py", "--combined", combined_csv, "--out", simulation, dry_run=args.dry_run)
    run(
        "exposure_support_audit_v4_1.py", combined_csv,
        "--out", output / "EXPOSURE_SUPPORT_AUDIT_V4_1.json", dry_run=args.dry_run,
    )
    run(
        "finalize_analysis_lock_v4.py", "--contract", CONTRACT, "--combined", combined_csv,
        "--mover-validation", mover / "MOVER_EXPORT_VALIDATION.json",
        "--mover-feature-audit", mover / "MOVER_FEATURE_AUDIT_V4.json",
        "--vitaldb-feature-audit", vital / "VITALDB_FEATURE_AUDIT_V4.json",
        "--harmonization-audit", combined / "HARMONIZATION_AUDIT_V4.json",
        "--simulation", simulation, "--model-preflight", preflight,
        "--surgical-rules", RULES, "--flamerisk-commit",
        settings["analysis"]["flameRisk_commit"], "--out", lock, dry_run=args.dry_run,
    )
    run(
        "fit_primary_model_v4.py", combined_csv, "--out-dir", primary,
        "--contract", CONTRACT, "--lock-manifest", lock, dry_run=args.dry_run,
    )
    run(
        "build_sensitivity_features_v4_1.py", "--combined", combined_csv,
        "--mover-archive", mover_archive, "--vitaldb-cases", vital_cases,
        "--vitaldb-manifest", vital_manifest, "--vitaldb-map-zip", vital_map,
        "--out-dir", sensitivity, dry_run=args.dry_run,
    )
    run(
        "fit_sensitivity_models_v4_1.py", "--combined", combined_csv,
        "--sensitivity", sensitivity / "SENSITIVITY_FEATURES_V4_1.csv",
        "--lock-manifest", lock, "--out-dir", sensitivity_results, dry_run=args.dry_run,
    )
    run(
        "standardized_risk_intervals_v4_1.py", combined_csv,
        "--out", primary / "STANDARDIZED_RISK_INTERVALS_V4_1.json", dry_run=args.dry_run,
    )
    run(
        "make_figures_v4_1.py", "--primary-report", primary / "PRIMARY_MODEL_REPORT.json",
        "--sensitivity-report", sensitivity_results / "SENSITIVITY_MODEL_REPORT_V4_1.json",
        "--risk-intervals", primary / "STANDARDIZED_RISK_INTERVALS_V4_1.json",
        "--out-dir", figures, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
