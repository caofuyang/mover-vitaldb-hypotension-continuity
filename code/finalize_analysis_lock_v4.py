"""Create the pre-model v4 analysis-lock manifest without fitting an outcome model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--mover-validation", type=Path, required=True)
    parser.add_argument("--mover-feature-audit", type=Path, required=True)
    parser.add_argument("--vitaldb-feature-audit", type=Path, required=True)
    parser.add_argument("--harmonization-audit", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--model-preflight", type=Path, required=True)
    parser.add_argument("--surgical-rules", type=Path, required=True)
    parser.add_argument("--flamerisk-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    validation = read_json(args.mover_validation)
    mover = read_json(args.mover_feature_audit)
    vital = read_json(args.vitaldb_feature_audit)
    harmonized = read_json(args.harmonization_audit)
    simulation = read_json(args.simulation)
    preflight = read_json(args.model_preflight)
    combined = pd.read_csv(args.combined)
    observed = combined[
        combined.outcome_observed_48h.astype(str).str.lower().isin(["true", "1", "yes"])
    ].copy()
    observed["early_aki_48h"] = pd.to_numeric(observed.early_aki_48h, errors="coerce")

    center_counts = {}
    for center, group in combined.groupby("center"):
        group_observed = observed[observed.center.eq(center)]
        center_counts[center] = {
            "parent_rows": int(len(group)),
            "outcome_observed_rows": int(len(group_observed)),
            "aki_events": int(group_observed.early_aki_48h.sum()),
            "longest_episode_gte_20": int(pd.to_numeric(group.longest_episode_minutes).ge(20).sum()),
        }

    checks = {
        "contract_status_locked": contract.get("status") == "locked_ready_for_final_v4_1_model_after_documented_technical_amendment",
        "mover_independent_validation_passed": validation.get("overall_status") == "pass",
        "mover_no_association_model_run": any(
            item.get("check") == "no_association_model_run" and item.get("status") == "pass"
            for item in validation.get("checks", [])
        ),
        "mover_feature_warnings_empty": not mover.get("warnings"),
        "vitaldb_raw_map_manifest_complete": bool(vital.get("raw_map_manifest", {}).get("complete")),
        "harmonization_consistency_zero": not any(harmonized.get("consistency", {}).values()),
        "combined_case_ids_unique": bool(combined.analysis_case.is_unique),
        "combined_outcomes_available": len(observed) == 2958 and int(observed.early_aki_48h.sum()) == 200,
        "twenty_minute_support_both_centers": all(
            values["longest_episode_gte_20"] >= 20 for values in center_counts.values()
        ),
        "full_episode_vectors_present": bool(combined.episode_durations_json.notna().all()),
        "synthetic_simulation_only": simulation.get("purpose", "").startswith("Prospective design simulation"),
        "model_matrix_preflight_passed": preflight.get("status") == "pass",
        "flamerisk_commit_pinned": (
            len(args.flamerisk_commit) == 40
            and contract.get("flexible_flame_analysis", {}).get("exact_commit_sha") == args.flamerisk_commit
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Analysis lock failed: " + ", ".join(failed))

    tracked = {
        "analysis_contract": args.contract,
        "combined_features": args.combined,
        "mover_validation": args.mover_validation,
        "mover_feature_audit": args.mover_feature_audit,
        "vitaldb_feature_audit": args.vitaldb_feature_audit,
        "harmonization_audit": args.harmonization_audit,
        "design_simulation": args.simulation,
        "model_preflight": args.model_preflight,
        "surgical_category_rules": args.surgical_rules,
        "primary_model_script": Path(__file__).with_name("fit_primary_model_v4.py"),
        "feature_harmonization_script": Path(__file__).with_name("harmonize_features_v4.py"),
        "mover_feature_script": Path(__file__).with_name("build_mover_features_v4.py"),
        "design_simulation_script": Path(__file__).with_name("design_simulation.py"),
        "model_preflight_script": Path(__file__).with_name("model_preflight_v4.py"),
    }
    manifest = {
        "status": "locked_ready_for_final_v4_1_model_after_documented_technical_amendment",
        "purpose": "v4.1 lock amendment after ICU coding correction limited to the observation-probability sensitivity model",
        "lock_date": "2026-09-23",
        "checks": checks,
        "all_hard_gates_passed": True,
        "prior_locked_v4_association_model_run": True,
        "amendment_changed_primary_outcome_exposure_or_covariates": False,
        "center_counts": center_counts,
        "combined_counts": {
            "parent_rows": int(len(combined)),
            "outcome_observed_rows": int(len(observed)),
            "aki_events": int(observed.early_aki_48h.sum()),
        },
        "flameRisk": {
            "repository": "https://github.com/xinkai-zhou/flameRisk",
            "commit": args.flamerisk_commit,
        },
        "sha256": {name: sha256(path) for name, path in tracked.items()},
    }
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
