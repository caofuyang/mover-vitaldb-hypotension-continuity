"""End-to-end synthetic test of exporter followed by independent validator."""

from __future__ import annotations

import tempfile
from pathlib import Path

from test_mover_reexport_v2 import main as run_export_test
from validate_mover_export_v2 import validate_export


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mover_validator_test_") as temp:
        root = Path(temp)
        archive = root / "synthetic_export.zip"
        report_path = root / "validation.json"
        run_export_test(archive)
        report = validate_export(
            archive,
            report_path,
            minimum_paired_operations=2,
            minimum_correlation=0.95,
            maximum_median_absolute_difference=3.0,
        )
        assert report["overall_status"] in {"pass", "warning"}
        statuses = {item["check"]: item["status"] for item in report["checks"]}
        assert statuses["map_dense_source_with_paired_scale_validation"] == "pass"
        assert statuses["parent_cohort_contains_cases_without_postop_creatinine"] == "pass"
        assert statuses["no_direct_identifier_or_calendar_columns"] == "pass"
        print("PASS: synthetic export accepted by independent validator")


if __name__ == "__main__":
    main()
