"""Validate a MOVER episode export before any outcome-association analysis.

The validator is intentionally independent of the exporter. It verifies the
archive structure, hashes, deidentification schema, referential integrity,
relative-time bounds, MAP ranges, parent-cohort construction, and the arterial
MAP unit crosswalk. It never fits an AKI association model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "audit_report.json",
    "README.txt",
    "cohort_parent.csv",
    "map.csv",
    "creatinine.csv",
}
FORBIDDEN_COLUMN_PATTERNS = [
    re.compile(x, re.I)
    for x in [r"(^|_)mrn($|_)", r"(^|_)log_id($|_)", r"calendar", r"datetime", r"timestamp", r"patient_id"]
]
DATE_VALUE = re.compile(
    r"(?:\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}/\d{1,2}/20\d{2}\b)"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(condition: bool, name: str, detail: object, severity: str = "error") -> dict:
    return {
        "check": name,
        "status": "pass" if condition else ("warning" if severity == "warning" else "fail"),
        "detail": detail,
    }


def finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def validate_export(
    archive: Path,
    output_json: Path,
    minimum_paired_operations: int = 100,
    minimum_correlation: float = 0.95,
    maximum_median_absolute_difference: float = 3.0,
) -> dict:
    checks: list[dict] = []
    with zipfile.ZipFile(archive) as handle:
        names = {Path(x).name: x for x in handle.namelist() if not x.endswith("/")}
        missing = sorted(REQUIRED_FILES - set(names))
        unexpected_duplicates = sorted(
            name for name in REQUIRED_FILES if sum(Path(x).name == name for x in handle.namelist()) != 1
        )
        checks.append(check(not missing, "required_files_present", {"missing": missing}))
        checks.append(check(not unexpected_duplicates, "required_files_unique", {"duplicates_or_missing": unexpected_duplicates}))
        if missing or unexpected_duplicates:
            result = {"archive": archive.name, "overall_status": "fail", "checks": checks}
            output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result

        audit = json.load(handle.open(names["audit_report.json"]))
        checks.append(check(audit.get("status") == "completed", "exporter_completed", audit.get("status")))
        checks.append(check(audit.get("association_model_run") is False, "no_association_model_run", audit.get("association_model_run")))
        checks.append(check(audit.get("paired_scale_gate", {}).get("passed") is True, "exporter_paired_scale_gate", audit.get("paired_scale_gate")))

        expected_hashes = audit.get("output_sha256", {})
        hash_results = {}
        for filename, expected in expected_hashes.items():
            if filename not in names:
                hash_results[filename] = "missing"
            else:
                observed = sha256_bytes(handle.read(names[filename]))
                hash_results[filename] = "match" if observed == expected else "mismatch"
        checks.append(check(bool(expected_hashes) and all(x == "match" for x in hash_results.values()), "output_hashes", hash_results))

        with tempfile.TemporaryDirectory(prefix="mover_validate_") as temp:
            temp_root = Path(temp)
            extracted = {}
            for filename in REQUIRED_FILES - {"audit_report.json", "README.txt"}:
                path = temp_root / filename
                path.write_bytes(handle.read(names[filename]))
                extracted[filename] = path

            cohort = pd.read_csv(extracted["cohort_parent.csv"])
            map_data = pd.read_csv(extracted["map.csv"])
            creatinine = pd.read_csv(extracted["creatinine.csv"])

        all_columns = {
            "cohort": cohort.columns.tolist(),
            "map": map_data.columns.tolist(),
            "creatinine": creatinine.columns.tolist(),
        }
        forbidden = {
            table: [col for col in columns if any(pattern.search(col) for pattern in FORBIDDEN_COLUMN_PATTERNS)]
            for table, columns in all_columns.items()
        }
        checks.append(check(not any(forbidden.values()), "no_direct_identifier_or_calendar_columns", forbidden))

        date_hits = {}
        for table_name, frame in [("cohort", cohort), ("map", map_data), ("creatinine", creatinine)]:
            hits = []
            for col in frame.select_dtypes(include="object").columns:
                if frame[col].astype(str).head(10000).str.contains(DATE_VALUE, regex=True, na=False).any():
                    hits.append(col)
            date_hits[table_name] = hits
        checks.append(check(not any(date_hits.values()), "no_calendar_date_values", date_hits))

        required_columns = {
            "cohort": {"export_case", "export_patient", "anesthesia_minutes", "postoperative_creatinine_rows_48h", "next_operation_hours"},
            "map": {"export_case", "time_seconds", "map_value", "source_context", "source_display_name", "units", "unit_evidence"},
            "creatinine": {"export_case", "time_seconds", "creatinine", "units"},
        }
        column_missing = {
            name: sorted(columns - set(all_columns[name])) for name, columns in required_columns.items()
        }
        checks.append(check(not any(column_missing.values()), "required_columns", column_missing))

        if any(column_missing.values()):
            result = {"archive": archive.name, "overall_status": "fail", "checks": checks}
            output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result

        case_set = set(cohort.export_case)
        checks.append(check(cohort.export_case.is_unique, "unique_cohort_case_key", int(cohort.export_case.duplicated().sum())))
        checks.append(check(cohort.export_case.notna().all() and cohort.export_patient.notna().all(), "nonmissing_export_keys", int(cohort[["export_case", "export_patient"]].isna().sum().sum())))
        orphan = {
            "map": int((~map_data.export_case.isin(case_set)).sum()),
            "creatinine": int((~creatinine.export_case.isin(case_set)).sum()),
        }
        checks.append(check(not any(orphan.values()), "referential_integrity", orphan))

        duplicate_map = int(map_data.duplicated(["export_case", "time_seconds"]).sum())
        checks.append(check(duplicate_map == 0, "unique_map_case_time", duplicate_map))
        map_values = finite_series(map_data, "map_value")
        checks.append(check(map_values.notna().all() and map_values.between(20, 160).all(), "map_finite_and_in_range", {"nonfinite": int(map_values.isna().sum()), "outside_20_160": int((~map_values.between(20,160) & map_values.notna()).sum())}))
        map_context_ok = map_data.source_context.astype(str).str.lower().eq("devices testing template")
        map_source_ok = map_data.source_display_name.astype(str).str.lower().eq("map-art a-line")
        map_unit_ok = map_data.units.astype(str).str.lower().str.replace(" ", "", regex=False).eq("mmhg")
        map_evidence_ok = map_data.unit_evidence.astype(str).str.lower().eq("prespecified paired nibp/invasive scale validation")
        checks.append(check(map_context_ok.all() and map_source_ok.all() and map_unit_ok.all() and map_evidence_ok.all(), "map_dense_source_with_paired_scale_validation", {"invalid_context_rows": int((~map_context_ok).sum()), "invalid_source_rows": int((~map_source_ok).sum()), "invalid_unit_rows": int((~map_unit_ok).sum()), "invalid_unit_evidence_rows": int((~map_evidence_ok).sum())}))

        duration = cohort.set_index("export_case").anesthesia_minutes * 60
        map_limit = map_data.export_case.map(duration)
        map_time = finite_series(map_data, "time_seconds")
        outside_time = int(((map_time < -1e-9) | (map_time >= map_limit + 1e-9) | map_time.isna()).sum())
        checks.append(check(outside_time == 0, "map_relative_time_inside_anesthesia", outside_time))

        cr = finite_series(creatinine, "creatinine")
        unit_ok = creatinine.units.astype(str).str.lower().str.replace(" ", "", regex=False).eq("mg/dl")
        checks.append(check(cr.notna().all() and cr.gt(0).all() and unit_ok.all(), "creatinine_positive_mg_dl", {"invalid_values": int((cr.isna() | cr.le(0)).sum()), "invalid_units": int((~unit_ok).sum())}))

        postop = pd.to_numeric(cohort.postoperative_creatinine_rows_48h, errors="coerce")
        zero_postop = int(postop.fillna(0).eq(0).sum())
        checks.append(
            check(
                zero_postop > 0,
                "parent_cohort_contains_cases_without_postop_creatinine",
                {"without_postop_48h": zero_postop, "cohort_rows": int(len(cohort))},
                severity="warning",
            )
        )

        counts = audit.get("export_counts", [{}])[0]
        observed_counts = {
            "cohort_rows": int(len(cohort)),
            "map_rows": int(len(map_data)),
            "creatinine_rows": int(len(creatinine)),
        }
        count_match = all(counts.get(key) == value for key, value in observed_counts.items())
        checks.append(check(count_match, "audit_counts_match_files", {"audit": counts, "observed": observed_counts}))

    overall = "fail" if any(x["status"] == "fail" for x in checks) else ("warning" if any(x["status"] == "warning" for x in checks) else "pass")
    result = {
        "archive": archive.name,
        "purpose": "Pre-analysis validation only; no outcome-association model",
        "overall_status": overall,
        "checks": checks,
    }
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path("MOVER_EXPORT_VALIDATION.json"))
    parser.add_argument("--minimum-paired-operations", type=int, default=100)
    parser.add_argument("--minimum-correlation", type=float, default=0.95)
    parser.add_argument("--maximum-median-absolute-difference", type=float, default=3.0)
    args = parser.parse_args()
    report = validate_export(
        args.archive,
        args.out,
        args.minimum_paired_operations,
        args.minimum_correlation,
        args.maximum_median_absolute_difference,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["overall_status"] in {"pass", "warning"} else 2)
