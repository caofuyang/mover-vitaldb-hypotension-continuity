"""Synthetic end-to-end test for mover_reexport_v2.py.

Creates a temporary miniature MOVER-shaped tree, runs the exporter, and checks
parent-cohort inclusion, deidentification, next-operation censoring, and the
Vital Signs / MAP (mmHg) source restriction. No real patient data are used.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from mover_reexport_v2 import build_export


def main(copy_archive_to: Path | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="mover_export_test_") as temp:
        base = Path(temp)
        root = base / "MOVER"
        emr = root / "data/EPIC_EMR/EMR"
        derived = root / "derived"
        emr.mkdir(parents=True)
        derived.mkdir(parents=True)

        info = pd.DataFrame(
            [
                ["RAW_CASE_A", "RAW_PATIENT_1", "54", "Male", "General", "3", "LAPAROTOMY, EXPLORATORY", "Inpatient", "Inpatient", "Y", "01/01/24 08:00", "01/01/24 10:00", "01/01/24 07:30", "01/01/24 10:30"],
                ["RAW_CASE_B", "RAW_PATIENT_2", "61", "Female", "General", "2", "VATS", "Inpatient", "Inpatient", "N", "01/02/24 08:00", "01/02/24 10:00", "01/02/24 07:30", "01/02/24 10:30"],
                # Later operation for patient 1 intentionally has no MAP and must
                # still define the censoring time for RAW_CASE_A.
                ["RAW_CASE_C", "RAW_PATIENT_1", "54", "Male", "General", "3", "MINOR PROCEDURE", "Outpatient", "Outpatient", "N", "01/03/24 08:00", "01/03/24 09:00", "01/03/24 07:30", "01/03/24 09:30"],
            ],
            columns=[
                "LOG_ID", "MRN", "BIRTH_DATE", "SEX", "PRIMARY_ANES_TYPE_NM", "ASA_RATING_C",
                "PRIMARY_PROCEDURE_NM", "PATIENT_CLASS_GROUP", "PATIENT_CLASS_NM", "ICU_ADMIN_FLAG",
                "AN_START_DATETIME", "AN_STOP_DATETIME", "IN_OR_DTTM", "OUT_OR_DTTM",
            ],
        )
        info.to_csv(emr / "patient_information.csv", index=False)

        labs = pd.DataFrame(
            [
                ["RAW_CASE_A", "RAW_PATIENT_1", "2160-0", "Creatinine", "1.0", "mg/dL", "12/31/23 08:00"],
                # Patient 1 has no postoperative value: must remain in parent cohort.
                ["RAW_CASE_B", "RAW_PATIENT_2", "2160-0", "Creatinine", "0.8", "mg/dL", "01/01/24 08:00"],
                ["RAW_CASE_B", "RAW_PATIENT_2", "2160-0", "Creatinine", "1.2", "mg/dL", "01/02/24 18:00"],
            ],
            columns=["LOG_ID", "MRN", "Lab Code", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"],
        )
        labs.to_csv(emr / "patient_labs.csv", index=False)
        pd.DataFrame(
            [["RAW_PATIENT_1", "I10", "Hypertension"]],
            columns=["mrn", "diagnosis_code", "dx_name"],
        ).to_csv(emr / "patient_history.csv", index=False)
        pd.DataFrame(
            [["RAW_CASE_B", "RAW_PATIENT_2", "E11", "Diabetes"]],
            columns=["LOG_ID", "mrn", "diagnosis_code", "dx_name"],
        ).to_csv(emr / "patient_visit.csv", index=False)

        rows = []
        for case, date in [("RAW_CASE_A", "2024-01-01"), ("RAW_CASE_B", "2024-01-02")]:
            for minute in range(120):
                stamp = f"{date} 08:{minute:02d}:00" if minute < 60 else f"{date} 09:{minute-60:02d}:00"
                value = 60.0 if 20 <= minute < 30 else 75.0
                rows.append([case, "Vital Signs", "MAP (mmHg)", stamp, value + 0.5, ""])
                rows.append([case, "Vitals", "Arterial Line MAP (ART)", stamp, value + 0.5, "mmHg"])
                # An empty-unit lookalike source must not enter the export.
                rows.append([case, "Vitals", "MAP-ART A-line", stamp, value, ""])
        map_data = pd.DataFrame(rows, columns=["LOG_ID", "FLO_NAME", "FLO_DISPLAY_NAME", "recorded_time", "map_value", "UNITS"])
        con = duckdb.connect()
        con.register("map_data", map_data)
        con.execute(f"COPY map_data TO '{(derived / 'epic_map_part1.parquet').as_posix()}' (FORMAT PARQUET)")
        paired = pd.DataFrame({
            "noninvasive_map": [60.0 + (i % 31) for i in range(2000)],
            "invasive_map": [61.0 + (i % 31) for i in range(2000)],
        })
        con.register("paired", paired)
        con.execute(f"COPY paired TO '{(derived / 'epic_nibp_nearest_invasive_map_pairs.parquet').as_posix()}' (FORMAT PARQUET)")
        con.close()

        out = base / "export"
        success = build_export(root, out)
        if not success:
            raise AssertionError((out / "audit_report.json").read_text(encoding="utf-8"))
        cohort = pd.read_csv(out / "cohort_parent.csv")
        exported_map = pd.read_csv(out / "map.csv")
        report = json.loads((out / "audit_report.json").read_text(encoding="utf-8"))

        assert len(cohort) == 2
        assert sorted(cohort.postoperative_creatinine_rows_48h.tolist()) == [0, 1]
        case_a = cohort.loc[cohort.procedure_name.eq("LAPAROTOMY, EXPLORATORY")].iloc[0]
        assert abs(case_a.next_operation_hours - 48.0) < 1e-9
        assert len(exported_map) == 240
        assert exported_map.source_context.eq("Devices Testing Template").all()
        assert exported_map.source_display_name.eq("MAP-ART A-line").all()
        assert exported_map.units.eq("mmHg").all()
        assert exported_map.unit_evidence.eq("prespecified paired NIBP/invasive scale validation").all()
        assert report["paired_scale_gate"]["passed"] is True
        assert report["status"] == "completed"
        assert report["association_model_run"] is False
        assert set(report["output_sha256"]) == {"README.txt", "cohort_parent.csv", "map.csv", "creatinine.csv"}
        for filename in ["cohort_parent.csv", "map.csv", "creatinine.csv"]:
            content = (out / filename).read_text(encoding="utf-8")
            assert "RAW_CASE" not in content
            assert "RAW_PATIENT" not in content
        if copy_archive_to is not None:
            copy_archive_to.write_bytes(out.with_suffix(".zip").read_bytes())
        print("PASS: synthetic MOVER re-export test")


if __name__ == "__main__":
    main()
