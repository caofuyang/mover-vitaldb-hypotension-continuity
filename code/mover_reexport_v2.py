"""Build a deidentified MOVER export for episode-duration AKI research.

This script is read-only with respect to the MOVER source tree.  Unlike the
previous feasibility export, inclusion does NOT depend on postoperative
creatinine availability.  That permits an audit of outcome-observation
selection.  Output identifiers are newly generated sequential keys and all
times are relative to anesthesia start.

Expected input tree (MOVER EPIC release):
  data/EPIC_EMR/EMR/patient_information.csv
  data/EPIC_EMR/EMR/patient_labs.csv
  data/EPIC_EMR/EMR/patient_history.csv   (optional)
  data/EPIC_EMR/EMR/patient_visit.csv     (optional)
  derived/epic_map_part*.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import traceback
import zipfile
from pathlib import Path


SCRIPT_VERSION = "2026-09-23-mover-reexport-v5.1-paired-scale-validation"


def sqlstr(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sqlident(value: object) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def timestamp(column: str) -> str:
    """Parse only explicit date formats; never guess slash-date ordering."""
    x = f"trim(cast({column} as varchar))"
    two = "['%m/%d/%y %H:%M:%S', '%m/%d/%y %H:%M', '%m/%d/%y %I:%M:%S %p', '%m/%d/%y %I:%M %p', '%m/%d/%y']"
    four = "['%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %I:%M %p', '%m/%d/%Y']"
    return (
        f"CASE WHEN regexp_matches({x}, '^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{2}}($| )') "
        f"THEN try_strptime({x}, {two}) "
        f"WHEN regexp_matches({x}, '^[0-9]{{1,2}}/[0-9]{{1,2}}/[0-9]{{4}}($| )') "
        f"THEN try_strptime({x}, {four}) "
        f"WHEN regexp_matches({x}, '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}($|[ T])') "
        f"THEN try_cast({x} as timestamp) ELSE NULL END"
    )


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_export(root: Path, out: Path) -> bool:
    import duckdb

    out.mkdir(parents=True, exist_ok=False)
    report: dict = {
        "script_version": SCRIPT_VERSION,
        "status": "running",
        "purpose": "Parent-cohort export before postoperative-creatinine selection",
        "source_root": str(root),
        "limitations": [
            "The primary MOVER stream is MAP-ART A-line; its raw unit field is empty.",
            "Scale compatibility must pass a prespecified validation against the existing nearest-paired NIBP/invasive-MAP table before any export.",
            "A 60-second charted MAP value is an observed minute value, not proof of continuous low pressure for the whole minute.",
            "Procedure keyword flags are audit aids, not the final locked surgical taxonomy.",
            "No AKI outcome or association model is calculated by this script.",
        ],
    }
    db_path = out / "temporary_work.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory=" + sqlstr(out / "duckdb_temp"))

    def query(sql: str):
        cursor = con.execute(sql)
        names = [x[0] for x in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def checkpoint(message: str) -> None:
        print(message, flush=True)
        (out / "audit_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    try:
        emr = root / "data/EPIC_EMR/EMR"
        info = emr / "patient_information.csv"
        labs = emr / "patient_labs.csv"
        history = emr / "patient_history.csv"
        visit = emr / "patient_visit.csv"
        map_parts = sorted((root / "derived").glob("epic_map_part*.parquet"))
        paired_map = root / "derived/epic_nibp_nearest_invasive_map_pairs.parquet"
        required = [info, labs]
        missing = [str(p) for p in required if not p.exists()]
        if not map_parts:
            missing.append(str(root / "derived/epic_map_part*.parquet"))
        if not paired_map.exists():
            missing.append(str(paired_map))
        if missing:
            raise FileNotFoundError("Missing required input: " + "; ".join(missing))

        report["inputs"] = [
            {"path": str(p.relative_to(root)), "bytes": p.stat().st_size}
            for p in required + map_parts + [paired_map]
        ]
        report["optional_inputs"] = {
            "patient_history": history.exists(),
            "patient_visit": visit.exists(),
        }

        checkpoint("[1/7] Reading operation windows and demographics")
        con.execute(
            f"""
            CREATE TABLE info_raw AS
            SELECT trim(LOG_ID) log_id, trim(MRN) patient_id,
                   try_cast(BIRTH_DATE AS double) age,
                   trim(SEX) sex,
                   trim(PRIMARY_ANES_TYPE_NM) anesthesia,
                   try_cast(ASA_RATING_C AS double) asa,
                   trim(PRIMARY_PROCEDURE_NM) procedure_name,
                   trim(PATIENT_CLASS_GROUP) patient_class_group,
                   trim(PATIENT_CLASS_NM) patient_class_name,
                   trim(ICU_ADMIN_FLAG) icu_flag,
                   {timestamp('AN_START_DATETIME')} anesthesia_start,
                   {timestamp('AN_STOP_DATETIME')} anesthesia_end,
                   {timestamp('IN_OR_DTTM')} or_start,
                   {timestamp('OUT_OR_DTTM')} or_end
            FROM read_csv({sqlstr(info)}, header=true, all_varchar=true)
            """
        )
        con.execute(
            """
            CREATE TABLE operations AS
            SELECT log_id, min(patient_id) patient_id, min(age) age, min(sex) sex,
                   min(anesthesia) anesthesia, min(asa) asa,
                   min(procedure_name) procedure_name,
                   min(patient_class_group) patient_class_group,
                   min(patient_class_name) patient_class_name,
                   min(icu_flag) icu_flag,
                   min(anesthesia_start) anesthesia_start,
                   min(anesthesia_end) anesthesia_end,
                   min(or_start) or_start, min(or_end) or_end
            FROM info_raw
            WHERE log_id IS NOT NULL AND log_id<>''
            GROUP BY log_id
            HAVING count(*)=count(anesthesia_start)
               AND count(*)=count(anesthesia_end)
               AND count(distinct patient_id)=1
               AND count(distinct anesthesia_start)=1
               AND count(distinct anesthesia_end)=1
               AND min(anesthesia_end)>min(anesthesia_start)
            """
        )
        report["operation_audit"] = query(
            """SELECT count(*) operations, count(distinct patient_id) patients,
                      count(*) FILTER(WHERE patient_id IS NULL OR patient_id='') missing_patient,
                      median(epoch(anesthesia_end-anesthesia_start)/60.0) median_anesthesia_minutes
               FROM operations"""
        )

        checkpoint("[2/7] Reading derived arterial-MAP partitions")
        literal = "[" + ",".join(sqlstr(p) for p in map_parts) + "]"
        con.execute(
            f"""
            CREATE VIEW map_raw AS
            SELECT trim(cast(LOG_ID as varchar)) log_id,
                   trim(FLO_NAME) context,
                   trim(FLO_DISPLAY_NAME) display_name,
                   try_cast(recorded_time as timestamp) recorded_time,
                   try_cast(map_value as double) map_value,
                   trim(UNITS) units
            FROM read_parquet({literal}, union_by_name=true)
            """
        )
        report["map_source_counts"] = query(
            """SELECT context, display_name, units, count(*) row_count,
                      count(distinct log_id) operations
               FROM map_raw GROUP BY ALL ORDER BY row_count DESC"""
        )

        # Primary dense intraoperative arterial-MAP stream. Its raw unit field
        # is empty, so scale validation below is a hard prerequisite.
        con.execute(
            """
            CREATE TABLE map_points AS
            SELECT m.log_id, m.recorded_time,
                   median(m.map_value) map_value,
                   count(*) duplicate_rows,
                   max(m.map_value)-min(m.map_value) duplicate_spread
            FROM map_raw m JOIN operations o USING(log_id)
            WHERE lower(m.display_name)='map-art a-line'
              AND m.recorded_time>=o.anesthesia_start
              AND m.recorded_time<o.anesthesia_end
              AND isfinite(m.map_value) AND m.map_value BETWEEN 20 AND 160
            GROUP BY m.log_id, m.recorded_time
            """
        )

        report["primary_map_source"] = query(
            """SELECT count(*) map_rows, count(distinct log_id) operations,
                      min(map_value) minimum_map, max(map_value) maximum_map,
                      median(map_value) median_map
               FROM map_points"""
        )
        pair_columns = [row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sqlstr(paired_map)})"
        ).fetchall()]
        lower_columns = {name: name.lower().replace('-', '_') for name in pair_columns}
        nibp_candidates = [name for name, low in lower_columns.items()
                           if any(x in low for x in ('nibp', 'noninvasive', 'non_invasive'))
                           and any(x in low for x in ('map', 'mbp', 'mean', 'value'))]
        invasive_candidates = [name for name, low in lower_columns.items()
                               if not any(x in low for x in ('nibp', 'noninvasive', 'non_invasive'))
                               and any(x in low for x in ('invasive', 'inv_', 'art', 'aline', 'a_line'))
                               and any(x in low for x in ('map', 'mbp', 'mean', 'value'))]
        report["paired_map_schema"] = {
            "columns": pair_columns,
            "nibp_candidates": nibp_candidates,
            "invasive_candidates": invasive_candidates,
        }
        if len(nibp_candidates) != 1 or len(invasive_candidates) != 1:
            raise ValueError("Could not uniquely identify NIBP and invasive MAP columns in paired validation table")
        nibp_column = nibp_candidates[0]
        invasive_column = invasive_candidates[0]
        con.execute(
            f"""
            CREATE TABLE paired_scale AS
            SELECT try_cast({sqlident(nibp_column)} AS double) nibp_map,
                   try_cast({sqlident(invasive_column)} AS double) invasive_map
            FROM read_parquet({sqlstr(paired_map)})
            """
        )
        report["paired_scale_validation"] = query(
            """SELECT count(*) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) valid_pairs,
                      corr(nibp_map,invasive_map) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) correlation,
                      median(abs(nibp_map-invasive_map)) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) median_absolute_difference_mmhg,
                      median(invasive_map) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) /
                        median(nibp_map) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) median_ratio_invasive_to_nibp,
                      median(nibp_map) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) median_nibp,
                      median(invasive_map) FILTER(WHERE nibp_map BETWEEN 20 AND 160 AND invasive_map BETWEEN 20 AND 160) median_invasive
               FROM paired_scale"""
        )
        scale = report["paired_scale_validation"][0]
        scale_passed = bool(
            scale["valid_pairs"] >= 1000
            and scale["correlation"] is not None and scale["correlation"] >= 0.50
            and scale["median_absolute_difference_mmhg"] is not None and scale["median_absolute_difference_mmhg"] <= 15
            and scale["median_ratio_invasive_to_nibp"] is not None and 0.80 <= scale["median_ratio_invasive_to_nibp"] <= 1.20
        )
        report["paired_scale_gate"] = {
            "minimum_valid_pairs": 1000,
            "minimum_correlation": 0.50,
            "maximum_median_absolute_difference_mmhg": 15,
            "allowed_median_ratio_invasive_to_nibp": [0.80, 1.20],
            "passed": scale_passed,
        }
        if not scale_passed:
            raise ValueError("Prespecified NIBP/invasive MAP scale-validation gate failed")

        checkpoint("[3/7] Building the MAP-quality parent cohort without outcome selection")
        con.execute(
            """
            CREATE TABLE map_quality AS
            SELECT p.log_id,
                   count(distinct floor(epoch(p.recorded_time-o.anesthesia_start)/60.0)) occupied_minutes,
                   greatest(1,ceil(epoch(o.anesthesia_end-o.anesthesia_start)/60.0)) total_minutes,
                   count(*) unique_recorded_times,
                   median(p.map_value) median_map
            FROM map_points p JOIN operations o USING(log_id)
            GROUP BY p.log_id, o.anesthesia_start, o.anesthesia_end
            """
        )
        con.execute(
            """
            CREATE TABLE parent_candidates AS
            SELECT o.*, q.occupied_minutes, q.total_minutes, q.unique_recorded_times,
                   q.median_map, q.occupied_minutes/q.total_minutes coverage,
                   regexp_matches(upper(coalesce(o.procedure_name,'')),
                     '(CABG|CORONARY ARTERY BYPASS|AORTIC VALVE|MITRAL VALVE|TRICUSPID VALVE|PULMONIC VALVE|CARDIAC|HEART |HEART$|TRANSPLANT, HEART|VENTRICULAR ASSIST|(^|[^A-Z])VAD([^A-Z]|$)|ASCENDING AORTA|AORTIC ARCH|AORTIC ROOT|MAZE PROCEDURE|ELECTROPHYSIOLOGY|ARRHYTHMOGENIC|ATRIAL SEPTAL|(^|[^A-Z])ASD([^A-Z]|$)|PACEMAKER|DEFIBRILLATOR|TAVR|TRANSCATHETER AORTIC|STERNOTOMY)') cardiac_keyword_flag,
                   regexp_matches(lower(coalesce(o.procedure_name,'')),'transplant|donor hepatectomy|donor nephrectomy') transplant_keyword_flag
            FROM operations o JOIN map_quality q USING(log_id)
            WHERE o.age BETWEEN 18 AND 120
              AND lower(o.anesthesia)='general'
              AND o.asa BETWEEN 1 AND 4
              AND o.patient_id IS NOT NULL AND o.patient_id<>''
              AND o.procedure_name IS NOT NULL AND trim(o.procedure_name)<>''
              AND q.occupied_minutes>=30
              AND q.occupied_minutes/q.total_minutes>=0.70
            """
        )
        report["parent_cohort"] = query(
            """SELECT count(*) operations, count(distinct patient_id) patients,
                      count(*) FILTER(WHERE cardiac_keyword_flag) cardiac_keyword,
                      count(*) FILTER(WHERE transplant_keyword_flag) transplant_keyword
               FROM parent_candidates"""
        )

        checkpoint("[4/7] Reading creatinine across all encounters for each patient")
        con.execute(
            f"""
            CREATE TABLE creatinine_raw AS
            SELECT trim(MRN) patient_id,
                   trim(LOG_ID) source_log_id,
                   trim("Lab Code") lab_code,
                   trim("Lab Name") lab_name,
                   try_cast("Observation Value" as double) creatinine,
                   trim("Measurement Units") units,
                   {timestamp('"Collection Datetime"')} collection_time
            FROM read_csv({sqlstr(labs)}, header=true, all_varchar=true)
            WHERE trim("Lab Code")='2160-0'
            """
        )
        con.execute(
            """
            CREATE TABLE creatinine_clean AS
            SELECT * FROM creatinine_raw
            WHERE patient_id IS NOT NULL AND patient_id<>''
              AND collection_time IS NOT NULL
              AND isfinite(creatinine) AND creatinine>0 AND creatinine<>9999999
              AND lower(replace(units,' ',''))='mg/dl'
            """
        )
        report["creatinine_audit"] = query(
            """SELECT count(*) clean_rows, count(distinct patient_id) patients,
                      count(*)-count(distinct (patient_id,collection_time,creatinine)) exact_duplicate_rows,
                      count(*) FILTER(WHERE creatinine>=4) values_gte_4
               FROM creatinine_clean"""
        )

        checkpoint("[5/7] Adding patient history flags where available")
        if history.exists() or visit.exists():
            union_sql = []
            if history.exists():
                union_sql.append(
                    f"SELECT trim(mrn) patient_id, lower(coalesce(dx_name,'')) dx_name, lower(coalesce(diagnosis_code,'')) diagnosis_code FROM read_csv({sqlstr(history)},header=true,all_varchar=true)"
                )
            if visit.exists():
                union_sql.append(
                    f"SELECT trim(mrn) patient_id, lower(coalesce(dx_name,'')) dx_name, lower(coalesce(diagnosis_code,'')) diagnosis_code FROM read_csv({sqlstr(visit)},header=true,all_varchar=true)"
                )
            con.execute("CREATE TABLE diagnoses AS " + " UNION ALL ".join(union_sql))
            con.execute(
                """
                CREATE TABLE history_flags AS
                SELECT patient_id,
                       bool_or(regexp_matches(dx_name,'hypertension|hypertensive') OR regexp_matches(diagnosis_code,'^i10')) hypertension_flag,
                       bool_or(regexp_matches(dx_name,'diabetes') OR regexp_matches(diagnosis_code,'^e1[0-4]')) diabetes_flag,
                       bool_or(regexp_matches(dx_name,'end.stage renal|dialysis') OR regexp_matches(diagnosis_code,'^n18\\.?6')) esrd_history_flag
                FROM diagnoses WHERE patient_id IS NOT NULL AND patient_id<>'' GROUP BY patient_id
                """
            )
        else:
            con.execute(
                """CREATE TABLE history_flags(patient_id varchar, hypertension_flag boolean,
                                                diabetes_flag boolean, esrd_history_flag boolean)"""
            )

        checkpoint("[6/7] Generating deidentified relative-time exports")
        con.execute(
            """
            CREATE TABLE operation_next AS
            SELECT log_id,
                   epoch(lead(anesthesia_start) OVER(
                       PARTITION BY patient_id ORDER BY anesthesia_start,log_id
                   )-anesthesia_start)/3600.0 next_operation_hours
            FROM operations
            WHERE patient_id IS NOT NULL AND patient_id<>''
            """
        )
        con.execute(
            """
            CREATE TABLE case_key AS
            SELECT p.*,
                   dense_rank() OVER(ORDER BY patient_id) export_patient,
                   row_number() OVER(ORDER BY log_id) export_case,
                   row_number() OVER(PARTITION BY patient_id ORDER BY anesthesia_start,log_id) patient_operation_order,
                   epoch(anesthesia_start-min(anesthesia_start) OVER(PARTITION BY patient_id))/86400.0 days_from_first_exported_operation,
                   n.next_operation_hours
            FROM parent_candidates p JOIN operation_next n USING(log_id)
            """
        )
        con.execute(
            """
            CREATE TABLE export_cohort AS
            SELECT k.export_case, k.export_patient, k.patient_operation_order,
                   k.days_from_first_exported_operation, k.next_operation_hours,
                   k.age, k.sex, k.asa, k.anesthesia, k.procedure_name,
                   k.patient_class_group, k.patient_class_name, k.icu_flag,
                   epoch(k.anesthesia_end-k.anesthesia_start)/60.0 anesthesia_minutes,
                   k.occupied_minutes, k.total_minutes, k.coverage, k.median_map,
                   k.cardiac_keyword_flag, k.transplant_keyword_flag,
                   coalesce(h.hypertension_flag,false) hypertension_flag,
                   coalesce(h.diabetes_flag,false) diabetes_flag,
                   coalesce(h.esrd_history_flag,false) esrd_history_flag,
                   count(c.collection_time) FILTER(WHERE c.collection_time>=k.anesthesia_start-INTERVAL 30 DAY AND c.collection_time<k.anesthesia_start) baseline_creatinine_rows_30d,
                   count(c.collection_time) FILTER(WHERE c.collection_time>=k.anesthesia_end AND c.collection_time<=k.anesthesia_end+INTERVAL 48 HOUR) postoperative_creatinine_rows_48h,
                   count(c.collection_time) FILTER(WHERE c.collection_time>=k.anesthesia_end AND c.collection_time<=k.anesthesia_end+INTERVAL 7 DAY) postoperative_creatinine_rows_7d
            FROM case_key k LEFT JOIN history_flags h USING(patient_id)
            LEFT JOIN creatinine_clean c USING(patient_id)
            GROUP BY ALL
            """
        )
        con.execute(
            """
            CREATE TABLE export_map AS
            SELECT k.export_case,
                   epoch(p.recorded_time-k.anesthesia_start) time_seconds,
                   p.map_value, p.duplicate_rows, p.duplicate_spread,
                   'Devices Testing Template' source_context,
                   'MAP-ART A-line' source_display_name,
                   'mmHg' units,
                   'prespecified paired NIBP/invasive scale validation' unit_evidence
            FROM case_key k JOIN map_points p USING(log_id)
            """
        )
        con.execute(
            """
            CREATE TABLE export_creatinine AS
            SELECT k.export_case,
                   epoch(c.collection_time-k.anesthesia_start) time_seconds,
                   c.creatinine, c.units, c.lab_code,
                   CASE WHEN c.source_log_id=k.log_id THEN 'same_encounter' ELSE 'same_patient_other_encounter' END linkage
            FROM case_key k JOIN creatinine_clean c USING(patient_id)
            WHERE c.collection_time>=k.anesthesia_start-INTERVAL 30 DAY
              AND c.collection_time<=k.anesthesia_end+INTERVAL 7 DAY
            """
        )
        for table, filename in [
            ("export_cohort", "cohort_parent.csv"),
            ("export_map", "map.csv"),
            ("export_creatinine", "creatinine.csv"),
        ]:
            con.execute(f"COPY {table} TO {sqlstr(out / filename)} (HEADER, DELIMITER ',')")

        report["export_counts"] = query(
            """SELECT (SELECT count(*) FROM export_cohort) cohort_rows,
                      (SELECT count(*) FROM export_map) map_rows,
                      (SELECT count(*) FROM export_creatinine) creatinine_rows,
                      (SELECT count(*) FROM export_cohort WHERE baseline_creatinine_rows_30d>0) with_baseline_30d,
                      (SELECT count(*) FROM export_cohort WHERE postoperative_creatinine_rows_48h>0) with_postop_48h,
                      (SELECT count(*) FROM export_cohort WHERE baseline_creatinine_rows_30d>0 AND postoperative_creatinine_rows_48h>0) outcome_assessable_48h
            """
        )
        report["status"] = "completed"
        report["association_model_run"] = False
        checkpoint("[7/7] Export complete; no outcome or association model was run")
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        checkpoint("Export failed; see audit_report.json")
    finally:
        con.close()
        if db_path.exists():
            db_path.unlink()

    readme = """MOVER episode-duration research export v5

This export intentionally forms the MAP-quality parent cohort before requiring
postoperative creatinine. It contains only newly generated sequential case and
patient keys and relative times. It contains no original LOG_ID, MRN, or
calendar date. No AKI classification or association model is included.

The dense MAP-ART A-line stream is exported only after passing the prespecified
scale-validation gate against the existing nearest-paired NIBP/invasive MAP
table. Review paired_scale_validation and paired_scale_gate in audit_report.json.
Review audit_report.json before analysis.
"""
    (out / "README.txt").write_text(readme, encoding="utf-8")

    if report.get("status") == "completed":
        files = [
            "audit_report.json",
            "README.txt",
            "cohort_parent.csv",
            "map.csv",
            "creatinine.csv",
        ]
        report["output_sha256"] = {
            name: sha256(out / name) for name in files
            if name != "audit_report.json" and (out / name).exists()
        }
        (out / "audit_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in files:
                if (out / name).exists():
                    zf.write(out / name, name)
        print("EXPORT ZIP:", archive.resolve(), flush=True)
    return report.get("status") == "completed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Local MOVER data root")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    destination = args.out or Path.cwd() / (
        "MOVER_episode_export_v5_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    raise SystemExit(0 if build_export(args.root, destination) else 1)
