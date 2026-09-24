"""Verify package integrity and the manuscript's numerical results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Expected {expected}, found {actual}")


manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
for relative, expected in manifest["sha256"].items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise AssertionError(f"Checksum mismatch: {relative}")

expected_results = ROOT / "expected_results"
results = json.loads((expected_results / "KEY_RESULTS.json").read_text(encoding="utf-8"))
scenario = results["scenarios"]["pooled_primary"]
close(results["likelihood_ratio_test_excess5"]["p_value"], 0.4418126401941287)
close(scenario["odds_ratio"], 1.0869586811944927)
close(scenario["standardized_reference_risk"], 0.06453146280402511)
close(scenario["standardized_comparison_risk"], 0.06928633069330702)
if results["models"]["pooled_primary"]["nobs"] != 2958:
    raise AssertionError("Unexpected primary analytic cohort size")

# Standardized-risk intervals must agree with the archived primary report.
intervals = json.loads((expected_results / "STANDARDIZED_RISK_INTERVALS_V4_1.json").read_text(encoding="utf-8"))
primary = intervals["pooled_primary"]
close(primary["four_by_five_risk"], scenario["standardized_reference_risk"])
close(primary["one_by_twenty_risk"], scenario["standardized_comparison_risk"])
close(primary["odds_ratio"], scenario["odds_ratio"])
close(primary["risk_difference_95ci"][0], scenario["standardized_risk_difference_95ci"][0])
close(primary["risk_difference_95ci"][1], scenario["standardized_risk_difference_95ci"][1])

# The released descriptive table must match manuscript Table 1, which describes
# the outcome-observed cohort.
with (expected_results / "TABLE1_DESCRIPTIVE_OBSERVED_COHORT.csv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))


def cell(group: str, variable: str) -> dict:
    for row in rows:
        if row["group"] == group and row["variable"] == variable and not row["level"]:
            return row
    raise AssertionError(f"Missing descriptive row: {group}/{variable}")


def medians(group: str, variable: str) -> tuple[float, float, float]:
    row = cell(group, variable)
    return float(row["median"]), float(row["q1"]), float(row["q3"])


close(medians("Overall", "total_low_minutes")[0], 12.0)
close(medians("MOVER", "episode_count")[0], 6.0)
close(medians("VitalDB", "anesthesia_minutes")[0], 225.0)
if int(cell("Overall", "age")["n"]) != 2958:
    raise AssertionError("Observed descriptive table is not the analytic cohort")

# Exposure support for the locked contrast.
support = json.loads((expected_results / "EXPOSURE_SUPPORT_AUDIT_V4_1.json").read_text(encoding="utf-8"))
observed_support = support["outcome_observed_cohort"]["overall"]
if observed_support["single_episode_of_20_minutes_or_more"] != 6:
    raise AssertionError("Unexpected number of single 20-minute episodes")
if observed_support["exactly_four_five_minute_episodes"] != 0:
    raise AssertionError("Unexpected number of exact four-by-five patterns")
close(support["correlation_total_duration_and_pattern_term"]["pooled"], 0.8802074789756332)

# Design-simulation operating characteristics.
design = json.loads((expected_results / "DESIGN_SIMULATION_RESULTS.json").read_text(encoding="utf-8"))
pooled = {
    item["scenario_or_1x20_vs_4x5"]: item["two_sided_rejection_probability"]
    for item in design["results"]
    if item["dataset"] == "Pooled_with_center_intercept"
}
close(pooled[1.0], 0.05, 2e-3)
close(pooled[1.3], 0.757, 2e-3)
close(pooled[1.5], 0.97, 2e-3)

print("Release integrity and manuscript-result checks passed.")
