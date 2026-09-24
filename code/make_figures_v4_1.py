"""Create manuscript-ready vector figures from locked v4.1 result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np

# Keep text as real text (not outlined paths) so the vector figures stay
# editable in Illustrator, Inkscape, or Word, and embed TrueType in PDF.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_all(fig, out_dir: Path, stem: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def flow_figure(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.50, 0.92, "Exported parent cohorts\nMOVER: 3,672 operations\nVitalDB: 6,388 operations"),
        (0.50, 0.76, "Eligibility and MAP quality\nNoncardiac, nontransplant, adult general anesthesia\nMAP coverage ≥70% and ≥30 observed minutes"),
        (0.50, 0.60, "Valid baseline kidney function\nMOVER: 1,456 operations\nVitalDB: 2,014 operations"),
        (0.50, 0.44, "One prespecified operation per patient\nMOVER: 1,178 patients\nVitalDB: 1,960 patients"),
        (0.50, 0.28, "48-hour creatinine outcome observed\nMOVER: 1,137 (132 AKI)\nVitalDB: 1,821 (68 AKI)"),
        (0.50, 0.12, "Pooled primary analysis\n2,958 patients; 200 AKI events"),
    ]
    width, height = 0.78, 0.105
    for x, y, label in boxes:
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=1.2, edgecolor="#274C77", facecolor="#F3F7FA",
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, linespacing=1.3)
    for upper, lower in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(0.5, lower[1] + height / 2), xytext=(0.5, upper[1] - height / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#274C77", lw=1.2))
    ax.set_title("Study cohort flow", fontsize=15, fontweight="bold", pad=18)
    save_all(fig, out_dir, "Figure_1_cohort_flow")


def forest_figure(primary: dict, sensitivity: dict, out_dir: Path) -> None:
    p = primary["scenarios"]
    entries = [
        ("Primary pooled analysis", p["pooled_primary"]),
        ("Observation-weighted", p["pooled_ipow"]),
        ("Common surgical support", p["pooled_common_surgical_support"]),
        ("MOVER", p["center_MOVER"]),
        ("VitalDB", p["center_VitalDB"]),
    ]
    lookup = {item["name"]: item["scenario"] for item in sensitivity["results"]}
    entries += [
        ("Threshold 60 mmHg", lookup["threshold_60"]),
        ("Threshold 70 mmHg", lookup["threshold_70"]),
        ("Bridge one missing minute", lookup["bridge_one_missing_minute"]),
        ("10-minute hinge", lookup["hinge_10_minutes"]),
        ("VitalDB, 10-second bins", lookup["vitaldb_10_second"]),
    ]
    labels = [item[0] for item in entries]
    odds = np.asarray([item[1]["odds_ratio"] for item in entries])
    low = np.asarray([item[1]["odds_ratio_95ci"][0] for item in entries])
    high = np.asarray([item[1]["odds_ratio_95ci"][1] for item in entries])
    y = np.arange(len(entries))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    ax.errorbar(odds, y, xerr=np.vstack([odds - low, high - odds]), fmt="o", color="#274C77",
                ecolor="#6096BA", elinewidth=1.8, capsize=3, markersize=6)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.45, 2.25)
    ticks = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.50", "0.75", "1.00", "1.25", "1.50", "2.00"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, labels)
    ax.set_xlabel("Odds ratio for one 20-min episode vs four 5-min episodes")
    ax.set_title("Duration-pattern association with early postoperative AKI", fontsize=14, fontweight="bold", pad=16)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    for yi, estimate, lower, upper in zip(y, odds, low, high):
        ax.text(2.20, yi, f"{estimate:.2f} ({lower:.2f}–{upper:.2f})", ha="right", va="center", fontsize=8.5)
    ax.text(0.98, 1.01, "OR (95% CI)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.31, right=0.98, top=0.89)
    save_all(fig, out_dir, "Figure_2_forest_duration_pattern")


def risk_figure(primary: dict, intervals: dict, out_dir: Path) -> None:
    names = ["Primary", "Observation-weighted", "Common support", "MOVER", "VitalDB"]
    keys = ["pooled_primary", "pooled_ipow", "pooled_common_surgical_support",
            "center_MOVER", "center_VitalDB"]
    reference = np.asarray([intervals[key]["four_by_five_risk"] * 100 for key in keys])
    comparison = np.asarray([intervals[key]["one_by_twenty_risk"] * 100 for key in keys])
    reference_low = np.asarray([intervals[key]["four_by_five_risk_95ci"][0] * 100 for key in keys])
    reference_high = np.asarray([intervals[key]["four_by_five_risk_95ci"][1] * 100 for key in keys])
    comparison_low = np.asarray([intervals[key]["one_by_twenty_risk_95ci"][0] * 100 for key in keys])
    comparison_high = np.asarray([intervals[key]["one_by_twenty_risk_95ci"][1] * 100 for key in keys])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    width = 0.36
    ax.bar(
        x - width / 2, reference, width, label="Four 5-min episodes", color="#A3CEF1",
        yerr=np.vstack([reference - reference_low, reference_high - reference]),
        error_kw=dict(ecolor="#3B3B3B", elinewidth=1.2, capsize=4),
    )
    ax.bar(
        x + width / 2, comparison, width, label="One 20-min episode", color="#274C77",
        yerr=np.vstack([comparison - comparison_low, comparison_high - comparison]),
        error_kw=dict(ecolor="#3B3B3B", elinewidth=1.2, capsize=4),
    )
    ax.set_ylabel("Standardized AKI risk (%)")
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0, max(comparison_high.max(), reference_high.max()) * 1.32)
    ax.set_title("Standardized 48-hour AKI risk under the primary contrast", fontweight="bold", pad=30)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    save_all(fig, out_dir, "Figure_3_standardized_risk")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--sensitivity-report", type=Path, required=True)
    parser.add_argument("--risk-intervals", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    primary = read_json(args.primary_report)
    sensitivity = read_json(args.sensitivity_report)
    intervals = read_json(args.risk_intervals)
    flow_figure(args.out_dir)
    forest_figure(primary, sensitivity, args.out_dir)
    risk_figure(primary, intervals, args.out_dir)
    print(json.dumps({"status": "completed", "figures": 3, "formats": ["png", "pdf", "svg"]}, indent=2))


if __name__ == "__main__":
    main()
