"""Figure 2 v2: standardised 48-hour AKI risk by cumulative hypotension.

Reads the numbers computed by analysis_clinical_v2.py so that the figure and the
text use one model, one spline specification and one standardisation.

python make_figure_absolute_risk_v2.py --clinical clinical_v2.json --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WAVEFORM, CHARTED = "#1f4e79", "#b3541e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    d = json.loads(a.clinical.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.85, 4.0))  # 174 mm wide

    for key, label, colour, ls in [
        ("pooled_waveform_1min", "MOVER + VitalDB (waveform, 1-minute)", WAVEFORM, "-"),
        ("inspire", "INSPIRE (charted ~5-minute)", CHARTED, "-"),
    ]:
        c = d["curves"][key]
        g = np.array(c["grid"], float)
        risk = np.array(c["risk"], float) * 100
        lo = np.array(c["lo"], float) * 100
        hi = np.array(c["hi"], float) * 100
        ax.fill_between(g, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(g, risk, color=colour, lw=2.0, ls=ls, label=label)

    ax.set_xlim(0, 120)
    # Upper limit taken from the data so the confidence band is never clipped.
    ymax = max(max(np.array(d["curves"][k]["hi"], float)) for k in
               ("pooled_waveform_1min", "inspire")) * 100
    ylim = 5 * np.ceil(ymax / 5)
    ax.set_ylim(0, ylim)
    ax.set_xticks([0, 30, 60, 90, 120])
    ax.set_yticks(np.arange(0, ylim + 1, 5))
    ax.set_xlabel("Cumulative minutes with mean arterial pressure below 65 mmHg", fontsize=8.5)
    ax.set_ylabel("Standardised 48-hour AKI risk (%)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 600), ("pdf", None), ("eps", None)]:
        fig.savefig(a.out_dir / f"Fig2.{ext}", dpi=dpi, bbox_inches="tight")
    print("written", a.out_dir / "Fig2.*")
    for key in ["pooled_waveform_1min", "inspire"]:
        c = d["curves"][key]
        print(key, "risk at 0/45/60/90/120:",
              [round(x * 100, 1) for x in c["risk_at_key"]])


if __name__ == "__main__":
    main()
