"""Build Table 1-4 and Supplementary Tables S1-S2 from the current results.

Every number is computed from the feature tables and the result JSON files, so a
table cannot drift from the analysis that produced it.

Example
-------
python build_tables.py \
    --waveform  ../数据表/COMBINED_FEATURES_V4.csv \
    --resampled ../重建_20261018/ds_v2/downsampled_5min_features.csv \
    --inspire   ../重建_20261019/inspire_v3/INSPIRE_FEATURES_HOLD.csv \
    --ladder-wave      ../重建_20261018/ladder_1min_v2.json \
    --ladder-resampled ../重建_20261018/ladder_5min_v2.json \
    --ladder-inspire   ../重建_20261019/inspire_v3/inspire_v3_ladder.json \
    --clinical  ../重建_20261019/inspire_v3/clinical_v3.json \
    --sensitivity-s1 ../重建_20261019/SENSITIVITY_S1.json \
    --out-dir   ../重建_20261019/tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Emu, Pt

FONT = "Times New Roman"
SIZE = Pt(9)
BANDS = ["0", "1-15", "16-30", "31-60", ">60"]
CATEGORIES = [
    ("abdominal", "abdominal", "abdominal"),
    ("thoracic", "thoracic", "thoracic"),
    ("vascular", "vascular", "vascular"),
    ("genitourinary_gynecologic", "genitourinary gynecologic", "genitourinary_gynecologic"),
    ("superficial_other", "superficial other", "superficial_other"),
    ("out_of_common_support", "out of common support", "out_of_support"),
]


def _twips(w: int) -> Emu:
    return Emu(int(w * 635))


def new_doc() -> Document:
    d = Document()
    st = d.styles["Normal"]
    st.font.name = FONT
    st.font.size = SIZE
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    return d


def caption(d: Document, text: str) -> None:
    p = d.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = SIZE
    r.font.name = FONT


def note(d: Document, text: str) -> None:
    p = d.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(8)
    r.font.name = FONT
    p.paragraph_format.space_after = Pt(6)


def table(d: Document, rows, widths, header_rows: int = 1, size=SIZE):
    # Fit inside an A4 text column (16 cm = 9,072 twips) and write the widths to
    # the table grid, not only to the cells: Word and LibreOffice both lay the
    # table out from tblGrid.
    sec = d.sections[0]
    avail_twips = int((sec.page_width - sec.left_margin - sec.right_margin) / 635)
    scale = min(1.0, avail_twips / sum(widths))
    widths = [int(w * scale) for w in widths]
    t = d.add_table(rows=len(rows), cols=len(widths))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for gc, w in zip(t._tbl.tblGrid.gridCol_lst, widths):
        gc.set(qn("w:w"), str(w))
    tblW = t._tbl.tblPr.find(qn("w:tblW"))
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(sum(widths)))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = _twips(widths[j])
            para = cell.paragraphs[0]
            if i >= header_rows and j >= 1:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(str(val))
            run.font.name = FONT
            run.font.size = size
            run.bold = i < header_rows
    for j, w in enumerate(widths):
        for cell in t.columns[j].cells:
            cell.width = _twips(w)
    return t


def load_observed(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])].copy()
    d["y"] = d.early_aki_48h.map({True: 1, False: 0, "True": 1, "False": 0,
                                  "true": 1, "false": 0}).astype(int)
    return d


def fmt_iqr(x: pd.Series, dec: int = 0) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    f = f"%.{dec}f"
    return f"{f % x.median()} ({f % x.quantile(0.25)}–{f % x.quantile(0.75)})"


def fmt_ms(x: pd.Series, dec: int = 1) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    f = f"%.{dec}f"
    return f"{f % x.mean()} ± {f % x.std()}"


def n_pct(k: int, n: int) -> str:
    return f"{k} ({100 * k / n:.1f})"


def orci(v: dict) -> str:
    return f"{v['OR']:.3f} ({v['lo']:.3f}–{v['hi']:.3f})"


def table1(wave: pd.DataFrame, inspire: pd.DataFrame) -> list:
    mov, vit = wave[wave.center == "MOVER"], wave[wave.center == "VitalDB"]
    cohorts = (mov, vit, inspire)
    rows = [["Characteristic", "MOVER", "VitalDB", "INSPIRE"],
            ["n"] + [len(d) for d in cohorts],
            ["Age, years"] + [fmt_ms(d.age) for d in cohorts]]
    rows.append(["Male sex"] + [n_pct(int((d.sex.astype(str).str.upper().str[0] == "M").sum()), len(d))
                                for d in cohorts])
    for a, label in ((1, "I"), (2, "II"), (3, "III"), (4, "IV")):
        rows.append([f"ASA {label}"] + [n_pct(int((d.asa.round() == a).sum()), len(d)) for d in cohorts])
    rows.append(["Baseline creatinine, mg/dl"] + [fmt_iqr(d.baseline_creatinine, 2) for d in cohorts])
    rows.append(["Baseline eGFR, ml/min/1.73 m²"] + [fmt_ms(d.baseline_egfr) for d in cohorts])
    rows.append(["Anaesthesia duration, min"] + [fmt_iqr(d.anesthesia_minutes) for d in cohorts])
    for wave_key, label, insp_key in CATEGORIES:
        vals = []
        for d, key in ((mov, wave_key), (vit, wave_key), (inspire, insp_key)):
            k = int((d.surgical_category == key).sum())
            vals.append(n_pct(k, len(d)) if k else "0 (0.0)")
        rows.append([f"Surgical category: {label}"] + vals)
    rows.append(["Any hypotension"] + [n_pct(int((d.any_hypotension == 1).sum()), len(d)) for d in cohorts])
    rows.append(["Total hypotension, min"] + [fmt_iqr(d.total_low_minutes) for d in cohorts])
    rows.append(["Mean depth below 65 mmHg, mmHg"] + [fmt_ms(d.mean_depth_below_65_mmhg, 2) for d in cohorts])
    rows.append(["Acute kidney injury"] + [n_pct(int(d.y.sum()), len(d)) for d in cohorts])
    return rows


def table2(wave_l: dict, res_l: dict, insp_l: dict) -> list:
    rows = [["Cohort", "Odds ratio (95% CI)"],
            ["Pooled (MOVER + VitalDB), 1-minute", orci(wave_l["pattern_15min_scenario"])]]
    for centre in ("MOVER", "VitalDB"):
        rows.append([centre, orci(wave_l["pattern_by_centre"][centre]["pattern_15min_scenario"])])
    rows.append(["Pooled, resampled to 5-minute", orci(res_l["pattern_15min_scenario"])])
    for centre in ("MOVER", "VitalDB"):
        rows.append([centre, orci(res_l["pattern_by_centre"][centre]["pattern_15min_scenario"])])
    rows.append(["INSPIRE, charted ~5-minute", orci(insp_l["pattern_15min_scenario"])])
    return rows


def table3(wave_l: dict, res_l: dict, insp_l: dict) -> list:
    rows = [["Model", "MOVER + VitalDB, 1-min", "MOVER + VitalDB, resampled 5-min",
             "INSPIRE, charted ~5-min"]]
    for i, step in enumerate(wave_l["ladder"]):
        vals = [step["step"]]
        for lad in (wave_l, res_l, insp_l):
            term = lad["ladder"][i]["total_low10"]
            vals.append(orci(term) if term else "—")
        rows.append(vals)
    return rows


def table4(clinical: dict) -> list:
    rows = [["Cohort", "Minutes below 65 mmHg", "n", "Events", "Crude risk, %",
             "Standardized risk, % (95% CI)"]]
    for key, label in (("pooled_waveform_1min", "MOVER + VitalDB"), ("inspire", "INSPIRE")):
        for i, band in enumerate(BANDS):
            v = clinical["bands"][key][band]
            rows.append([label if i == 0 else "", band, v["n"], v["events"],
                         f"{100 * v['crude'] + 1e-9:.1f}",
                         f"{100 * v['standardised'] + 1e-9:.1f} ({100 * v['lo'] + 1e-9:.1f}–{100 * v['hi'] + 1e-9:.1f})"])
    return rows


S1_LABELS = {
    "threshold_60": "Hypotension threshold 60 mmHg",
    "threshold_70": "Hypotension threshold 70 mmHg",
    "bridge_one_missing_minute": "Bridging one missing minute",
    "hinge_10_minutes": "Episode hinge moved to 10 minutes",
    "vitaldb_10_second": "VitalDB analysed at 10-second resolution",
}


def table_s1(s1: dict) -> list:
    rows = [["Analysis", "Odds ratio (95% CI)"]]
    for r in s1["rows"]:
        label = S1_LABELS.get(r["analysis"], r["analysis"])
        rows.append([label, f'{r["odds_ratio"]:.2f} ({r["ci"][0]:.2f}–{r["ci"][1]:.2f})'])
    return rows


def table_s2(wave_all: pd.DataFrame, inspire_all: pd.DataFrame) -> list:
    rows = [["Cohort", "Group", "n", "Age, years", "ASA", "Anaesthesia, min",
             "Hypotension, min", "eGFR", "Creatinine tests, n", "First test, h"]]

    def block(name: str, allrows: pd.DataFrame):
        included = allrows[allrows.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])]
        excluded = allrows[~allrows.outcome_observed_48h.astype(str).str.lower().isin(["true", "1"])]
        out = []
        for label, part in (("Included", included), ("Excluded", excluded)):
            inc = label == "Included"
            out.append([name if inc else "", label, len(part),
                        f"{part.age.mean():.1f}", f"{part.asa.mean():.1f}",
                        fmt_iqr(part.anesthesia_minutes), fmt_iqr(part.total_low_minutes),
                        f"{part.baseline_egfr.mean():.1f}",
                        fmt_iqr(part.postop_creatinine_count_48h) if inc else "—",
                        fmt_iqr(part.first_postop_creatinine_hours, 1) if inc else "—"])
        return out

    return rows + block("MOVER + VitalDB", wave_all) + block("INSPIRE", inspire_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waveform", type=Path, required=True)
    ap.add_argument("--resampled", type=Path, required=True)
    ap.add_argument("--inspire", type=Path, required=True)
    ap.add_argument("--ladder-wave", type=Path, required=True)
    ap.add_argument("--ladder-resampled", type=Path, required=True)
    ap.add_argument("--ladder-inspire", type=Path, required=True)
    ap.add_argument("--clinical", type=Path, required=True)
    ap.add_argument("--sensitivity-s1", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    wave_all = pd.read_csv(a.waveform)
    insp_all = pd.read_csv(a.inspire)
    wave = load_observed(a.waveform)
    inspire = load_observed(a.inspire)
    wl = json.loads(a.ladder_wave.read_text(encoding="utf-8"))
    rl = json.loads(a.ladder_resampled.read_text(encoding="utf-8"))
    il = json.loads(a.ladder_inspire.read_text(encoding="utf-8"))
    cl = json.loads(a.clinical.read_text(encoding="utf-8"))
    s1 = json.loads(a.sensitivity_s1.read_text(encoding="utf-8"))

    d = new_doc()
    caption(d, "Table 1. Baseline characteristics of the analytic cohorts")
    note(d, "Values are n (%), mean ± SD, or median (IQR) among patients with an assessable "
            "48-hour creatinine outcome. AKI, acute kidney injury; ASA, American Society of "
            "Anesthesiologists physical status; eGFR, estimated glomerular filtration rate.")
    table(d, table1(wave, inspire), [3515, 2041, 2041, 2041])
    d.save(a.out_dir / "Table1.docx")

    d = new_doc()
    caption(d, "Table 2. Episode-pattern contrast by cohort and centre")
    note(d, "Odds ratio for one 20-minute hypotensive episode versus four 5-minute episodes "
            "at the same total duration and mean depth, from the model-based contrast "
            "(exp(1.5 × beta)). The waveform cohorts were prespecified; the resampled and "
            "charted cohorts are exploratory.")
    table(d, table2(wl, rl, il), [5669, 3402])
    d.save(a.out_dir / "Table2.docx")

    d = new_doc()
    caption(d, "Table 3. Sequential logistic models for cumulative hypotension")
    note(d, "Odds ratios per 10 minutes with mean arterial pressure below 65 mmHg, with robust "
            "(HC0) standard errors. The full prespecified model applies to the two waveform "
            "cohorts; the resampled and charted cohorts are exploratory extensions.")
    table(d, table3(wl, rl, il), [2608, 2381, 2494, 2494])
    d.save(a.out_dir / "Table3.docx")

    d = new_doc()
    caption(d, "Table 4. Standardized 48-hour acute kidney injury risk by cumulative hypotension")
    note(d, "Risks are standardized to the covariate distribution of each cohort, with pointwise "
            "95% confidence intervals from 400 draws of the sampling distribution. Post hoc and "
            "exploratory.")
    table(d, table4(cl), [2268, 2041, 1134, 1134, 1474, 2494])
    d.save(a.out_dir / "Table4.docx")

    d = new_doc()
    caption(d, "Supplementary Table S1. Prespecified sensitivity analyses of the episode-pattern contrast")
    note(d, "Odds ratio for one 20-minute hypotensive episode versus four 5-minute episodes at the "
            "same total duration and mean depth, in the pooled MOVER + VitalDB waveform cohort.")
    table(d, table_s1(s1), [5669, 3402])
    d.add_paragraph(s1["flame_note"])
    d.save(a.out_dir / "Supplementary_Table_S1.docx")

    d = new_doc()
    caption(d, "Supplementary Table S2. Patients with and without an assessable 48-hour creatinine outcome")
    note(d, "Values are mean, median (IQR) or n. Patients without a postoperative creatinine within "
            "48 hours were excluded from outcome models and were not counted as uninjured. "
            "Creatinine tests, number of postoperative measurements within 48 hours; first test, "
            "hours after the end of anaesthesia.")
    table(d, table_s2(wave_all, insp_all), [1250, 950, 650, 600, 560, 1180, 1180, 640, 990, 930], size=Pt(8))
    d.save(a.out_dir / "Supplementary_Table_S2.docx")

    print("written tables to", a.out_dir)


if __name__ == "__main__":
    main()
