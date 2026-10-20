# Intraoperative hypotension and acute kidney injury across three perioperative databases (MOVER, VitalDB, INSPIRE)

Analysis code, aggregate expected results and manuscript figures for a multi-database study of
intraoperative hypotension and 48-hour creatinine-defined acute kidney injury.

## What the study does

The prespecified primary analysis (v1.0) tested whether concentrating 20 minutes of intraoperative
hypotension into one continuous episode carries greater 48-hour creatinine-defined acute kidney injury
risk than distributing the same duration across four 5-minute episodes, using the MOVER (United States)
and VitalDB (South Korea) databases.

The v2.x analyses asked two further questions, both post hoc and exploratory: how large the
cumulative-duration association is in absolute terms and how its shape varies across the exposure range,
and whether rebuilding the same exposure from five-minute charted pressure changes the estimate or only
its precision. Three cohorts were used: MOVER and VitalDB (waveform, summarized in 1-minute bins) and
INSPIRE (charted approximately every five minutes, processed with zero-order hold). As an internal
control, the two waveform cohorts were resampled to every fifth minute with patients, covariates and
outcomes held fixed.

**Everything added after v1.0 is post hoc and exploratory.** It was not part of the locked analysis
contract in `config/ANALYSIS_CONTRACT_v4.yaml` and should be read as hypothesis-generating.

## What is and is not included

No source data and no patient-level derived data are included, and none may be uploaded here. MOVER,
VitalDB and INSPIRE are each governed by their own access and data-use conditions; a complete rerun
requires authorised local access to all three. The package contains code, the locked contract, aggregate
model reports, audit records and figures only.

## Environment

The archived v2.0.2 results were produced with Python 3.10.12 and the versions pinned in
`requirements.txt` (notably statsmodels 0.15.0, pandas 2.3.3, numpy 2.2.6, matplotlib 3.10.9;
`python-docx` is needed only for the table builder).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

## Reproducing the v2.x analyses

The scripts take the harmonized feature tables produced by the v1.0 pipeline plus the INSPIRE 1.4.2
source files. Paths below are arguments, not hard-coded.

1. **INSPIRE cohort** — validity filter first, then the quality gate, then baseline kidney function,
   then one operation per patient (the order matters; see version history):

```bash
python code/build_inspire_v1.py \
    --inspire-dir /path/to/INSPIRE \
    --vitaldb-cases /path/to/vitaldb/clinical_data.csv \
    --out-dir out/inspire
```

The build writes `INSPIRE_FEATURES_HOLD.csv` and `INSPIRE_FEATURE_AUDIT.json`. The audit records the
flow: 130,960 operations → 91,636 eligible → 39,307 with arterial pressure → 33,178 passing the
pressure-quality gate → 29,917 with valid baseline kidney function → 26,532 one per patient →
22,125 with an assessable outcome (1,184 AKI).

2. **Resolution control** — resample the MOVER/VitalDB 1-minute grid to every fifth minute:

```bash
python code/downsample_to_5min.py \
    --mover-features /path/to/MOVER_FEATURES_V4.csv \
    --mover-map /path/to/mover_export_map.csv \
    --vitaldb-features /path/to/VITALDB_FEATURES_V4.csv \
    --vitaldb-cases /path/to/vitaldb/clinical_data.csv \
    --vitaldb-map-dir /path/to/vitaldb/map_raw \
    --vitaldb-manifest /path/to/vitaldb/download_report.json \
    --out-dir out/resampled
```

Each kept reading is carried to the next kept reading, so internal gaps are held over as well; the last
reading stops at the end of anaesthesia.

3. **Model ladders** — duration coefficient per 10 minutes below 65 mmHg, adding terms in a fixed order:

```bash
python code/model_ladder.py --input out/COMBINED_FEATURES_V4.csv \
    --low-column total_low_minutes --center --out out/ladder_1min.json

python code/model_ladder.py --input out/resampled/downsampled_5min_features.csv \
    --low-column ds_total --depth-column ds_depth --excess5-column ds_excess5 \
    --center --out out/ladder_5min.json

python code/model_ladder.py --input out/inspire/INSPIRE_FEATURES_HOLD.csv \
    --low-column total_low_minutes --out out/ladder_inspire.json
```

4. **Absolute risk and shape** — banded standardized risks, restricted cubic spline, centre-level
   sensitivity. Point estimates use the fitted coefficients; 95% intervals come from 400 draws of the
   coefficient vector from a multivariate normal with the HC0 covariance, standardizing in each draw:

```bash
python code/analysis_clinical.py \
    --waveform-csv out/COMBINED_FEATURES_V4.csv \
    --resampled-csv out/resampled/downsampled_5min_features.csv \
    --inspire-csv out/inspire/INSPIRE_FEATURES_HOLD.csv \
    --out out/clinical.json
```

5. **Figures** — regenerated from those files, so they cannot drift from the reported numbers:

```bash
python code/make_figure_resolution.py \
    --waveform-csv out/COMBINED_FEATURES_V4.csv \
    --resampled-csv out/resampled/downsampled_5min_features.csv \
    --inspire-csv out/inspire/INSPIRE_FEATURES_HOLD.csv --out-dir figures

python code/make_figure_absolute_risk.py --clinical out/clinical.json --out-dir figures
```

The y axis of the absolute-risk figure is derived from the upper confidence limit, so the band is never
clipped.

6. **Carry-forward audit and capped-hold sensitivity** — the sensitivity needs anaesthesia start and end
   times, which the main feature table does not carry:

```bash
python - <<'PY'
import pandas as pd
f = pd.read_csv("out/inspire/INSPIRE_FEATURES_HOLD.csv")
ops = pd.read_csv("/path/to/INSPIRE/operations.csv.gz",
                  usecols=["op_id", "anstart_time", "anend_time"])
f.merge(ops, left_on="inspire_case", right_on="op_id", how="left") \
 .to_csv("out/inspire_aug.csv", index=False)
PY

python code/inspire_gap_sensitivity.py \
    --features out/inspire_aug.csv --inspire-dir /path/to/INSPIRE --out-dir out/gaps
```

Events are broken at any uncovered interval when the hold is capped, so an episode can never be longer
than the exposure it is built from. The writer also stores exact counts for the interval audit.

7. **Tables** — Table 1–4 and Supplementary Tables S1–S2 are generated from the feature tables and the
   JSON above; no number is typed into a Word file by hand:

```bash
python code/build_tables.py \
    --waveform out/COMBINED_FEATURES_V4.csv \
    --resampled out/resampled/downsampled_5min_features.csv \
    --inspire out/inspire/INSPIRE_FEATURES_HOLD.csv \
    --ladder-wave out/ladder_1min.json \
    --ladder-resampled out/ladder_5min.json \
    --ladder-inspire out/ladder_inspire.json \
    --clinical out/clinical.json \
    --sensitivity-s1 expected_results/SENSITIVITY_S1.json \
    --out-dir out/tables
```

## Archived results

Cumulative hypotension duration, odds ratio per 10 minutes below 65 mmHg (95% CI):

| Cohort | Resolution | Crude | Adjusted* | + anaesthesia duration |
|---|---|---|---|---|
| MOVER + VitalDB | 1-minute waveform | 1.093 (1.060–1.127) | 1.055 (1.019–1.091) | 1.026 (0.991–1.063) |
| MOVER + VitalDB | resampled to 5-minute | 1.080 (1.050–1.112) | 1.043 (1.010–1.078) | 1.015 (0.982–1.050) |
| INSPIRE | charted ~5-minute | 1.040 (1.029–1.050) | 1.054 (1.043–1.066) | 1.028 (1.014–1.041) |

\* Age, sex, ASA physical status, baseline eGFR and surgical category; centre included in the pooled
cohorts. The full ladder (adding mean depth and then the episode-pattern term) is in
`expected_results/MODEL_LADDER_*.json`; with the pattern term the duration estimates are 0.996
(0.912–1.089), 1.026 (0.912–1.155) and 1.024 (0.971–1.080).

The episode-pattern contrast, expressed for the 15-minute scenario (exp(1.5 × coefficient)), was
1.087 (0.883–1.338) in the pooled waveform cohort, 0.975 (0.779–1.221) after resampling, and 1.002
(0.909–1.105) in INSPIRE. Correlations between total duration and the episode-pattern term were 0.88
(1-minute), 0.939 (resampled) and 0.966 (INSPIRE).

Standardized 48-hour acute kidney injury risk by cumulative duration is archived in
`expected_results/STANDARDIZED_RISKS.csv` and `expected_results/CLINICAL_ANALYSES.json`. In the waveform
cohort the spline gives 6.6%, 6.2%, 7.5%, 11.3% and 14.7% at 0, 45, 60, 90 and 120 minutes; in INSPIRE
4.9%, 6.1%, 8.1%, 8.6% and 8.1%.

Sensitivity analyses of the charted cohort are archived alongside:

- capping the carry-forward interval at 15 or 30 minutes leaves the adjusted odds ratio at 1.054;
- 295 of the 26,532 included patients (1.1%) had another operation in the VitalDB open dataset; 275 of
  them had an outcome, and excluding those 275 leaves 21,850 patients and an odds ratio of 1.054
  (1.042–1.066);
- the interval audit (`expected_results/INSPIRE_GAP_AUDIT.json`) covers 1,127,121 intervals: 99.92% were
  10 minutes or shorter, 141 intervals exceeded 15 minutes, and those contributed 0.08% of held time.

## Archived v1.0 primary result

The complete-outcome analytic cohort contained 2,958 patients and 200 acute kidney injury events. For one
20-minute episode versus four 5-minute episodes at the same total duration and mean depth, the adjusted
odds ratio was 1.08696 (95% confidence interval 0.88313 to 1.33783), and the likelihood-ratio P value for
the duration-pattern term was 0.44181.

## Interpretation boundary

The analyses estimate associations. They do not establish that changing episode continuity, or changing
how often pressure is recorded, would change kidney outcomes. The outcome is creatinine-defined acute
kidney injury within 48 hours and is not complete KDIGO adjudication because urine output was unavailable.
The episode-pattern contrast is imprecisely estimated; the data do not provide evidence of an additional
association, but they do not establish equivalence between episode patterns.

## Version history

- **v2.0.2** (2026-10-19) — correction release. (1) `build_inspire_v1.py` applies the same 20–160 mmHg
  validity filter used for the waveform cohorts **and rebuilds cohort membership after that filter**: the
  earlier rebuild recomputed the exposure without re-applying the pressure-quality gate, so 1,039
  patients with an assessable outcome (4.5%) violated the stated 15-minute first/last reading rule. The
  cohort is now 26,532 patients (22,125 with an outcome, 1,184 AKI). (2) `downsample_to_5min.py` no
  longer extends the final interval past the end of anaesthesia and keeps every case with at least one
  valid reading (2,958 patients). (3) `inspire_gap_sensitivity.py` breaks low-pressure episodes at
  uncovered intervals when the hold is capped, and records exact interval counts. (4)
  `analysis_clinical.py` uses the fitted coefficients for standardized point estimates and the
  coefficient draws only for intervals. (5) `make_figure_absolute_risk.py` derives the y axis from the
  data. (6) Added `build_tables.py`, which generates every manuscript table from the feature tables and
  result JSON, and `SENSITIVITY_S1.json`, which is generated from the archived sensitivity report
  `SENSITIVITY_S1_SOURCE_V4_1.json` (included, SHA-256 recorded in the JSON) and formats the numbers shown
  in Supplementary Table S1.
- **v2.0.1** — correction release. An outcome-encoding error in v2.0.0 passed the binary outcome to the
  model as a categorical variable, so the models fitted the complement of the event and every odds ratio
  for the 5-minute resampled cohort and for INSPIRE was the reciprocal of the correct value. The error is
  fixed in `code/model_ladder.py` and `code/make_figure_resolution.py`; all affected results and figures
  have been regenerated. The direction of the association is positive in all three cohorts, and the
  v2.0.0 claim that measurement resolution reverses the association is withdrawn.
- **v2.0.0** — added the INSPIRE cohort, the 5-minute resampling control, the model ladders and the
  resolution figure. Superseded by v2.0.1.
- **v1.0.0** — locked two-centre primary and sensitivity analyses.

## Release contents

- `code/`: export, validation, feature, model, sensitivity, figure and table scripts
- `expected_results/`: aggregate archived outputs only
- `figures/`: manuscript figures in PNG, PDF and EPS
- `config/`: locked contract
- `requirements.txt`: pinned environment
