# Intraoperative hypotension and acute kidney injury: blood-pressure sampling resolution (MOVER, VitalDB, INSPIRE)

Analysis code, locked analysis contract, aggregate expected results and manuscript figure for a multi-database study of intraoperative hypotension and 48-hour creatinine-defined acute kidney injury.

## What the study does

The prespecified primary analysis (v1.0) tested whether concentrating 20 minutes of intraoperative hypotension into one continuous episode carries greater 48-hour creatinine-defined acute kidney injury risk than distributing the same duration across four 5-minute episodes, using the MOVER (United States) and VitalDB (South Korea) databases. See the archived primary results below.

The v2.0 analyses asked a different question: does the cumulative-duration association itself depend on how often arterial pressure was recorded? Three cohorts were used — MOVER, VitalDB (both waveform, summarized in 1-minute bins) and INSPIRE (charted approximately every 5 minutes, processed with zero-order hold). As an internal control, the two waveform cohorts were resampled to every fifth minute with patients, covariates and outcomes held fixed.

**Everything added in v2.0 is post hoc and exploratory.** It was not part of the locked analysis contract in `config/ANALYSIS_CONTRACT_v4.yaml` and should be read as hypothesis-generating.

## What is and is not included

No source data and no patient-level derived data are included, and none may be uploaded here. MOVER, VitalDB and INSPIRE are each governed by their own access and data-use conditions; a complete rerun requires authorised local access to all three. The package contains code, the locked contract, aggregate model reports, audit records and figures only.

## Environment

Python 3.12 was used. Create an isolated environment and install the locked dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS or Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.

## Reproducing the v2.0 analyses

These scripts assume the working directories produced by the v1.0 pipeline (harmonized feature tables) plus the INSPIRE source files.

1. **INSPIRE cohort** (charted pressure, zero-order hold; VitalDB-overlapping operations removed):

```bash
python code/build_inspire_v1.py
```

2. **Resolution control** — resample the MOVER/VitalDB 1-minute pressure grid to every fifth minute:

```bash
python code/downsample_to_5min.py
```

3. **Model ladders** — duration coefficient per 10 minutes below 65 mmHg, adding covariates in steps, with and without the episode-pattern term:

```bash
# pooled waveform cohorts
python code/model_ladder.py --input COMBINED_FEATURES_V4.csv \
    --low-column total_low_minutes --center --out model_ladder_1min.json

# the same patients resampled to 5-minute charting
python code/model_ladder.py --input downsampled_5min_features.csv \
    --low-column ds_total --center --out model_ladder_5min.json

# INSPIRE charted cohort
python code/model_ladder.py --input INSPIRE_FEATURES_HOLD.csv \
    --low-column total_low_minutes --out model_ladder_inspire.json
```

4. **Figure 1** (regenerated from the feature tables, not from hard-coded numbers):

```bash
python code/make_figure_resolution.py
```

Compare the output with the archived values in `expected_results/`.

## Archived v2.0 results

Cumulative hypotension duration, adjusted odds ratio per 10 minutes below 65 mmHg (95% CI):

| Cohort | Resolution | Adjusted OR |
|---|---|---|
| MOVER + VitalDB | 1-minute waveform | 1.055 (1.019–1.091) |
| MOVER + VitalDB | resampled to 5-minute | 0.959 (0.928–0.990) |
| INSPIRE | charted ~5-minute | 0.948 (0.938–0.958) |

The episode-pattern contrast was null in all three cohorts. Correlations between total duration and the episode-pattern term were 0.88 (1-minute), 0.939 (resampled) and 0.967 (INSPIRE).

## Archived v1.0 primary result

The complete-outcome analytic cohort contained 2,958 patients and 200 acute kidney injury events. For one 20-minute episode versus four 5-minute episodes at the same total duration and mean depth, the adjusted odds ratio was 1.08696 (95% confidence interval 0.88313 to 1.33783), and the likelihood-ratio P value for the duration-pattern term was 0.44181. These values are checked by `verify_release.py`.

## Interpretation boundary

The analyses estimate associations. They do not establish that changing episode continuity, or changing how often pressure is recorded, would change kidney outcomes. The outcome is creatinine-defined acute kidney injury within 48 hours and is not complete KDIGO adjudication because urine output was unavailable.

## Release contents

- `code/`: export, validation, feature, model, sensitivity and figure scripts
- `code/build_inspire_v1.py`, `code/downsample_to_5min.py`, `code/model_ladder.py`, `code/make_figure_resolution.py`: the v2.0 post hoc analyses
- `tests/`: synthetic-data method tests
- `config/`: locked contract, category rules and original analysis-lock record
- `expected_results/`: aggregate archived outputs only
- `figures/`: manuscript figures in PNG, PDF and SVG
- `docs/DATA_DICTIONARY.md`: harmonized variable definitions
- `release_manifest.json`: SHA-256 checksums

## Version history

- **v2.0.0** — adds the INSPIRE cohort, the 5-minute resampling control, the model ladders and the resolution figure. All post hoc and exploratory.
- **v1.0.0** — locked two-centre primary and sensitivity analyses.
