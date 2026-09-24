# MOVER and VitalDB hypotension episode continuity study

This package contains the analysis code, locked analysis contract, aggregate expected results, and manuscript figures for the two-center study of intraoperative hypotension episode continuity and 48-hour creatinine-defined acute kidney injury.

## What is and is not included

The package does not contain source data, patient-level derived data, credentials, or local absolute paths. A complete rerun requires authorized local access to MOVER and the public VitalDB source files under their respective data-use conditions. Aggregate model reports, coefficients, descriptive summaries, and figures are included so reviewers can inspect the reported calculations and compare a rerun with the archived results.

## Environment

Python 3.12 was used. Create an isolated environment and install the locked dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS or Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.

## Tests without source data

```bash
python run_smoke_tests.py
```

These tests use synthetic inputs. They validate MOVER export logic, feature construction, harmonization, primary-model behavior, release checksums, and the principal archived estimates.

## Full reproduction

1. Produce the MOVER export with an explicit local path:

```bash
python code/mover_reexport_v5_1.py --root "<MOVER_ROOT>"
```

2. Copy `config.example.yaml` to a private working location and replace every `/path/to/...` value. Do not commit credentials or source-data paths.
3. Run the locked pipeline:

```bash
python run_pipeline.py --config path/to/config.yaml
```

The pipeline validates and constructs both center-specific feature files, harmonizes variables, performs the preflight and analysis lock, fits the primary and sensitivity models, audits exposure support, computes standardized-risk intervals, and rebuilds all three figures. It refuses to overwrite an existing output directory.

## Archived primary result

The complete-outcome analytic cohort contained 2,958 patients and 200 acute kidney injury events. For one 20-minute episode versus four 5-minute episodes at the same total duration and mean depth, the adjusted odds ratio was 1.08696 (95% confidence interval 0.88313 to 1.33783), and the likelihood-ratio P value for the duration-pattern term was 0.44181. These archived values are checked by `verify_release.py`.

## Interpretation boundary

The analysis estimates associations. It does not establish that changing episode continuity changes kidney outcomes. The outcome is creatinine-defined acute kidney injury within 48 hours and is not complete KDIGO adjudication because urine output was unavailable. The prespecified contrast is a model-based comparison: in the analytic cohort only 6 patients accumulated 20 hypotensive minutes within a single episode and none had exactly four 5-minute episodes (see `expected_results/EXPOSURE_SUPPORT_AUDIT_V4_1.json`).

## Release contents

- `code/`: export, validation, feature, model, sensitivity, support, interval, and figure scripts
- `tests/`: synthetic-data method tests
- `config/`: locked contract, category rules, and original analysis-lock record
- `expected_results/`: aggregate archived outputs only
- `figures/`: manuscript figures in PNG, PDF, and SVG
- `docs/DATA_DICTIONARY.md`: harmonized variable definitions
- `release_manifest.json`: SHA-256 checksums

Before public repository release, the authors should add author metadata, a repository DOI, and a code license approved by their institution.

## Supportive and secondary analyses (v1.2)

`code/run_flame.R` reproduces the prespecified supportive flexible accumulation
analysis (R plus `flameRisk` at the pinned commit). `code/secondary_analyses.py`
and `code/remaining_secondary.py` reproduce the remaining prespecified analyses,
including a self-check that recomputes the 48-hour outcome from raw sources and
matches the archived 200 events.

## Citing

See `CITATION.cff`. This archive contains no patient-level data; source data must
be obtained from MOVER and VitalDB under their respective access conditions.
