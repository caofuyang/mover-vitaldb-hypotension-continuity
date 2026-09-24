# Reproducibility package validation

Validation date: 2026-09-23 (release v1.1)

## Completed checks

- All Python files compiled under Python 3.12.
- The synthetic MOVER export completed and passed the independent export validator.
- Synthetic MOVER feature construction passed.
- Synthetic two-center harmonization passed.
- The synthetic primary-model test recovered the expected positive duration-pattern signal.
- All three manuscript figures were regenerated from the archived aggregate JSON reports in PNG, PDF, and SVG formats; Figure 3 now carries the 95% confidence intervals its legend describes.
- The archived principal estimate, confidence interval inputs, likelihood-ratio P value, cohort size, and checksums are tested by `verify_release.py`.
- The released descriptive table now describes the outcome-observed analytic cohort, matching Table 1 of the manuscript, and is checked against manuscript values by `verify_release.py`.
- The exposure-support audit reproduces the support statements in the manuscript.
- The standardized-risk interval file reproduces the archived primary odds ratio and risk-difference interval.
- The release contains no source data, case-level feature tables, credentials, Python caches, or machine-specific source paths.

## Reproduction of the locked analysis

Ran on 2026-09-23 from the authorized local MOVER export and the rebuilt VitalDB
feature file:

- `build_mover_features_v4.py` reproduced 1,178 MOVER patients, 1,137 with an
  observed 48-hour creatinine outcome, and 132 events.
- `harmonize_features_v4.py` reproduced a combined feature table whose SHA-256
  equals the hash recorded in `config/ANALYSIS_LOCK_MANIFEST_V4_1.json`
  (`0bd96f2f…`).
- `fit_primary_model_v4.py` passed the analysis-lock hash gate and reproduced the
  archived scenario odds ratios to machine precision, including the
  observation-weighted scenario (1.095409946262) and the 500-sample stratified
  bootstrap (499 valid fits; median 1.1009, percentile interval 0.9135 to 1.3997).

## Scope of validation

The complete real-data pipeline cannot run without authorized local source
files. The included tests validate the computation on synthetic inputs, while
`expected_results/` preserves the aggregate outputs from the locked real-data
analysis for comparison. A full independent replication requires obtaining
MOVER and VitalDB under their applicable access and data-use conditions.

## Changes in v1.1

- Added `code/standardized_risk_intervals_v4_1.py`, which reports 95% intervals
  for each standardized risk in the primary contrast.
- `code/make_figures_v4_1.py` now takes `--risk-intervals` and draws 95%
  confidence intervals on Figure 3.
- Added `code/exposure_support_audit_v4_1.py`; the manuscript now states how many
  patients occupy the compared episode patterns.
- `code/descriptive_observation_audit_v4.py` now also writes
  `TABLE1_DESCRIPTIVE_OBSERVED_COHORT.csv`, the table that corresponds to
  manuscript Table 1.
- `verify_release.py` extended to check the new artifacts.
- `expected_results/STANDARDIZED_RISK_INTERVALS_V4_1.json`,
  `expected_results/EXPOSURE_SUPPORT_AUDIT_V4_1.json`, and a final-cohort
  `expected_results/DESIGN_SIMULATION_RESULTS.json` added.

## Changes in v1.2

The manuscript previously did not report several analyses that the locked
contract prespecifies. All of them are now reported, and the package carries
the code and aggregate outputs that produced them.

- `code/run_flame.R` runs the prespecified supportive flexible accumulation
  analysis with `flameRisk` pinned at commit
  `85a6d1b81dcbdd1f45b488d13567cbc5577649bc`. The estimated accumulation
  function retained 1.00 effective degrees of freedom (chi-square 1.99,
  P=0.158) at basis dimensions 5 and 6, and a basis dimension of 4 failed to
  converge. Aggregate output: `expected_results/FLAME_SUMMARY.json`.
- `code/secondary_analyses.py` runs the two-hinge, sustained-fraction,
  episode-count, episode-variability, and patient-clustering checks.
  Aggregate output: `expected_results/SECONDARY_ANALYSES.json`.
- `code/remaining_secondary.py` runs the 7-day creatinine outcome and
  next-surgery censoring analyses. It recomputes the 48-hour outcome from the
  raw sources as a self-check and reproduces the archived 200 events exactly
  (MOVER 132, VitalDB 68), which validates the time conventions used.
  Aggregate output: `expected_results/REMAINING_SECONDARY.json`.

Note that `run_flame.R` requires R and the pinned `flameRisk` commit; it is not
part of the Python environment.

## Changes in v1.3

- `expected_results/MOVER_SCALE_BLAND_ALTMAN.json` adds the Bland-Altman
  characterisation of the MOVER arterial pressure scale, computed on the same
  241,132 paired records as the locked validation subset (reproduced here:
  n=241,132, r=0.5553, median absolute difference 8 mmHg). Bias -3.7 mmHg,
  95% limits of agreement -36.7 to +29.3 mmHg.
- `expected_results/FLAME_SUMMARY.json` now also records the contrast on the
  log-odds scale, which under the linear functional specification does not
  involve the covariates and is therefore identical for every patient.

## Changes in v1.4

- `code/make_figures_v4_1.py`: Figure 1 said "anaesthesia". The journal requires
  American English, so the figure text is now "anesthesia" and all three figures
  were regenerated.
- The manuscript itself was corrected where it used "analysed"; page numbers and
  per-section page breaks were added to satisfy the journal's layout rules, and
  reference page ranges now use en dashes as in the journal's examples.
- The journal's actual limits were confirmed from its submission guidelines:
  Original articles 4,000 words; structured abstract 250 words. The manuscript
  sits at 3,324 and 249 words respectively, so no cutting was necessary.

## Changes in v1.5

- `code/remaining_secondary.py` now also reports the composition of the primary
  48-hour outcome by criterion. Of the 200 events, 114 met the 1.5x-baseline
  ratio criterion and 86 met only the 0.3 mg/dl absolute criterion.
- `expected_results/REMAINING_SECONDARY.json` carries that breakdown.
