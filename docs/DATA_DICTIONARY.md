# Harmonized analysis data dictionary

The pipeline creates `COMBINED_FEATURES_V4.csv`; this file is not distributed in the public code package because it contains case-level derived records.

| Variable | Meaning |
| --- | --- |
| center | Source database, MOVER or VitalDB |
| analysis_case | Deidentified operation identifier |
| analysis_patient | Deidentified patient identifier used for one-operation selection |
| age | Age in years |
| sex | Harmonized recorded sex |
| asa | American Society of Anesthesiologists physical status |
| anesthesia_minutes | Anesthesia duration |
| surgical_category | Locked harmonized surgical category |
| common_surgical_support | Indicator for categories represented at both centers |
| baseline_creatinine | Latest valid preoperative creatinine within 30 days |
| baseline_egfr | 2021 race-free CKD-EPI estimated glomerular filtration rate |
| icu_flag | Perioperative intensive care indicator used in the observation model |
| map_observed_minutes | Minutes with valid invasive mean arterial pressure |
| map_total_minutes | Expected minutes in the anesthesia interval |
| map_coverage | Valid invasive mean arterial pressure coverage proportion |
| total_low_minutes | Minutes with mean arterial pressure below 65 mmHg |
| episode_count | Number of hypotensive episodes |
| longest_episode_minutes | Longest hypotensive episode |
| excess5_minutes | Sum of episode time beyond the first 5 minutes of each episode |
| excess10_minutes | Sum of episode time beyond the first 10 minutes of each episode |
| auc_below_65_mmhg_minutes | Area below 65 mmHg |
| mean_depth_below_65_mmhg | Area below threshold divided by hypotensive minutes |
| episode_durations_json | Complete vector of episode durations |
| outcome_observed_48h | Availability of an assessable postoperative creatinine outcome |
| postop_creatinine_count_48h | Number of valid postoperative creatinine measurements |
| first_postop_creatinine_hours | Time from anesthesia end to first valid postoperative value |
| early_aki_48h | Creatinine-defined acute kidney injury within 48 hours |
