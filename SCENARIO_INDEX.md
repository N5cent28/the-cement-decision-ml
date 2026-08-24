# Scenario Index

Single source of truth mapping every ground-truth × feature-set sensitivity
scenario to its scripts and output directories. The numbering scheme (`03g`,
`04g`, ... following the existing `03b`/`03c`/`03d`/`03e`/`03f` pattern) is
intentional: each preprocessing script either establishes a new patient split
("base" scripts) or explicitly reuses an existing split by reading another
scenario's `train.csv`/`test.csv` ("same split" scripts), so the family of
scripts for one ground truth always varies exactly one thing — the feature
set — while holding the split and target fixed. New scenarios should follow
this pattern: pick the next unused letter, and if you're adding new feature
sets for an existing ground truth, always derive from that ground truth's
existing base split rather than re-deriving your own.

## Ground truths

| Column | Definition | Built by |
|---|---|---|
| `gt_majority` | ≥2 of 3 raters (original surgeon, Halldor, 3D surgeon) vote cemented | `03_preprocessing.py` |
| `gt_original` | Original operating surgeon's real-time decision only | `03_preprocessing.py`, `03j_preprocessing_original_ct_only.py` |
| `gt_vote_fraction` | Continuous fraction of the 3 raters voting cemented (regression target) | `03b_preprocessing_vote_fraction.py` |
| `gt_h3d_agree` | Halldor and 3D surgeon agree (2 of 3 raters; **excludes** the original surgeon's vote) | `03e_preprocessing_agree_ct_only.py` |
| `gt_unanimous` | All 3 raters agree (true 3-way unanimous consensus) | `03g_preprocessing_unanimous_ct_only.py` |

`gt_h3d_agree` and `gt_unanimous` are easy to confuse — `gt_h3d_agree` only
requires 2 named raters to match and says nothing about the original
surgeon; `gt_unanimous` requires all 3, including the original surgeon.

## Full scenario matrix

| Scenario | Ground truth | Feature set | Preprocessing script(s) | Train/eval script | Processed data dir | Results dir |
|---|---|---|---|---|---|---|
| 1 | Majority vote | CT-only | `03_preprocessing.py` | `04_train_evaluate.py` | `processed_data/` | `results/` |
| 3 | Majority vote | CT + demographics | `03c_build_split_with_demographics.py` (reuses scenario 1 split) | `04c_train_evaluate_majority_with_demographics.py` | `processed_data_demographics_same_split/` | `results_majority_with_demographics_same_split/` |
| 4 | Majority vote | Demographics-only | `03d_build_demo_only_same_split.py` (derived from scenario 3) | `04d_train_evaluate_majority_demographics_only.py` | `processed_data_demographics_only_same_split/` | `results_majority_demographics_only_same_split/` |
| 2 | Vote fraction | CT-only | `03b_preprocessing_vote_fraction.py` | `04b_train_evaluate_vote_fraction.py` | `processed_data_vote_fraction/` | `results_vote_fraction/` |
| 13 | Vote fraction | CT + demographics | `03l_build_vote_fraction_with_demographics_same_split.py` (reuses scenario 2 split) | `04l_train_evaluate_vote_fraction_with_demographics.py` | `processed_data_vote_fraction_with_demographics_same_split/` | `results_vote_fraction_with_demographics_same_split/` |
| 12 | Vote fraction | Demographics-only | `03m_build_vote_fraction_demo_only_same_split.py` (derived from scenario 13) | `04m_train_evaluate_vote_fraction_demographics_only.py` | `processed_data_vote_fraction_demographics_only_same_split/` | `results_vote_fraction_demographics_only_same_split/` |
| 5 | Halldor+3D agree (2/3) | CT-only | `03e_preprocessing_agree_ct_only.py` | `04e_train_evaluate_agree_ct_only.py` | `processed_data_scenario5_agree_ct_only/` | `results_scenario5_agree_ct_only/` |
| 7 | Unanimous (3/3) | CT-only | `03g_preprocessing_unanimous_ct_only.py` | `04g_train_evaluate_unanimous_ct_only.py` | `processed_data_unanimous_ct_only/` | `results_unanimous_ct_only/` |
| 9 | Unanimous (3/3) | CT + demographics | `03h_build_unanimous_with_demographics_same_split.py` (reuses scenario 7 split) | `04h_train_evaluate_unanimous_with_demographics.py` | `processed_data_unanimous_with_demographics_same_split/` | `results_unanimous_with_demographics_same_split/` |
| 8 | Unanimous (3/3) | Demographics-only | `03i_build_unanimous_demo_only_same_split.py` (derived from scenario 9) | `04i_train_evaluate_unanimous_demographics_only.py` | `processed_data_unanimous_demographics_only_same_split/` | `results_unanimous_demographics_only_same_split/` |
| 10 | Original surgeon only | CT-only | `03j_preprocessing_original_ct_only.py` | `04j_train_evaluate_original_ct_only.py` | `processed_data_original_ct_only/` | `results_original_ct_only/` |
| 6 | Original surgeon only | CT + demographics | `03f_preprocessing_original_ct_plus_demo.py` (independent split; verified identical to scenario 10) | `04f_train_evaluate_original_ct_plus_demo.py` | `processed_data_scenario6_original_ct_plus_demo/` | `results_scenario6_original_ct_plus_demo/` |
| 11 | Original surgeon only | Demographics-only | `03k_build_original_demo_only_same_split.py` (reuses scenario 10 split) | `04k_train_evaluate_original_demographics_only.py` | `processed_data_original_demographics_only_same_split/` | `results_original_demographics_only_same_split/` |

Rows are grouped by ground truth (not scenario number) so each 3-row family
reads as one isolated feature-set comparison.

## Other pipeline scripts

- `05_compare_all_experiments.py` — cross-scenario comparison chart for the majority-vote family (scenarios 1/3/4). Not yet extended to scenarios 7–13.
- `06_feature_importance_best_model_scenarios_1_3_4_5_6.py` — best-model feature importance for all classification scenarios (currently 1, 3, 4, 5, 6, 7, 8, 9, 10, 11). Vote-fraction scenarios (2, 12, 13) are regression tasks and out of scope.
- `07_repeated_grouped_split_eval.py` — repeated patient-grouped split robustness check with 95% confidence intervals, covering **all 13 scenarios** (30 repeats each). Output: `results_repeated_splits_all_scenarios/`. This supersedes the older `results_repeated_splits_scenarios_1_3_4_5_6/` (10 repeats, scenarios 1/3/4/5/6 only), which is kept only as a historical artifact — treat the new directory as authoritative.
- `08_build_all_models_metrics_table.py` — the single master table: every model's accuracy/precision/recall/F1/ROC-AUC/PR-AUC (or RMSE/MAE/R2 for vote-fraction) across all 13 scenarios, joined with the 95% CI from `07`'s repeated-split output. Output: `reports/all_models_metrics_table.csv` / `.md`. Rerun `07` before `08` if you've changed any scenario's data or models, so the CI columns stay in sync.

## Recommended follow-ups (not yet done)

1. Extend `05_compare_all_experiments.py`'s comparison chart to include all 13 scenarios, or add a parallel chart grouped by ground truth instead of by feature set.
2. Before adding a 14th scenario, update this file's matrix table and confirm which existing "base" split it should reuse — don't create a new independent split unless the ground truth itself is new. Also add it to `07_repeated_grouped_split_eval.py`'s `SCENARIOS` list so it gets a CI immediately rather than only a single-split point estimate.
3. Consider deleting `results_repeated_splits_scenarios_1_3_4_5_6/` now that `results_repeated_splits_all_scenarios/` fully supersedes it (same scenarios plus 7 more, 30 repeats instead of 10) — kept for now since removing it is a destructive action outside this pass's scope.
4. The 13 near-duplicated `load_and_clean`-style functions across the `03*` scripts (and now also inside `07`) are a standing maintenance risk — a future data-cleaning fix has to be found and applied in every copy. Worth consolidating into one shared module (e.g. `common_preprocessing.py`) in a dedicated pass.
