# Methods Document — THA Cement Decision ML Study

**Working title:** The Artificial Intelligence versus the Orthopaedic Surgeon in Decision-Making of Cemented or Uncemented Total Hip Arthroplasty

**Last updated:** 2026-08-23

---

## 1. Study Design

Retrospective cohort study using preoperative CT scans from patients who underwent total hip arthroplasty (THA) at Landspítali University Hospital. The study develops supervised machine learning models to classify optimal THA implant fixation (cemented vs. non-cemented) using exclusively CT-derived imaging features.

## 2. Data Sources

### 2.1 Cohorts

| Cohort | Scanner | Period | Patients | Notes |
|--------|---------|--------|----------|-------|
| Cohort_2 | Toshiba | 2023–2024 | ~50 | Newer acquisitions, generally high quality |
| Cohort_1 | Philips | ~2013 | ~100 | Older acquisitions, more variable quality |

Both cohorts followed similar preoperative CT protocols (bilateral hips, pelvis, and proximal femur acquired one day before THA). Cohort_1 data originates from prior studies conducted by the research team at the University of Reykjavík and Landspítali.

### 2.2 Variables

**CT-derived imaging features (model inputs):**
- Gruen zone BMD values: zones 1–7 (`zone_1_bmd` through `zone_7_bmd`)
- Cortical geometry: `cortical_area_mm2`, `cortical_thickness_mm`
- Radial measurements: `avg_outer_radius_mm`, `ray_inner_radius_mm`, `geometric_inner_R_mm`, `inner_radius_std_mm`

**Engineered features:**
- Zone average pairs: `zones_1_7_avg` (mean of zones 1 and 7), `zones_2_6_avg` (mean of zones 2 and 6)
- BMD-to-cortical ratios: zone averages divided by cortical thickness and cortical area
- BMD summary statistics: mean, standard deviation, and range across all 7 zones

**Demographic variables (recorded but excluded from modeling):**
- Age, sex, weight, height, BMI

**Surgeon classifications:**
- `Cement_vs_noCement_original`: original operating surgeon's decision
- `Halldor_decision`: independent classification by Dr. Halldór Jónsson Jr.
- `3d_surg`: classification by a third surgeon reviewer

### 2.3 Unit of Analysis

Each hip (left/right) constitutes one observation. Patients with bilateral scans contribute two observations, but all hips from the same patient are always kept in the same data split (train or test) to prevent data leakage.

## 3. Ground Truth Definition

Three ground truth strategies were defined to examine the impact of label uncertainty:

1. **Original surgeon only:** The operating surgeon's preoperative decision serves as the label.
2. **Majority vote:** A hip is labeled "cemented" if at least 2 of 3 surgeons classified it as cemented; otherwise "non-cemented."
3. **Vote fraction (probabilistic):** The fraction of surgeons voting "cemented" (0, 0.33, 0.67, or 1.0) provides a soft label for potential probabilistic training experiments.

The primary analysis uses the **majority vote** label.

### 3.1 Inter-Rater Agreement

Before defining the ground truth, inter-surgeon agreement was quantified:
- **Pairwise percent agreement** for all three rater pairs
- **Fleiss' kappa** for the three-rater panel (binary classification)
- **Case-level categorization:** unanimous (3/3 agree), majority only (2/3 agree)

Label harmonization: "Uncemented" was mapped to "Non-cemented." Rows with the label "Already operated" were excluded from agreement analysis and modeling.

## 4. Data Quality & Exclusion Criteria

### 4.1 Scan Quality Flags

Scans annotated in the `notes` column as "Unusable with same HU range" were excluded. Scans flagged as "Questionable Quality" or "Questionable scan quality" were retained but their presence noted for sensitivity analysis.

### 4.2 Anomalous Value Detection

Rows were excluded if they contained CT feature values outside physiologically plausible ranges:
- Zone BMD values < 0.5 (artifact values, e.g. 0.14)
- Cortical area < 50 mm² (values such as 0.47, 3.96, 4.54 indicating failed segmentation)
- Cortical thickness < 2.0 mm (values such as 0.74, 1.24 indicating failed segmentation)

## 5. Preprocessing Pipeline

All preprocessing was performed in a leakage-safe manner: imputation and scaling parameters were fitted exclusively on the training set.

### 5.1 Train/Test Split

- **Split ratio:** 80% train / 20% test
- **Grouping:** Patient-level (`Anonymize_ID`) — all hips from a patient stay in the same split
- **Stratification:** Patient-level stratified split by target + cohort where feasible, with explicit fallback order: `target_plus_cohort` -> `target_only_fallback` -> `unstratified_fallback`
- **When reusing the “same split” in later analyses:** this refers to the same patient/hip membership in train and test (not just another 80/20 split). Matching was verified at the row key level (`Anonymize_ID` + `hip_side`).

The split mode actually used is recorded in each preprocessing metadata file as `split_stratification_mode`, and repeated-split experiments aggregate usage in `results_repeated_splits_scenarios_1_3_4_5_6/split_stratification_usage.csv`.

### 5.2 Missing Value Imputation

K-nearest neighbors (KNN) imputation with `k=5` and distance-weighted averaging was applied **separately by cohort**:
- Cohort_2 missing values were imputed using only other Cohort_2 training observations
- Cohort_1 missing values were imputed using only other Cohort_1 training observations

This prevents cross-cohort contamination given known distributional differences between scanner manufacturers (Toshiba vs. Philips).

Imputers were fitted on training data only; the same fitted imputers were applied to the corresponding cohort partitions in the test set.

### 5.3 Feature Engineering

After imputation, the following derived features were computed:
1. `zones_1_7_avg`: mean of zone 1 and zone 7 BMD
2. `zones_2_6_avg`: mean of zone 2 and zone 6 BMD
3. Ratios: each zone average divided by `cortical_thickness_mm` and `cortical_area_mm2`
4. Summary statistics: mean, standard deviation, and range of zones 1–7 BMD

### 5.4 Feature Scaling

StandardScaler (zero mean, unit variance) was fitted on the training set and applied to both train and test sets.

### 5.5 Total Feature Count

13 base CT features + 9 engineered features = **22 features** used as model inputs.

## 6. Machine Learning Models

Four classifiers were trained and compared:

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | solver=lbfgs, max_iter=2000 |
| Random Forest | n_estimators=200, min_samples_leaf=3 |
| Gradient Boosting | n_estimators=200, max_depth=3, learning_rate=0.1 |
| SVM (RBF kernel) | probability=True, default C and gamma |

Demographic variables (age, sex, weight, height, BMI) were excluded from the baseline scenario (1) to establish a CT-only reference point, but are included as model features in scenarios 3, 4, 6, and 7–13 to quantify their marginal and standalone contribution (see "Comparative sensitivity analyses" below). `Anonymize_ID` was never used as a feature in any scenario.

## 7. Model Evaluation

### 7.1 Cross-Validation

Grouped 5-fold cross-validation was used during training, with patient ID as the grouping variable, to obtain unbiased performance estimates and prevent leakage.

### 7.2 Test Set Metrics

The following metrics were computed on the held-out test set:
- ROC-AUC
- PR-AUC (precision-recall area under curve, important if class imbalance exists)
- Accuracy
- Precision
- Recall (sensitivity)
- F1 score
- Confusion matrix

### 7.3 Cohort-Specific Evaluation

All test metrics were also reported separately for Cohort_1 and Cohort_2 to assess the impact of cohort/scanner differences on model performance.

### 7.4 Feature Importance

For tree-based models (Random Forest, Gradient Boosting), feature importance scores were extracted to identify which CT-derived measurements contribute most to the classification.

## 8. Software & Reproducibility

- **Language:** Python 3.x
- **Key libraries:** pandas, numpy, scikit-learn, scipy, statsmodels, matplotlib, seaborn
- **Random seed:** 42 (used consistently across all stochastic operations)
- All pipeline steps are implemented as sequential scripts (`01_data_audit.py` through `04_train_evaluate.py`) that produce intermediate outputs to enable inspection at each stage.

## 9. Sensitivity Analyses: Status

- Model performance on **unanimous-only** ground truth subset (3/3 surgeon agreement) — **done** (scenarios 7–9; see "Comparative sensitivity analyses").
- Model performance on **majority-vote** full set — **done** (scenarios 1, 3, 4; the primary analysis).
- Comparison with models that include demographic features (age, sex, BMI) to quantify the marginal value of CT-only classification — **done** (scenarios 3, 4, 6, 8, 9, 11, 12, 13).
- Cohort-specific models (train/test within a single cohort, i.e. a Philips-only model and a Toshiba-only model trained independently) — **not done / out of scope for this revision.** What exists today (Section 7.3) is weaker: the pooled model (trained on both cohorts) evaluated separately on each cohort's test hips. That does not tell us whether a model trained exclusively within one cohort would perform differently — it only tells us how the pooled model generalizes to each cohort's test subset. True cohort-specific training would also roughly halve the training set size (n≈70–100 patients per cohort), which is a real constraint worth discussing with the team before committing to it.
- Impact of including vs. excluding "Questionable Quality" scans — **not done / out of scope for this revision.** Scans flagged "Questionable Quality" or "Questionable scan quality" are currently always retained (Section 4.1); no script re-runs the pipeline with them excluded to test whether they change results. Would need a new preprocessing variant that filters on the `notes` column before the split.

## 10. Ethical Considerations

- All patient data were de-identified prior to analysis (`Anonymize_ID` used as the only identifier)
- CT scans were acquired as part of standard preoperative protocols; participation did not influence surgeon decisions
- Approved by the Ethics Committee of Landspítali (reference 33_2033) and the Scientific Research Committee (VRN LSH 230530)
- All patients provided signed informed consent for use of their CT scans in prosthetic joint research

---

## Revision Log

| Date | Change |
|------|--------|
| 2026-03-03 | Initial methods document created. Pipeline steps 1–4 implemented. |
| 2026-03-05 | Pipeline steps b-d implemented. |
| 2026-03-05 | Added scenarios 5–6, split-stratification logging, and repeated-split robustness analysis. |
| 2026-08-23 | Added scenarios 7–13 to complete the 3×3 ground-truth × feature-set sensitivity matrix (unanimous agreement, original-surgeon-only, and vote-fraction targets each now tested against CT-only, demographics-only, and CT+demographics). Corrected scenario 5/6 prose figures to match committed result files, fixed a stale image reference, repointed the feature-importance script at current (non-stale) result directories, and removed orphaned pre-split-fix result directories (`results_majority/`, `results_majority_CT+Demographics/`, `results_majority_demographics_only/`). See `SCENARIO_INDEX.md`. |
| 2026-08-23 | Added precision/recall/F1 to the vote-fraction scenarios (2, 12, 13), which previously reported only regression metrics. Added a consolidated all-scenarios metrics table (`08_build_all_models_metrics_table.py`, `reports/all_models_metrics_table.md`). Extended the repeated-split robustness check to all 13 scenarios with 30 repeats and 95% confidence intervals (`07_repeated_grouped_split_eval.py` → `results_repeated_splits_all_scenarios/`), joined into the master metrics table. Corrected a stale claim that demographics were excluded from all models, and marked the "cohort-specific models" and "questionable-quality-scan exclusion" sensitivity analyses explicitly as not done. |

---

## Discussion

This analysis continues to support the central premise that CT-derived quantitative features carry clinically useful signal for fixation decision support, while also showing that performance estimates are sensitive to label definition and split selection.

Inter-rater agreement remained limited (Fleiss' kappa = 0.2367, fair agreement), which means any supervised target built from surgeon labels carries irreducible uncertainty. This motivates maintaining multiple label scenarios and reporting uncertainty explicitly.

After quality filtering and anomaly exclusion, 200 hips from 111 patients were retained. For majority-vote classification with corrected patient-level stratified split handling, test performance changed versus earlier runs:
- Scenario 1 (majority, CT-only): best test ROC-AUC = **0.7368** (Random Forest), best accuracy = **0.6977** (SVM).
- Scenario 3 (majority, CT + demographics): best test ROC-AUC = **0.7566** (SVM).
- Scenario 4 (majority, demographics-only): best test ROC-AUC = **0.7061** (Logistic Regression).

These updated results suggest CT + demographics provides the strongest discrimination among majority-vote feature-set variants, but the margin over CT-only is modest in a single split.

![Original majority-vote ROC curves](results/figures/roc_curves.png)

Scenario 5 (Halldor+3D agreement-only, CT-only) achieved best test ROC-AUC **0.8860** (Random Forest) on the currently committed split (train 119, test 31). Scenario 6 (original-surgeon target, CT+demographics) achieved best test ROC-AUC **0.7963** (Logistic Regression, train 161, test 39).

> **Note on corrected figures (2026-08-23):** the values above for scenarios 5 and 6 replace earlier prose figures in this document (0.6944 and 0.9926 respectively, with train/test counts of 121/29) that did not match the `model_results.json` files actually committed to this repository. The cause of that drift was not tracked down (it predates this revision and the underlying scripts have not changed), but the numbers above are reproducible directly from `results_scenario5_agree_ct_only/model_results.json` and `results_scenario6_original_ct_plus_demo/model_results.json` by re-running `04e_train_evaluate_agree_ct_only.py` / `04f_train_evaluate_original_ct_plus_demo.py`. Treat the committed result files as the source of truth going forward rather than prose summaries.

The vote-fraction analysis remains a distinct regression task; best RMSE = **0.3173** (Linear Regression, R2 = 0.1060) for CT-only features. These metrics are not directly interchangeable with majority-vote ROC-AUC.

### Comparative sensitivity analyses (scenarios 1–13)

An earlier revision of this document reported six scenarios (1–6) that varied ground truth and feature set together, but did not hold each ground truth fixed across all three feature sets (CT-only / demographics-only / CT+demographics). In particular, the "agreement" scenario (5) used 2-of-3 rater agreement (Halldor + 3D surgeon only, ignoring the original surgeon), not true 3-rater unanimity, and the "original surgeon" scenario (6) changed both ground truth and feature set at once relative to the scenario-1 baseline, so its result could not be attributed to either change in isolation. Scenarios 7–13 close these gaps so that every ground truth is now tested against all three feature sets, changing exactly one variable per comparison:

| # | Ground truth | Feature set | Same split as | Best model (test) | Best test ROC-AUC / RMSE |
|---|---|---|---|---|---|
| 1 | Majority vote (≥2/3) | CT-only | — (baseline) | Random Forest | ROC-AUC 0.7368 |
| 3 | Majority vote (≥2/3) | CT + demographics | Scenario 1 | SVM (RBF) | ROC-AUC 0.7566 |
| 4 | Majority vote (≥2/3) | Demographics-only | Scenario 1 | Logistic Regression | ROC-AUC 0.7061 |
| 2 | Vote fraction (continuous) | CT-only | — (baseline) | Linear Regression | RMSE 0.3173 |
| 13 | Vote fraction (continuous) | CT + demographics | Scenario 2 | Linear Regression | RMSE 0.2875 |
| 12 | Vote fraction (continuous) | Demographics-only | Scenario 2 | Gradient Boosting Regressor | RMSE 0.3359 |
| 5 | Halldor+3D agree (2/3, excludes original surgeon) | CT-only | — (own split) | Random Forest | ROC-AUC 0.8860 |
| 7 | Unanimous (3/3 surgeons agree) | CT-only | — (baseline) | Logistic Regression | ROC-AUC 0.9773 |
| 9 | Unanimous (3/3 surgeons agree) | CT + demographics | Scenario 7 | Logistic Regression | ROC-AUC 0.9432 |
| 8 | Unanimous (3/3 surgeons agree) | Demographics-only | Scenario 7 | SVM (RBF) | ROC-AUC 0.9091 |
| 10 | Original surgeon only | CT-only | — (baseline; verified identical to scenario 6's split) | SVM (RBF) | ROC-AUC 0.5503 |
| 6 | Original surgeon only | CT + demographics | Scenario 10 | Logistic Regression | ROC-AUC 0.7963 |
| 11 | Original surgeon only | Demographics-only | Scenario 10 | Logistic Regression | ROC-AUC 0.8095 |

Notes on interpretation:
- Scenario 5 (2-rater "agree" subset, kept for continuity with prior reporting) and scenarios 7–9 (true 3-rater unanimity, the more defensible "all surgeons agree" definition) both show high apparent discrimination, but on very small held-out sets (test n=31 and n=19 respectively) drawn from a subset of hips where surgeons already agreed — these estimates carry wide uncertainty and are optimistic relative to the full-cohort majority-vote scenarios.
- Scenario 10 (original surgeon, CT-only, test ROC-AUC 0.55, near chance) versus scenario 6 (original surgeon, CT+demographics, 0.80) versus scenario 11 (original surgeon, demographics-only, 0.81) now isolates the effect cleanly: demographics, not CT features, are doing essentially all of the work in predicting the original surgeon's real-time decision. This confirms the caution flagged in the prior revision — the original-surgeon target is likely encoding demographic-driven surgeon heuristics (e.g., age- or weight-based rules of thumb) rather than bone-quality signal.
- Preprocessing scripts: scenarios 7/8/9 are built by `03g`/`03h`/`03i` (+ `04g`/`04h`/`04i`); scenarios 10/11 by `03j`/`03k` (+ `04j`/`04k`); scenarios 12/13 by `03l`/`03m` (+ `04l`/`04m`). See `SCENARIO_INDEX.md` for the full script-to-scenario mapping and output directories.

![Comparison across majority-vote feature-set experiments](results_comparison_all_experiments/figures/majority_experiments_comparison.png)

Best-model feature-importance outputs for scenarios 1, 3, 4, 5, 6, 7, 8, 9, 10, and 11 (all classification scenarios) are stored in `results_feature_importance_best_model/`, with per-scenario CSV rankings and top-feature plots indexed in `best_model_feature_importance_summary.csv`. Vote-fraction scenarios (2, 12, 13) are regression tasks and are out of scope for that script.

### Repeated split robustness check and 95% confidence intervals

A single train/test split is a noisy estimate of performance, particularly with test sets as small as 19–43 hips — one misclassified hip can move an AUC by several points. To quantify that noise, `07_repeated_grouped_split_eval.py` now re-runs the patient-grouped split **30 times** (previously 10, and previously only for scenarios 1/3/4/5/6) for **all 13 scenarios**, retrains every model on each resplit, and reports the resulting distribution — including an empirical 95% confidence interval (2.5th–97.5th percentile across the 30 repeats) — rather than a single point estimate. Full results: `results_repeated_splits_all_scenarios/repeated_split_metrics_summary.csv` (per-scenario ROC-AUC boxplots in `figures/`).

Best model per scenario by mean repeated-split ROC-AUC, with 95% CI:

| # | Scenario | Best model (by mean) | Mean ROC-AUC | 95% CI |
|---|---|---|---|---|
| 1 | Majority, CT-only | Logistic Regression | 0.7102 | [0.560, 0.884] |
| 2 | Vote fraction, CT-only | Random Forest Regressor | 0.6712 | [0.530, 0.781] |
| 3 | Majority, CT+demo | Logistic Regression | 0.7424 | [0.555, 0.900] |
| 4 | Majority, demo-only | Logistic Regression | 0.6656 | [0.461, 0.891] |
| 5 | Halldor+3D agree (2/3), CT-only | Random Forest | 0.8247 | [0.660, 0.932] |
| 6 | Original surgeon, CT+demo | Logistic Regression | 0.8920 | [0.808, 0.986] |
| 7 | Unanimous (3/3), CT-only | Logistic Regression | 0.8468 | [0.641, 1.000] |
| 8 | Unanimous (3/3), demo-only | Logistic Regression | 0.9588 | [0.833, 1.000] |
| 9 | Unanimous (3/3), CT+demo | Logistic Regression | 0.9654 | [0.812, 1.000] |
| 10 | Original surgeon, CT-only | Logistic Regression | 0.5850 | [0.365, 0.809] |
| 11 | Original surgeon, demo-only | Logistic Regression | 0.9214 | [0.845, 1.000] |
| 12 | Vote fraction, demo-only | Linear Regression | 0.6374 | [0.465, 0.822] |
| 13 | Vote fraction, CT+demo | Linear Regression | 0.6993 | [0.538, 0.857] |

Two things this table changes about how the point estimates elsewhere in this document should be read:

- **The CIs are wide relative to the differences the single-split numbers imply.** For example, scenario 1's single-split best AUC (0.7368) and scenario 3's (0.7566) look like CT+demographics modestly beats CT-only — but their repeated-split CIs, [0.560, 0.884] and [0.555, 0.900], overlap almost entirely. None of the feature-set or ground-truth comparisons in this document should be read as a statistically confirmed difference without checking that the CIs are actually disjoint; most are not.
- **The near-ceiling numbers for scenarios 8, 9, and 11 are the least trustworthy, not the most impressive.** Their means (0.92–0.97) look like the strongest results in the whole study, but their lower CI bounds (0.81–0.85) show genuinely wide uncertainty on n=19–39 test hips, and their upper bounds pinning at 1.000 is itself a sign of a small, easily-saturated test set rather than a robust ceiling effect. Scenario 10 shows the opposite, useful pattern: a tight-ish CI [0.365, 0.809] centered on chance, consistent with the single-split finding that CT-only carries little signal for the original surgeon's decision once demographics are held out.

One reassuring pattern: Logistic Regression is the best-by-mean model in 10 of 13 scenarios, which somewhat mitigates (but does not eliminate) the "best of 4 models on the test set" multiple-comparisons concern — the winner is not bouncing unpredictably between model families run to run.

