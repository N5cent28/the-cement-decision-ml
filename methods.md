# Methods Document — THA Cement Decision ML Study

**Working title:** The Artificial Intelligence versus the Orthopaedic Surgeon in Decision-Making of Cemented or Uncemented Total Hip Arthroplasty

**Last updated:** 2026-03-05

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

Demographic variables (age, sex, weight, height, BMI) were **not** included as model features. `Anonymize_ID` was never used as a feature.

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

## 9. Planned Sensitivity Analyses

- Model performance on **unanimous-only** ground truth subset (3/3 surgeon agreement)
- Model performance on **majority-vote** full set
- Cohort-specific models (train/test within a single cohort)
- Impact of including vs. excluding "Questionable Quality" scans
- Comparison with models that include demographic features (age, sex, BMI) to quantify the marginal value of CT-only classification

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

### Repeated split robustness check

To reduce dependence on a single test split, repeated patient-grouped split evaluation was run for scenarios 1, 3, 4, 5, and 6 (`repeats=10`):
- Scenario 1: Logistic Regression median ROC-AUC **0.7569**
- Scenario 3: Logistic Regression median ROC-AUC **0.8220**
- Scenario 4: Logistic Regression median ROC-AUC **0.7230**
- Scenario 5: Random Forest median ROC-AUC **0.8176**
- Scenario 6: Logistic Regression median ROC-AUC **0.9167**

These repeated-split distributions provide a more stable estimate than any single hold-out run and are available in `results_repeated_splits_scenarios_1_3_4_5_6/`. Note that scenario 6's repeated-split median (0.9167) is also well above its corrected single-split figure (0.7963) reported above — this robustness check has not yet been extended to scenarios 7–13 and would be a natural next step before relying on any single-split number for the new scenarios.

