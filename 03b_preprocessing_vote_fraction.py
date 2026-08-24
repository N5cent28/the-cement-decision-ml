"""
Step 3B / Scenario 2: Leakage-Safe Preprocessing Pipeline (Vote Fraction, CT-only)
====================================================================================
This is the BASE preprocessing script for the vote-fraction ground truth
(scenario 2: CT-only features, regression target). Scenarios 12 and 13
(demographics-only and CT+demographics) reuse the exact patient split this
script produces — see `03l_build_vote_fraction_with_demographics_same_split.py`
and `03m_build_vote_fraction_demo_only_same_split.py`.

All cleaning rules, ground-truth definitions, split logic, imputation, and
feature engineering live in `common_preprocessing.py` — see that file's
module docstring for the full rationale. This script only decides: which
ground truth (`gt_vote_fraction`), which feature set (CT-only), and where
to write the output (`processed_data_vote_fraction/`).

Run:  python 03b_preprocessing_vote_fraction.py
"""

import json
import os

import common_preprocessing as cp

OUTPUT_DIR = "processed_data_vote_fraction"
TARGET_COL = "gt_vote_fraction"


def main():
    print("=" * 70)
    print("STEP 3B / SCENARIO 2: VOTE FRACTION, CT-ONLY")
    print("=" * 70 + "\n")

    df = cp.load_and_clean_raw()
    print(f"Cleaned base cohort: {len(df)} hips from {df['Anonymize_ID'].nunique()} patients.\n")

    df = df[df[TARGET_COL].notna()].copy()
    print(f"Usable for {TARGET_COL}: {len(df)} hips.")
    print(f"  Vote-fraction distribution: {df[TARGET_COL].value_counts().sort_index().to_dict()}\n")

    # gt_vote_fraction is continuous (in {0, 1/3, 2/3, 1}); is_regression=True
    # rounds it for stratification bins instead of casting to int.
    train, test, split_strategy = cp.grouped_split(df, target_col=TARGET_COL, is_regression=True)
    print(f"Train/test split (patient-grouped): {len(train)} train, {len(test)} test hips.")
    print(f"  Split stratification mode used: {split_strategy}\n")

    train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
    train = cp.engineer_ct_features(train)
    test = cp.engineer_ct_features(test)

    feature_cols = cp.ct_feature_list()
    train, test, scaler = cp.scale_features(train, test, feature_cols)
    print(f"Engineered + scaled {len(feature_cols)} CT features.\n")

    save_outputs(train, test, feature_cols, split_strategy)

    print("=" * 70)
    print("PREPROCESSING (VOTE FRACTION) COMPLETE — proceed to 04b_train_evaluate_vote_fraction.py")
    print("=" * 70)


def save_outputs(df_train, df_test, feature_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group", "agreement_category"]
    gt_cols = ["gt_original", "gt_majority", "gt_vote_fraction"]
    save_cols = id_cols + feature_cols + gt_cols

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    df_train[save_cols].to_csv(train_path, index=False)
    df_test[save_cols].to_csv(test_path, index=False)

    meta = {
        "analysis_variant": "vote_fraction",
        "primary_target": "gt_vote_fraction",
        "feature_columns": feature_cols,
        "ground_truth_columns": gt_cols,
        "test_size": cp.TEST_SIZE,
        "random_state": cp.RANDOM_STATE,
        "knn_neighbors": cp.KNN_NEIGHBORS,
        "n_train_rows": len(df_train),
        "n_test_rows": len(df_test),
        "n_train_patients": int(df_train["Anonymize_ID"].nunique()),
        "n_test_patients": int(df_test["Anonymize_ID"].nunique()),
        "split_stratification_mode": split_strategy,
    }
    meta_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved:\n  {train_path}\n  {test_path}\n  {meta_path}")


if __name__ == "__main__":
    main()
