"""
Step 3 / Scenario 1: Leakage-Safe Preprocessing Pipeline (Majority Vote, CT-only)
==================================================================================
This is the BASE preprocessing script for the majority-vote ground truth
(scenario 1: CT-only features). Scenarios 3 and 4 (CT+demographics and
demographics-only) reuse the exact patient split this script produces —
see `03c_build_split_with_demographics.py` and
`03d_build_demo_only_same_split.py` — so all three majority-vote feature-set
variants are directly comparable.

All cleaning rules, ground-truth definitions, split logic, imputation, and
feature engineering live in `common_preprocessing.py`. See that file's
module docstring for the full rationale (leakage-safety, exclusion order,
what each `gt_*` column means). This script only decides: which ground
truth (`gt_majority`), which feature set (CT-only), and where to write the
output (`processed_data/`).

Run:  python 03_preprocessing.py
"""

import json
import os

import common_preprocessing as cp

OUTPUT_DIR = "processed_data"
TARGET_COL = "gt_majority"


def main():
    print("=" * 60)
    print("STEP 3 / SCENARIO 1: MAJORITY VOTE, CT-ONLY")
    print("=" * 60 + "\n")

    df = cp.load_and_clean_raw()
    print(f"Cleaned base cohort: {len(df)} hips from {df['Anonymize_ID'].nunique()} patients.\n")

    df = df[df[TARGET_COL].notna()].copy()
    print(f"Usable for {TARGET_COL}: {len(df)} hips.")
    print(f"  Class balance: {df[TARGET_COL].value_counts().to_dict()}\n")

    train, test, split_strategy = cp.grouped_split(df, target_col=TARGET_COL)
    print(f"Train/test split (patient-grouped): {len(train)} train, {len(test)} test hips.")
    print(f"  Split stratification mode used: {split_strategy}\n")

    train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
    train = cp.engineer_ct_features(train)
    test = cp.engineer_ct_features(test)

    feature_cols = cp.ct_feature_list()
    train, test, scaler = cp.scale_features(train, test, feature_cols)
    print(f"Engineered + scaled {len(feature_cols)} CT features.\n")

    save_outputs(train, test, feature_cols, split_strategy)

    print("=" * 60)
    print("PREPROCESSING COMPLETE — proceed to 04_train_evaluate.py")
    print("=" * 60)


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
        "feature_columns": feature_cols,
        "ct_base_features": cp.CT_FEATURE_COLS,
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
