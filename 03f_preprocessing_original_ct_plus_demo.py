"""
Scenario 6 Preprocessing: Original surgeon target, CT + demographics.

This scenario's split is independently computed (not derived from another
scenario's output file), but because it applies the same cleaning, target,
random_state, and stratification logic (all shared via
`common_preprocessing.py`) to the same underlying dataframe as
`03j_preprocessing_original_ct_only.py` (scenario 10), the two splits are
deterministically identical — verified at runtime in `03j`.

Run:
  python 03f_preprocessing_original_ct_plus_demo.py
"""

import json
import os

import common_preprocessing as cp

OUTPUT_DIR = "processed_data_scenario6_original_ct_plus_demo"
TARGET_COL = "gt_original"


def all_features():
    return cp.ct_feature_list() + cp.DEMO_FEATURES


def save(train, test, feat_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group"]
    target_cols = ["gt_original", "gt_majority"]
    save_cols = id_cols + feat_cols + target_cols
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, columns=save_cols)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, columns=save_cols)

    meta = {
        "analysis_variant": "scenario6_original_ct_plus_demo",
        "primary_target": TARGET_COL,
        "feature_columns": feat_cols,
        "demographic_feature_columns": cp.DEMO_FEATURES,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_patients": int(train["Anonymize_ID"].nunique()),
        "n_test_patients": int(test["Anonymize_ID"].nunique()),
        "test_size": cp.TEST_SIZE,
        "random_state": cp.RANDOM_STATE,
        "split_stratification_mode": split_strategy,
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    df = cp.load_and_clean_raw()
    df = df[df[TARGET_COL].notna()].copy()

    train, test, split_strategy = cp.grouped_split(df, target_col=TARGET_COL)

    train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
    train = cp.engineer_ct_features(train)
    test = cp.engineer_ct_features(test)
    train, test = cp.impute_and_scale_demographics(train, test, cp.DEMO_FEATURES)

    feat_cols = all_features()
    # CT features were already scaled by cohort_knn_impute's downstream
    # engineering step needing raw values; scale them now (demographics were
    # already scaled by impute_and_scale_demographics above).
    train, test, _ = cp.scale_features(train, test, cp.ct_feature_list())

    save(train, test, feat_cols, split_strategy)
    print(f"Split stratification mode used: {split_strategy}")
    print(f"Saved scenario 6 processed data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
