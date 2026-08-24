"""
Scenario 7 Preprocessing: Unanimous agreement (all 3 surgeons), CT features only.

Ground truth here is `gt_unanimous`, the TRUE 3-rater unanimous consensus
(original surgeon + Halldor + 3D all agree), which is distinct from
`gt_h3d_agree` used in scenario 5 (that target only requires the Halldor
and 3D raters to agree, ignoring the original surgeon). See
`common_preprocessing.py`'s module docstring for both definitions.

This script establishes the base patient-grouped split for the "unanimous"
ground-truth family; scenarios 8 (demo-only) and 9 (CT+demo) reuse this
same split — see `03h_build_unanimous_with_demographics_same_split.py` and
`03i_build_unanimous_demo_only_same_split.py`.

Run:
  python 03g_preprocessing_unanimous_ct_only.py
"""

import json
import os

import common_preprocessing as cp

OUTPUT_DIR = "processed_data_unanimous_ct_only"
TARGET_COL = "gt_unanimous"


def main():
    df = cp.load_and_clean_raw()
    df = df[df[TARGET_COL].notna()].copy()
    print(f"Unanimous (3/3) subset: {len(df)} hips from {df['Anonymize_ID'].nunique()} patients")

    train, test, split_strategy = cp.grouped_split(df, target_col=TARGET_COL)
    train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
    train = cp.engineer_ct_features(train)
    test = cp.engineer_ct_features(test)

    feature_cols = cp.ct_feature_list()
    train, test, scaler = cp.scale_features(train, test, feature_cols)

    save(train, test, feature_cols, split_strategy)
    print(f"Split stratification mode used: {split_strategy}")
    print(f"Saved scenario 7 processed data to {OUTPUT_DIR}")


def save(train, test, feat_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # After filtering to the unanimous-only subset, no NaNs remain in
    # TARGET_COL — cast to int so the saved CSV matches the classification
    # target's natural dtype (common_preprocessing carries it as float64
    # while it's still a NaN-able column on the full, unfiltered cohort).
    train = train.copy()
    test = test.copy()
    train[TARGET_COL] = train[TARGET_COL].astype(int)
    test[TARGET_COL] = test[TARGET_COL].astype(int)

    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group"]
    target_cols = [TARGET_COL]
    save_cols = id_cols + feat_cols + target_cols
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, columns=save_cols)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, columns=save_cols)

    meta = {
        "analysis_variant": "scenario7_unanimous_ct_only",
        "primary_target": TARGET_COL,
        "feature_columns": feat_cols,
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


if __name__ == "__main__":
    main()
