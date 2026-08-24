"""
Scenario 5 Preprocessing: Halldor+3D agreement only (2/3 raters), CT-only.

IMPORTANT: `gt_h3d_agree` requires only Halldor and the 3D surgeon to agree
with each other — the original surgeon's vote is NOT required to match.
This is 2-of-3 agreement, not 3-way unanimity. The true "all 3 surgeons
agree" ground truth is `gt_unanimous` (scenarios 7-9, see
`03g_preprocessing_unanimous_ct_only.py`). This scenario is kept under its
original name for continuity with prior reporting — see
`common_preprocessing.py`'s module docstring for why both targets exist.

All cleaning rules, ground-truth definitions, split logic, imputation, and
feature engineering live in `common_preprocessing.py`. This script only
decides: which ground truth (`gt_h3d_agree`), which feature set (CT-only),
and where to write the output.

Run:
  python 03e_preprocessing_agree_ct_only.py
"""

import json
import os

import common_preprocessing as cp

OUTPUT_DIR = "processed_data_scenario5_agree_ct_only"
TARGET_COL = "gt_h3d_agree"


def main():
    df = cp.load_and_clean_raw()
    df = df[df[TARGET_COL].notna()].copy()
    print(f"Halldor+3D agreement (2/3) subset: {len(df)} hips from {df['Anonymize_ID'].nunique()} patients")

    train, test, split_strategy = cp.grouped_split(df, target_col=TARGET_COL)
    train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
    train = cp.engineer_ct_features(train)
    test = cp.engineer_ct_features(test)

    feature_cols = cp.ct_feature_list()
    train, test, scaler = cp.scale_features(train, test, feature_cols)

    save(train, test, feature_cols, split_strategy)
    print(f"Split stratification mode used: {split_strategy}")
    print(f"Saved scenario 5 processed data to {OUTPUT_DIR}")


def save(train, test, feat_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # After filtering to the agreement-only subset, no NaNs remain in
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
        "analysis_variant": "scenario5_h3d_agreement_ct_only",
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
