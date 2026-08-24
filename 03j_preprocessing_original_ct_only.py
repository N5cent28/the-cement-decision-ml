"""
Scenario 10 Preprocessing: Original operating surgeon's decision, CT features only.

This establishes the base patient-grouped split for the "original surgeon"
ground-truth family; scenario 11 (demo-only) reuses this same split. Scenario
6 (03f/04f, original surgeon + CT+demographics) is left as originally run —
it uses an independently computed split, but since it applies identical
cleaning, target, random_state, and stratification logic (all now shared via
`common_preprocessing.py`) to the same underlying dataframe, that split is
deterministically identical to this one (verified at runtime below).

Run:
  python 03j_preprocessing_original_ct_only.py
"""

import json
import os

import pandas as pd

import common_preprocessing as cp

OUTPUT_DIR = "processed_data_original_ct_only"
REFERENCE_SPLIT_DIR = "processed_data_scenario6_original_ct_plus_demo"
TARGET_COL = "gt_original"


def verify_matches_scenario6_split(train, test):
    """Confirm this split is identical to scenario 6's, so scenarios 6/10/11
    are a genuinely isolated feature-set comparison despite being built by
    separate scripts."""
    ref_train_path = os.path.join(REFERENCE_SPLIT_DIR, "train.csv")
    ref_test_path = os.path.join(REFERENCE_SPLIT_DIR, "test.csv")
    if not (os.path.exists(ref_train_path) and os.path.exists(ref_test_path)):
        print("  (Scenario 6 outputs not found yet — skipping split-consistency check.)")
        return
    ref_train = pd.read_csv(ref_train_path)
    ref_test = pd.read_csv(ref_test_path)
    this_train_ids = set(zip(train["Anonymize_ID"], train["hip_side"]))
    this_test_ids = set(zip(test["Anonymize_ID"], test["hip_side"]))
    ref_train_ids = set(zip(ref_train["Anonymize_ID"], ref_train["hip_side"]))
    ref_test_ids = set(zip(ref_test["Anonymize_ID"], ref_test["hip_side"]))
    if this_train_ids == ref_train_ids and this_test_ids == ref_test_ids:
        print("  Split matches scenario 6 (results_scenario6_original_ct_plus_demo) exactly.")
    else:
        print("  WARNING: split does NOT match scenario 6's split. "
              "Scenarios 6, 10, and 11 would not be directly comparable.")


def save(train, test, feat_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group"]
    target_cols = [TARGET_COL]
    save_cols = id_cols + feat_cols + target_cols
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, columns=save_cols)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, columns=save_cols)

    meta = {
        "analysis_variant": "scenario10_original_ct_only",
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


def main():
    df = cp.load_and_clean_raw()
    df = df[df[TARGET_COL].notna()].copy()

    train, test, split_strategy = cp.grouped_split(df, target_col=TARGET_COL)
    verify_matches_scenario6_split(train, test)

    train, test = cp.cohort_knn_impute(train, test, cp.CT_FEATURE_COLS)
    train = cp.engineer_ct_features(train)
    test = cp.engineer_ct_features(test)

    feature_cols = cp.ct_feature_list()
    train, test, scaler = cp.scale_features(train, test, feature_cols)

    save(train, test, feature_cols, split_strategy)
    print(f"Split stratification mode used: {split_strategy}")
    print(f"Saved scenario 10 processed data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
