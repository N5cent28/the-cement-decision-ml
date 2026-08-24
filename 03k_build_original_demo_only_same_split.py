"""
Scenario 11: Build Original-Surgeon Demographics-Only Dataset (Same Split)
===========================================================================
Uses the exact patient split from `processed_data_original_ct_only/`
(scenario 10, which itself matches scenario 6's split) and keeps only
demographic features, so scenarios 6, 10, and 11 differ by exactly one
variable (feature set) with an identical ground truth and split.

Output:
  processed_data_original_demographics_only_same_split/

Run: python 03k_build_original_demo_only_same_split.py
"""

import json
import os

import pandas as pd

import common_preprocessing as cp

BASE_SPLIT_DIR = "processed_data_original_ct_only"
OUTPUT_DIR = "processed_data_original_demographics_only_same_split"
TARGET_COL = "gt_original"


def load_raw_demographics():
    raw = cp.load_and_clean_raw()
    return raw[["split_key"] + cp.DEMO_FEATURES]


def load_base_split():
    train = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "test.csv"))
    with open(os.path.join(BASE_SPLIT_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    return train, test, meta


def main():
    raw_demo = load_raw_demographics()
    train_base, test_base, base_meta = load_base_split()

    train_aug = cp.attach_demographics_by_split_key(train_base, raw_demo, cp.DEMO_FEATURES)
    test_aug = cp.attach_demographics_by_split_key(test_base, raw_demo, cp.DEMO_FEATURES)
    train_aug, test_aug = cp.impute_and_scale_demographics(train_aug, test_aug, cp.DEMO_FEATURES)

    save_cols = ["Anonymize_ID", "hip_side", "Cohort_group"] + cp.DEMO_FEATURES + [TARGET_COL]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug[save_cols].to_csv(train_out, index=False)
    test_aug[save_cols].to_csv(test_out, index=False)

    meta_out = {
        **base_meta,
        "analysis_variant": "scenario11_original_demographics_only",
        "feature_columns": cp.DEMO_FEATURES,
        "demographic_feature_columns": cp.DEMO_FEATURES,
        "ct_feature_columns": [],
        "notes": "Same row split as processed_data_original_ct_only (matches scenario 6); demographics-only feature set.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved train: {train_out} ({len(train_aug)} rows)")
    print(f"Saved test:  {test_out} ({len(test_aug)} rows)")


if __name__ == "__main__":
    main()
