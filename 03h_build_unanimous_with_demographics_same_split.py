"""
Scenario 9: Build Unanimous-Agreement + CT + Demographics Dataset (Same Split)
===============================================================================
Uses the exact patient split from `processed_data_unanimous_ct_only/` (scenario
7) and appends demographic variables, so scenarios 7, 8, and 9 differ by
exactly one variable (feature set) with an identical ground truth and split.

Outputs are written to:
  processed_data_unanimous_with_demographics_same_split/

Run: python 03h_build_unanimous_with_demographics_same_split.py
"""

import json
import os

import pandas as pd

import common_preprocessing as cp

BASE_SPLIT_DIR = "processed_data_unanimous_ct_only"
OUTPUT_DIR = "processed_data_unanimous_with_demographics_same_split"


def load_base_split():
    train = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "test.csv"))
    with open(os.path.join(BASE_SPLIT_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    return train, test, meta


def main():
    raw_clean = cp.load_and_clean_raw()
    train_base, test_base, base_meta = load_base_split()

    base_keys = set(train_base["Anonymize_ID"].astype(str) + "|" + train_base["hip_side"].astype(str)).union(
        set(test_base["Anonymize_ID"].astype(str) + "|" + test_base["hip_side"].astype(str))
    )
    raw_keys = set(raw_clean["split_key"])
    if not base_keys.issubset(raw_keys):
        missing = list(base_keys - raw_keys)[:5]
        raise ValueError(f"Base split keys not found in cleaned raw data. Example missing keys: {missing}")

    train_aug = cp.attach_demographics_by_split_key(train_base, raw_clean, cp.DEMO_FEATURES)
    test_aug = cp.attach_demographics_by_split_key(test_base, raw_clean, cp.DEMO_FEATURES)
    train_aug, test_aug = cp.impute_and_scale_demographics(train_aug, test_aug, cp.DEMO_FEATURES)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug.to_csv(train_out, index=False)
    test_aug.to_csv(test_out, index=False)

    feature_cols_aug = base_meta["feature_columns"] + cp.DEMO_FEATURES
    meta_out = {
        **base_meta,
        "analysis_variant": "scenario9_unanimous_ct_plus_demo",
        "base_split_source": BASE_SPLIT_DIR,
        "feature_columns": feature_cols_aug,
        "demographic_feature_columns": cp.DEMO_FEATURES,
        "notes": "Uses identical train/test rows as processed_data_unanimous_ct_only; adds train-only-imputed/scaled demographics.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved augmented train set: {train_out} ({len(train_aug)} rows)")
    print(f"Saved augmented test set:  {test_out} ({len(test_aug)} rows)")
    print("No original files were modified.")


if __name__ == "__main__":
    main()
