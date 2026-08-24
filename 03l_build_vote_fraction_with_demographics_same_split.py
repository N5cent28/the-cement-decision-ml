"""
Scenario 13: Build Vote-Fraction + CT + Demographics Dataset (Same Split)
==========================================================================
Uses the exact patient split from `processed_data_vote_fraction/` (scenario
2) and appends demographic variables, so scenarios 2, 12, and 13 differ by
exactly one variable (feature set) with an identical ground truth and split.

Output:
  processed_data_vote_fraction_with_demographics_same_split/

Run: python 03l_build_vote_fraction_with_demographics_same_split.py
"""

import json
import os

import pandas as pd

import common_preprocessing as cp

BASE_SPLIT_DIR = "processed_data_vote_fraction"
OUTPUT_DIR = "processed_data_vote_fraction_with_demographics_same_split"


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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug.to_csv(train_out, index=False)
    test_aug.to_csv(test_out, index=False)

    feature_cols_aug = base_meta["feature_columns"] + cp.DEMO_FEATURES
    meta_out = {
        **base_meta,
        "analysis_variant": "scenario13_vote_fraction_ct_plus_demo",
        "base_split_source": BASE_SPLIT_DIR,
        "feature_columns": feature_cols_aug,
        "demographic_feature_columns": cp.DEMO_FEATURES,
        "notes": "Uses identical train/test rows as processed_data_vote_fraction; adds train-only-imputed/scaled demographics.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved augmented train set: {train_out} ({len(train_aug)} rows)")
    print(f"Saved augmented test set:  {test_out} ({len(test_aug)} rows)")


if __name__ == "__main__":
    main()
