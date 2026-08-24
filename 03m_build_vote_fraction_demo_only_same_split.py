"""
Scenario 12: Build Vote-Fraction Demographics-Only Dataset (Same Split)
========================================================================
Keeps the exact patient split from scenario 13
(processed_data_vote_fraction_with_demographics_same_split), but retains
only demographic features for model input.

Output:
  processed_data_vote_fraction_demographics_only_same_split/

Run: python 03m_build_vote_fraction_demo_only_same_split.py
"""

import json
import os

import pandas as pd

SOURCE_DIR = "processed_data_vote_fraction_with_demographics_same_split"
OUTPUT_DIR = "processed_data_vote_fraction_demographics_only_same_split"

DEMO_FEATURES = ["patient_age_years", "sex_binary", "weight", "height", "BMI"]
ID_COLS = ["Anonymize_ID", "hip_side", "Cohort_group", "agreement_category"]
GT_COLS = ["gt_original", "gt_majority", "gt_vote_fraction"]


def build_demo_only(df):
    cols = ID_COLS + DEMO_FEATURES + GT_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[cols].copy()


def main():
    train = pd.read_csv(os.path.join(SOURCE_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(SOURCE_DIR, "test.csv"))
    with open(os.path.join(SOURCE_DIR, "preprocessing_metadata.json")) as f:
        source_meta = json.load(f)

    train_demo = build_demo_only(train)
    test_demo = build_demo_only(test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_demo.to_csv(train_out, index=False)
    test_demo.to_csv(test_out, index=False)

    meta_out = {
        **source_meta,
        "analysis_variant": "scenario12_vote_fraction_demographics_only",
        "feature_columns": DEMO_FEATURES,
        "demographic_feature_columns": DEMO_FEATURES,
        "ct_feature_columns": [],
        "notes": "Same row split as processed_data_vote_fraction; demographics-only feature set.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved train: {train_out} ({len(train_demo)} rows)")
    print(f"Saved test:  {test_out} ({len(test_demo)} rows)")


if __name__ == "__main__":
    main()
