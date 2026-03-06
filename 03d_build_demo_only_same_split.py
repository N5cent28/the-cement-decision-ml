"""
Step 3D: Build Demographics-Only Dataset (Same Original Split)
==============================================================
Creates a new processed dataset that keeps the exact original majority-vote
train/test split, but contains only demographic features for model input.

Input:
  processed_data_demographics_same_split/

Output:
  processed_data_demographics_only_same_split/

Run: python 03d_build_demo_only_same_split.py
"""

import json
import os

import pandas as pd

SOURCE_DIR = "processed_data_demographics_same_split"
OUTPUT_DIR = "processed_data_demographics_only_same_split"

DEMO_FEATURES = [
    "patient_age_years",
    "sex_binary",
    "weight",
    "height",
    "BMI",
]

ID_COLS = ["Anonymize_ID", "hip_side", "Cohort_group", "agreement_category"]
GT_COLS = ["gt_original", "gt_majority", "gt_vote_fraction"]


def build_demo_only(df):
    cols = ID_COLS + DEMO_FEATURES + GT_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[cols].copy()


def main():
    print("=" * 68)
    print("STEP 3D: BUILD DEMOGRAPHICS-ONLY DATASET (SAME SPLIT)")
    print("=" * 68 + "\n")

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
        "analysis_variant": "majority_demographics_only_same_split",
        "feature_columns": DEMO_FEATURES,
        "demographic_feature_columns": DEMO_FEATURES,
        "ct_feature_columns": [],
        "notes": "Same row split as processed_data; demographics-only feature set.",
    }
    meta_out_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(meta_out_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved train: {train_out} ({len(train_demo)} rows)")
    print(f"Saved test:  {test_out} ({len(test_demo)} rows)")
    print(f"Saved meta:  {meta_out_path}\n")
    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()
