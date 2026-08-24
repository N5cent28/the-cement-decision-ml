"""
Step 3C / Scenario 3: Build Demographics-Augmented Dataset (Same Original Split)
===================================================================================
Uses the exact existing split from `processed_data/train.csv` and
`processed_data/test.csv` (scenario 1's split) and appends demographic
variables, so scenarios 1, 3, and 4 differ by exactly one variable (feature
set) with an identical ground truth and split.

Demographics added: patient_age (parsed numeric years), sex (binary), weight,
height, BMI. See `common_preprocessing.py` for the shared cleaning/imputation
logic this script builds on.

Outputs are written to a separate folder: processed_data_demographics_same_split/

Run: python 03c_build_split_with_demographics.py
"""

import json
import os

import pandas as pd

import common_preprocessing as cp

BASE_SPLIT_DIR = "processed_data"
OUTPUT_DIR = "processed_data_demographics_same_split"


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

    # Keep the raw patient_age/sex strings alongside the numeric transforms,
    # for provenance/inspection, matching the original script's output schema.
    demo_cols_with_raw = ["patient_age", "patient_age_years", "sex", "sex_binary", "weight", "height", "BMI"]
    train_aug = cp.attach_demographics_by_split_key(train_base, raw_clean, demo_cols_with_raw)
    test_aug = cp.attach_demographics_by_split_key(test_base, raw_clean, demo_cols_with_raw)
    train_aug, test_aug = cp.impute_and_scale_demographics(train_aug, test_aug, cp.DEMO_FEATURES)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug.to_csv(train_out, index=False)
    test_aug.to_csv(test_out, index=False)

    feature_cols_aug = base_meta["feature_columns"] + cp.DEMO_FEATURES
    meta_out = {
        **base_meta,
        "analysis_variant": "majority_with_demographics_same_split",
        "base_split_source": BASE_SPLIT_DIR,
        "feature_columns": feature_cols_aug,
        "demographic_feature_columns": cp.DEMO_FEATURES,
        "notes": "Uses identical train/test rows as processed_data; adds train-only-imputed/scaled demographics.",
    }
    meta_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved augmented train set: {train_out} ({len(train_aug)} rows)")
    print(f"Saved augmented test set:  {test_out} ({len(test_aug)} rows)")
    print(f"Saved metadata:            {meta_path}\n")
    print("No original files were modified.")


if __name__ == "__main__":
    main()
