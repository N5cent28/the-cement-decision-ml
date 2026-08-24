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
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FILE = "Raw_data_03.03.2026.csv"
BASE_SPLIT_DIR = "processed_data_original_ct_only"
OUTPUT_DIR = "processed_data_original_demographics_only_same_split"

DEMO_FEATURES = ["patient_age_years", "sex_binary", "weight", "height", "BMI"]


def parse_age_to_years(age_val):
    if pd.isna(age_val):
        return np.nan
    m = re.search(r"(\d+)", str(age_val))
    return float(m.group(1)) if m else np.nan


def load_raw_demographics():
    df = pd.read_csv(RAW_FILE)
    df["patient_age_years"] = df["patient_age"].apply(parse_age_to_years)
    df["sex_binary"] = df["sex"].map({"F": 0.0, "M": 1.0})
    df["split_key"] = df["Anonymize_ID"].astype(str) + "|" + df["hip_side"].astype(str)
    return df[["split_key", "patient_age_years", "sex_binary", "weight", "height", "BMI"]]


def load_base_split():
    train = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "test.csv"))
    with open(os.path.join(BASE_SPLIT_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    train["split_key"] = train["Anonymize_ID"].astype(str) + "|" + train["hip_side"].astype(str)
    test["split_key"] = test["Anonymize_ID"].astype(str) + "|" + test["hip_side"].astype(str)
    return train, test, meta


def attach_demographics(base_df, raw_demo):
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group", "split_key"]
    merged = base_df[id_cols + ["gt_original"]].merge(raw_demo, on="split_key", how="left", validate="one_to_one")
    missing = merged[DEMO_FEATURES].isna().all(axis=1).sum()
    if missing > 0:
        raise ValueError(f"Merge failed for {missing} rows: no demographics found by split_key.")
    return merged


def impute_and_scale(train_df, test_df):
    medians = train_df[["patient_age_years", "weight", "height", "BMI"]].median()
    sex_mode = train_df["sex_binary"].mode(dropna=True)
    sex_fill = sex_mode.iloc[0] if len(sex_mode) > 0 else 0.0
    for col in ["patient_age_years", "weight", "height", "BMI"]:
        train_df[col] = train_df[col].fillna(medians[col])
        test_df[col] = test_df[col].fillna(medians[col])
    train_df["sex_binary"] = train_df["sex_binary"].fillna(sex_fill)
    test_df["sex_binary"] = test_df["sex_binary"].fillna(sex_fill)

    scaler = StandardScaler()
    train_df[DEMO_FEATURES] = scaler.fit_transform(train_df[DEMO_FEATURES])
    test_df[DEMO_FEATURES] = scaler.transform(test_df[DEMO_FEATURES])
    return train_df, test_df


def main():
    raw_demo = load_raw_demographics()
    train_base, test_base, base_meta = load_base_split()

    train_aug = attach_demographics(train_base, raw_demo)
    test_aug = attach_demographics(test_base, raw_demo)
    train_aug, test_aug = impute_and_scale(train_aug, test_aug)

    save_cols = ["Anonymize_ID", "hip_side", "Cohort_group"] + DEMO_FEATURES + ["gt_original"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug[save_cols].to_csv(train_out, index=False)
    test_aug[save_cols].to_csv(test_out, index=False)

    meta_out = {
        **base_meta,
        "analysis_variant": "scenario11_original_demographics_only",
        "feature_columns": DEMO_FEATURES,
        "demographic_feature_columns": DEMO_FEATURES,
        "ct_feature_columns": [],
        "notes": "Same row split as processed_data_original_ct_only (matches scenario 6); demographics-only feature set.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved train: {train_out} ({len(train_aug)} rows)")
    print(f"Saved test:  {test_out} ({len(test_aug)} rows)")


if __name__ == "__main__":
    main()
