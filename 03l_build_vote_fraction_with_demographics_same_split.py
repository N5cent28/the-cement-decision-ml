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
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FILE = "Raw_data_03.03.2026.csv"
BASE_SPLIT_DIR = "processed_data_vote_fraction"
OUTPUT_DIR = "processed_data_vote_fraction_with_demographics_same_split"

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
    merged = base_df.merge(raw_demo, on="split_key", how="left", validate="one_to_one")
    missing = merged[DEMO_FEATURES].isna().all(axis=1).sum()
    if missing > 0:
        raise ValueError(f"Merge failed for {missing} rows: no demographics found by split_key.")
    return merged


def preprocess_demographics(train_df, test_df):
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
    train_aug, test_aug = preprocess_demographics(train_aug, test_aug)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug.to_csv(train_out, index=False)
    test_aug.to_csv(test_out, index=False)

    feature_cols_aug = base_meta["feature_columns"] + DEMO_FEATURES
    meta_out = {
        **base_meta,
        "analysis_variant": "scenario13_vote_fraction_ct_plus_demo",
        "base_split_source": BASE_SPLIT_DIR,
        "feature_columns": feature_cols_aug,
        "demographic_feature_columns": DEMO_FEATURES,
        "notes": "Uses identical train/test rows as processed_data_vote_fraction; adds train-only-imputed/scaled demographics.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved augmented train set: {train_out} ({len(train_aug)} rows)")
    print(f"Saved augmented test set:  {test_out} ({len(test_aug)} rows)")


if __name__ == "__main__":
    main()
