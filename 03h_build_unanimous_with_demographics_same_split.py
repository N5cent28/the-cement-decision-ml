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
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FILE = "Raw_data_03.03.2026.csv"
BASE_SPLIT_DIR = "processed_data_unanimous_ct_only"
OUTPUT_DIR = "processed_data_unanimous_with_demographics_same_split"

SURGEON_COLS = ["Cement_vs_noCement_original", "Halldor_decision", "3d_surg"]
LABEL_MAP = {
    "Cemented": 1,
    "cemented": 1,
    "Non-cemented": 0,
    "non-cemented": 0,
    "Uncemented": 0,
    "uncemented": 0,
}
EXCLUDED_LABELS = {"Already operated", "already operated"}

DEMO_FEATURES = ["patient_age_years", "sex_binary", "weight", "height", "BMI"]


def parse_age_to_years(age_val):
    if pd.isna(age_val):
        return np.nan
    m = re.search(r"(\d+)", str(age_val))
    return float(m.group(1)) if m else np.nan


def load_and_clean_raw():
    df = pd.read_csv(RAW_FILE)
    for col in SURGEON_COLS:
        df[col] = df[col].astype(str).str.strip()
    excluded = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded |= df[col].isin(EXCLUDED_LABELS)
    df = df[~excluded].copy()
    for col in SURGEON_COLS:
        df[col + "_bin"] = df[col].map(LABEL_MAP)
    df = df[df["notes"] != "Unusable with same HU range"].copy()

    df["patient_age_years"] = df["patient_age"].apply(parse_age_to_years)
    df["sex_binary"] = df["sex"].map({"F": 0.0, "M": 1.0})
    df["split_key"] = df["Anonymize_ID"].astype(str) + "|" + df["hip_side"].astype(str)
    return df


def load_base_split():
    train = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(BASE_SPLIT_DIR, "test.csv"))
    with open(os.path.join(BASE_SPLIT_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    train["split_key"] = train["Anonymize_ID"].astype(str) + "|" + train["hip_side"].astype(str)
    test["split_key"] = test["Anonymize_ID"].astype(str) + "|" + test["hip_side"].astype(str)
    return train, test, meta


def attach_demographics(base_df, raw_df):
    keep_cols = ["split_key", "patient_age_years", "sex_binary", "weight", "height", "BMI"]
    merged = base_df.merge(raw_df[keep_cols], on="split_key", how="left", validate="one_to_one")
    missing_demo = merged[DEMO_FEATURES].isna().all(axis=1).sum()
    if missing_demo > 0:
        raise ValueError(f"Merge failed for {missing_demo} rows: no demographics found by split_key.")
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
    raw_clean = load_and_clean_raw()
    train_base, test_base, base_meta = load_base_split()

    base_keys = set(train_base["split_key"]).union(set(test_base["split_key"]))
    raw_keys = set(raw_clean["split_key"])
    if not base_keys.issubset(raw_keys):
        missing = list(base_keys - raw_keys)[:5]
        raise ValueError(f"Base split keys not found in cleaned raw data. Example missing keys: {missing}")

    train_aug = attach_demographics(train_base, raw_clean)
    test_aug = attach_demographics(test_base, raw_clean)
    train_aug, test_aug = preprocess_demographics(train_aug, test_aug)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_out = os.path.join(OUTPUT_DIR, "train.csv")
    test_out = os.path.join(OUTPUT_DIR, "test.csv")
    train_aug.to_csv(train_out, index=False)
    test_aug.to_csv(test_out, index=False)

    feature_cols_aug = base_meta["feature_columns"] + DEMO_FEATURES
    meta_out = {
        **base_meta,
        "analysis_variant": "scenario9_unanimous_ct_plus_demo",
        "base_split_source": BASE_SPLIT_DIR,
        "feature_columns": feature_cols_aug,
        "demographic_feature_columns": DEMO_FEATURES,
        "notes": "Uses identical train/test rows as processed_data_unanimous_ct_only; adds train-only-imputed/scaled demographics.",
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"Saved augmented train set: {train_out} ({len(train_aug)} rows)")
    print(f"Saved augmented test set:  {test_out} ({len(test_aug)} rows)")
    print("No original files were modified.")


if __name__ == "__main__":
    main()
