"""
Step 1: Data Audit
==================
Load the raw CSV, inspect columns/types, quantify missingness,
check class balance across all three surgeon labels, flag anomalous
values, and report cohort-level summaries.

Run:  python 01_data_audit.py
"""

import pandas as pd
import numpy as np
import os

DATA_FILE = "Raw_data_03.03.2026.csv"
REPORT_DIR = "reports"

FEATURE_COLS = [
    "zone_1_bmd", "zone_2_bmd", "zone_3_bmd", "zone_4_bmd",
    "zone_5_bmd", "zone_6_bmd", "zone_7_bmd",
    "cortical_area_mm2", "cortical_thickness_mm",
    "avg_outer_radius_mm", "ray_inner_radius_mm",
    "geometric_inner_R_mm", "inner_radius_std_mm",
]

SURGEON_COLS = [
    "Cement_vs_noCement_original",
    "Halldor_decision",
    "3d_surg",
]

DEMOGRAPHIC_COLS = ["patient_age", "sex", "weight", "height", "BMI"]

# Thresholds for flagging clearly anomalous imaging values
BMD_ANOMALY_THRESHOLD = 0.5       # zone BMDs near 0.14 are artifacts
CORTICAL_AREA_MIN = 50.0          # areas below this (e.g. 0.47, 3.96) are bad scans
CORTICAL_THICKNESS_MIN = 2.0


def load_data():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {DATA_FILE}: {df.shape[0]} rows x {df.shape[1]} columns\n")
    return df


def print_column_overview(df):
    print("=" * 60)
    print("COLUMN OVERVIEW")
    print("=" * 60)
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct = 100 * n_miss / len(df)
        print(f"  {col:<30s}  dtype={str(df[col].dtype):<10s}  "
              f"missing={n_miss} ({pct:.1f}%)")
    print()


def print_patient_summary(df):
    print("=" * 60)
    print("PATIENT & HIP SUMMARY")
    print("=" * 60)
    n_patients = df["Anonymize_ID"].nunique()
    hips_per_patient = df.groupby("Anonymize_ID").size()
    print(f"  Unique patients (Anonymize_ID):  {n_patients}")
    print(f"  Hips per patient distribution:")
    print(f"    {hips_per_patient.value_counts().sort_index().to_dict()}")
    print(f"  Total hip-level rows:            {len(df)}")
    print()

    print("  Cohort distribution:")
    for cohort, cnt in df["Cohort_group"].value_counts().items():
        n_pat = df.loc[df["Cohort_group"] == cohort, "Anonymize_ID"].nunique()
        print(f"    {cohort}: {cnt} hips from {n_pat} patients")
    print()


def print_class_balance(df):
    print("=" * 60)
    print("SURGEON LABEL DISTRIBUTIONS")
    print("=" * 60)
    for col in SURGEON_COLS:
        print(f"\n  {col}:")
        vc = df[col].value_counts(dropna=False)
        for label, cnt in vc.items():
            print(f"    {str(label):<25s}  {cnt:>4d}  ({100*cnt/len(df):.1f}%)")
    print()


def flag_anomalies(df):
    print("=" * 60)
    print("ANOMALY FLAGS")
    print("=" * 60)

    # BMD anomalies (values suspiciously low, e.g. 0.14)
    bmd_cols = [c for c in FEATURE_COLS if "bmd" in c]
    for col in bmd_cols:
        mask = df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD)
        if mask.any():
            bad = df.loc[mask, ["Anonymize_ID", "hip_side", col, "notes"]]
            print(f"\n  {col} < {BMD_ANOMALY_THRESHOLD} ({mask.sum()} rows):")
            print(bad.to_string(index=False))

    # Cortical area anomalies
    mask = df["cortical_area_mm2"].notna() & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    if mask.any():
        bad = df.loc[mask, ["Anonymize_ID", "hip_side", "cortical_area_mm2", "notes"]]
        print(f"\n  cortical_area_mm2 < {CORTICAL_AREA_MIN} ({mask.sum()} rows):")
        print(bad.to_string(index=False))

    # Cortical thickness anomalies
    mask = (df["cortical_thickness_mm"].notna()
            & (df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN))
    if mask.any():
        bad = df.loc[mask, ["Anonymize_ID", "hip_side", "cortical_thickness_mm", "notes"]]
        print(f"\n  cortical_thickness_mm < {CORTICAL_THICKNESS_MIN} ({mask.sum()} rows):")
        print(bad.to_string(index=False))

    print()


def print_notes_summary(df):
    print("=" * 60)
    print("NOTES COLUMN SUMMARY")
    print("=" * 60)
    vc = df["notes"].value_counts(dropna=False)
    for label, cnt in vc.items():
        print(f"  {str(label):<40s}  {cnt:>3d}")
    print()


def print_missingness_by_cohort(df):
    print("=" * 60)
    print("FEATURE MISSINGNESS BY COHORT")
    print("=" * 60)
    for cohort in sorted(df["Cohort_group"].unique()):
        sub = df[df["Cohort_group"] == cohort]
        print(f"\n  {cohort} ({len(sub)} rows):")
        for col in FEATURE_COLS:
            n_miss = sub[col].isna().sum()
            if n_miss > 0:
                pct = 100 * n_miss / len(sub)
                print(f"    {col:<30s}  missing={n_miss} ({pct:.1f}%)")
    print()


def save_report(df):
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "01_data_audit_summary.csv")
    summary = []
    for col in df.columns:
        row = {
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": df[col].isna().sum(),
            "pct_missing": round(100 * df[col].isna().sum() / len(df), 2),
            "n_unique": df[col].nunique(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row["min"] = df[col].min()
            row["max"] = df[col].max()
            row["mean"] = round(df[col].mean(), 4)
        summary.append(row)
    pd.DataFrame(summary).to_csv(report_path, index=False)
    print(f"Saved column summary to {report_path}\n")


def main():
    df = load_data()
    print_column_overview(df)
    print_patient_summary(df)
    print_class_balance(df)
    print_notes_summary(df)
    print_missingness_by_cohort(df)
    flag_anomalies(df)
    save_report(df)

    print("=" * 60)
    print("AUDIT COMPLETE — review output above before proceeding to Step 2")
    print("=" * 60)


if __name__ == "__main__":
    main()
