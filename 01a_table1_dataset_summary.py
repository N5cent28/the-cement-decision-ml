"""
Step 1a: Table 1 and Excluded-Hip Log
====================================
Build publication-ready Table 1 characteristics for:
  - the raw/initial dataset (all 212 hips, before ML cleaning),
  - the full ML analysis cohort (hips retained after the same cleaning
    rules used in 03_preprocessing.py),
  - ML Cohort 1 (Philips) alone,
  - ML Cohort 2 (Toshiba) alone,
plus a detailed log of excluded hips and exclusion reasons.

Run:
  python3.11 08_table1_dataset_summary.py
  # or, after activating the project venv:
  python 08_table1_dataset_summary.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

DATA_FILE = "Raw_data_03.03.2026.csv"
REPORT_DIR = "reports"

SURGEON_COLS = [
    "Cement_vs_noCement_original",
    "Halldor_decision",
    "3d_surg",
]

LABEL_MAP = {
    "Cemented": 1, "cemented": 1,
    "Non-cemented": 0, "non-cemented": 0,
    "Uncemented": 0, "uncemented": 0,
}

EXCLUDED_LABELS = {"Already operated", "already operated"}
UNUSABLE_NOTES = {"Unusable with same HU range"}

CT_FEATURE_COLS = [
    "zone_1_bmd", "zone_2_bmd", "zone_3_bmd", "zone_4_bmd",
    "zone_5_bmd", "zone_6_bmd", "zone_7_bmd",
    "cortical_area_mm2", "cortical_thickness_mm",
    "avg_outer_radius_mm", "ray_inner_radius_mm",
    "geometric_inner_R_mm", "inner_radius_std_mm",
]

BMD_ANOMALY_THRESHOLD = 0.5
CORTICAL_AREA_MIN = 50.0
CORTICAL_THICKNESS_MIN = 2.0


def parse_age_to_years(age_val):
    if pd.isna(age_val):
        return np.nan
    text = str(age_val).strip()
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def mean_sd(series):
    vals = series.dropna()
    if len(vals) == 0:
        return "—"
    return f"{vals.mean():.1f} ± {vals.std(ddof=1):.1f}"


def n_pct(n, denom):
    pct = 100 * n / denom if denom else 0
    return f"{n} ({pct:.1f}%)"


def hip_key(df):
    return df["Anonymize_ID"].astype(str) + "|" + df["hip_side"].astype(str)


def collect_exclusion_reasons(df):
    """Return one row per excluded hip with all applicable reasons."""
    rows = []

    for col in SURGEON_COLS:
        mask = df[col].isin(EXCLUDED_LABELS)
        for _, row in df.loc[mask].iterrows():
            rows.append({
                "Anonymize_ID": row["Anonymize_ID"],
                "hip_side": row["hip_side"],
                "Cohort_group": row["Cohort_group"],
                "notes": row["notes"],
                "exclusion_category": "unavailable_surgeon_label",
                "exclusion_detail": f"{col} = '{row[col]}'",
            })

    notes_mask = df["notes"].isin(UNUSABLE_NOTES)
    for _, row in df.loc[notes_mask].iterrows():
        rows.append({
            "Anonymize_ID": row["Anonymize_ID"],
            "hip_side": row["hip_side"],
            "Cohort_group": row["Cohort_group"],
            "notes": row["notes"],
            "exclusion_category": "unusable_scan_note",
            "exclusion_detail": "notes = 'Unusable with same HU range'",
        })

    bmd_cols = [c for c in CT_FEATURE_COLS if "bmd" in c]
    for col in bmd_cols:
        mask = df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD)
        for _, row in df.loc[mask].iterrows():
            rows.append({
                "Anonymize_ID": row["Anonymize_ID"],
                "hip_side": row["hip_side"],
                "Cohort_group": row["Cohort_group"],
                "notes": row["notes"],
                "exclusion_category": "anomalous_bmd",
                "exclusion_detail": f"{col} = {row[col]:.2f} (< {BMD_ANOMALY_THRESHOLD})",
            })

    mask = df["cortical_area_mm2"].notna() & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    for _, row in df.loc[mask].iterrows():
        rows.append({
            "Anonymize_ID": row["Anonymize_ID"],
            "hip_side": row["hip_side"],
            "Cohort_group": row["Cohort_group"],
            "notes": row["notes"],
            "exclusion_category": "anomalous_cortical_area",
            "exclusion_detail": (
                f"cortical_area_mm2 = {row['cortical_area_mm2']:.2f} (< {CORTICAL_AREA_MIN})"
            ),
        })

    mask = (
        df["cortical_thickness_mm"].notna()
        & (df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN)
    )
    for _, row in df.loc[mask].iterrows():
        rows.append({
            "Anonymize_ID": row["Anonymize_ID"],
            "hip_side": row["hip_side"],
            "Cohort_group": row["Cohort_group"],
            "notes": row["notes"],
            "exclusion_category": "anomalous_cortical_thickness",
            "exclusion_detail": (
                f"cortical_thickness_mm = {row['cortical_thickness_mm']:.2f} "
                f"(< {CORTICAL_THICKNESS_MIN})"
            ),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Anonymize_ID", "hip_side", "Cohort_group", "notes",
            "exclusion_category", "exclusion_detail",
        ])

    detail_df = pd.DataFrame(rows)
    summary = (
        detail_df.groupby(
            ["Anonymize_ID", "hip_side", "Cohort_group", "notes"],
            as_index=False,
        )
        .agg(
            exclusion_categories=("exclusion_category", lambda s: "; ".join(sorted(set(s)))),
            exclusion_details=("exclusion_detail", lambda s: "; ".join(sorted(set(s)))),
            n_reason_flags=("exclusion_detail", "count"),
        )
        .sort_values(["Anonymize_ID", "hip_side"])
    )
    return summary


def apply_ml_cleaning(df):
    """Mirror 03_preprocessing.load_and_clean exclusion order."""
    cleaned = df.copy()
    for col in SURGEON_COLS:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    excluded_mask = pd.Series(False, index=cleaned.index)
    for col in SURGEON_COLS:
        excluded_mask |= cleaned[col].isin(EXCLUDED_LABELS)
    cleaned = cleaned[~excluded_mask].copy()

    notes_mask = cleaned["notes"].isin(UNUSABLE_NOTES)
    cleaned = cleaned[~notes_mask].copy()

    anomaly_mask = pd.Series(False, index=cleaned.index)
    bmd_cols = [c for c in CT_FEATURE_COLS if "bmd" in c]
    for col in bmd_cols:
        anomaly_mask |= cleaned[col].notna() & (cleaned[col] < BMD_ANOMALY_THRESHOLD)
    anomaly_mask |= (
        cleaned["cortical_area_mm2"].notna()
        & (cleaned["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    )
    anomaly_mask |= (
        cleaned["cortical_thickness_mm"].notna()
        & (cleaned["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN)
    )
    cleaned = cleaned[~anomaly_mask].copy()
    return cleaned


def add_ground_truth(df):
    out = df.copy()
    for col in SURGEON_COLS:
        out[col + "_bin"] = out[col].map(LABEL_MAP)

    bin_cols = [c + "_bin" for c in SURGEON_COLS]
    valid_all = out[bin_cols].notna().all(axis=1)
    cemented_votes = out.loc[valid_all, bin_cols].sum(axis=1)
    out.loc[valid_all, "gt_majority"] = (cemented_votes >= 2).astype(int)
    return out


def build_table1(df, column_label, include_cohort_breakdown=True):
    """Publication-style Table 1 for an arbitrary hip-level dataframe."""
    n_hips = len(df)
    n_patients = df["Anonymize_ID"].nunique()

    patients = df.drop_duplicates("Anonymize_ID").copy()
    patients["patient_age_years"] = patients["patient_age"].apply(parse_age_to_years)

    rows = [
        ("Hips, n", str(n_hips)),
        ("Patients, n", str(n_patients)),
        ("Age, years, mean ± SD (patient-level)", mean_sd(patients["patient_age_years"])),
        ("Sex — Male (patient-level)", n_pct((patients["sex"] == "M").sum(), len(patients))),
        ("Sex — Female (patient-level)", n_pct((patients["sex"] == "F").sum(), len(patients))),
        ("Weight, kg, mean ± SD (patient-level)", mean_sd(patients["weight"])),
        ("Height, cm, mean ± SD (patient-level)", mean_sd(patients["height"])),
        ("BMI, kg/m², mean ± SD (patient-level)", mean_sd(patients["BMI"])),
    ]

    if include_cohort_breakdown:
        rows.extend([
            ("Cohort 1 (Philips) — hips", n_pct((df["Cohort_group"] == "Cohort_1").sum(), n_hips)),
            (
                "Cohort 1 (Philips) — patients",
                str(df.loc[df["Cohort_group"] == "Cohort_1", "Anonymize_ID"].nunique()),
            ),
            ("Cohort 2 (Toshiba) — hips", n_pct((df["Cohort_group"] == "Cohort_2").sum(), n_hips)),
            (
                "Cohort 2 (Toshiba) — patients",
                str(df.loc[df["Cohort_group"] == "Cohort_2", "Anonymize_ID"].nunique()),
            ),
        ])

    rows.extend([
        ("Hip side — Left", n_pct((df["hip_side"] == "Left").sum(), n_hips)),
        ("Hip side — Right", n_pct((df["hip_side"] == "Right").sum(), n_hips)),
        ("Operated side — Left", n_pct((df["op_side"] == "Left").sum(), n_hips)),
        ("Operated side — Right", n_pct((df["op_side"] == "Right").sum(), n_hips)),
        ("Bilateral scans (2 hips), patients", str((df.groupby("Anonymize_ID").size() == 2).sum())),
        ("Unilateral scans (1 hip), patients", str((df.groupby("Anonymize_ID").size() == 1).sum())),
    ])

    if "gt_majority" in df.columns:
        rows.extend([
            ("Majority vote — Cemented", n_pct((df["gt_majority"] == 1).sum(), n_hips)),
            ("Majority vote — Non-cemented", n_pct((df["gt_majority"] == 0).sum(), n_hips)),
        ])

    for note, cnt in df["notes"].value_counts().items():
        rows.append((f"Scan note — {note}", n_pct(cnt, n_hips)))

    table = pd.DataFrame(rows, columns=["Characteristic", column_label])
    return table


def save_markdown_table(table, path, title):
    lines = [f"# {title}", "", "| Characteristic | Value |", "|---|---|"]
    for _, row in table.iterrows():
        lines.append(f"| {row.iloc[0]} | {row.iloc[1]} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    raw = pd.read_csv(DATA_FILE)
    print(f"Loaded raw data: {len(raw)} hips from {raw['Anonymize_ID'].nunique()} patients")

    excluded = collect_exclusion_reasons(raw)
    cleaned = apply_ml_cleaning(raw)
    cleaned = add_ground_truth(cleaned)

    excluded_keys = set(hip_key(excluded)) if len(excluded) else set()
    cleaned_keys = set(hip_key(cleaned))
    removed_keys = excluded_keys - cleaned_keys
    if removed_keys:
        excluded = excluded[hip_key(excluded).isin(removed_keys)].copy()

    n_removed = len(excluded)
    print(f"Excluded hips: {n_removed}")
    print(f"ML cohort: {len(cleaned)} hips from {cleaned['Anonymize_ID'].nunique()} patients")

    excluded_path = os.path.join(REPORT_DIR, "excluded_hips_with_reasons.csv")
    excluded.to_csv(excluded_path, index=False)

    table1 = build_table1(cleaned, f"ML cohort (N={len(cleaned)} hips)")
    table1_csv = os.path.join(REPORT_DIR, "table1_ml_dataset.csv")
    table1_md = os.path.join(REPORT_DIR, "table1_ml_dataset.md")
    table1.to_csv(table1_csv, index=False)
    save_markdown_table(table1, table1_md, "Table 1. ML Analysis Cohort Characteristics")

    # Raw starting dataset (all 212 hips, before any ML cleaning/exclusions)
    raw_labeled = add_ground_truth(raw)
    table1_raw = build_table1(raw_labeled, f"Raw dataset (N={len(raw)} hips)")
    table1_raw_csv = os.path.join(REPORT_DIR, "table1_raw_dataset.csv")
    table1_raw_md = os.path.join(REPORT_DIR, "table1_raw_dataset.md")
    table1_raw.to_csv(table1_raw_csv, index=False)
    save_markdown_table(
        table1_raw, table1_raw_md, "Table 1. Initial/Raw Dataset Characteristics (All Hips)"
    )

    # ML cohort split by imaging cohort (Cohort 1 = Philips, Cohort 2 = Toshiba)
    cleaned_c1 = cleaned[cleaned["Cohort_group"] == "Cohort_1"].copy()
    cleaned_c2 = cleaned[cleaned["Cohort_group"] == "Cohort_2"].copy()

    table1_c1 = build_table1(
        cleaned_c1, f"Cohort 1 / Philips, ML cohort (N={len(cleaned_c1)} hips)",
        include_cohort_breakdown=False,
    )
    table1_c1_csv = os.path.join(REPORT_DIR, "table1_ml_cohort1.csv")
    table1_c1_md = os.path.join(REPORT_DIR, "table1_ml_cohort1.md")
    table1_c1.to_csv(table1_c1_csv, index=False)
    save_markdown_table(
        table1_c1, table1_c1_md, "Table 1. ML Cohort 1 (Philips) Characteristics"
    )

    table1_c2 = build_table1(
        cleaned_c2, f"Cohort 2 / Toshiba, ML cohort (N={len(cleaned_c2)} hips)",
        include_cohort_breakdown=False,
    )
    table1_c2_csv = os.path.join(REPORT_DIR, "table1_ml_cohort2.csv")
    table1_c2_md = os.path.join(REPORT_DIR, "table1_ml_cohort2.md")
    table1_c2.to_csv(table1_c2_csv, index=False)
    save_markdown_table(
        table1_c2, table1_c2_md, "Table 1. ML Cohort 2 (Toshiba) Characteristics"
    )

    summary_path = os.path.join(REPORT_DIR, "table1_exclusion_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Dataset exclusion summary\n")
        f.write("=========================\n")
        f.write(f"Raw hips: {len(raw)}\n")
        f.write(f"Excluded hips: {n_removed}\n")
        f.write(f"ML cohort hips: {len(cleaned)}\n")
        f.write(f"ML cohort patients: {cleaned['Anonymize_ID'].nunique()}\n\n")
        f.write("Excluded hips:\n")
        for _, row in excluded.iterrows():
            f.write(
                f"  ID {row['Anonymize_ID']} {row['hip_side']} "
                f"({row['Cohort_group']}): {row['exclusion_details']}\n"
            )

    saved_paths = [
        excluded_path,
        table1_csv, table1_md,
        table1_raw_csv, table1_raw_md,
        table1_c1_csv, table1_c1_md,
        table1_c2_csv, table1_c2_md,
        summary_path,
    ]
    print("\nSaved:\n  " + "\n  ".join(saved_paths))
    print("\nExcluded hips:")
    print(excluded.to_string(index=False))
    print("\nTable 1 preview (ML cohort):")
    print(table1.to_string(index=False))
    print("\nTable 1 preview (raw dataset):")
    print(table1_raw.to_string(index=False))
    print("\nTable 1 preview (ML Cohort 1):")
    print(table1_c1.to_string(index=False))
    print("\nTable 1 preview (ML Cohort 2):")
    print(table1_c2.to_string(index=False))


if __name__ == "__main__":
    main()
