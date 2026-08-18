"""
Step 3: Leakage-Safe Preprocessing Pipeline
============================================
1. Harmonize labels, remove unusable scans, apply anomaly filters.
2. Construct ground truth from surgeon labels (multiple strategies).
3. Patient-grouped train/test split stratified by target + cohort.
4. Cohort-specific KNN imputation fitted on training data only.
5. Engineer derived features (zone averages, BMD-to-geometry ratios).
6. Scale features using training fit only.
7. Save processed train/test sets for modeling.

Run:  python 03_preprocessing.py
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Raw_data_03.03.2026.csv"
OUTPUT_DIR = "processed_data"

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

# CT-derived feature columns (no demographics)
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

TEST_SIZE = 0.20
RANDOM_STATE = 42
KNN_NEIGHBORS = 5


def load_and_clean():
    """Load CSV, harmonize labels, drop unusable rows."""
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows.\n")

    # Harmonize surgeon labels to binary
    for col in SURGEON_COLS:
        df[col] = df[col].astype(str).str.strip()

    # Remove rows where any rater has an excluded label
    excluded_mask = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded_mask |= df[col].isin(EXCLUDED_LABELS)
    n_excl = excluded_mask.sum()
    if n_excl > 0:
        print(f"  Dropped {n_excl} rows with unavailable rater labels.")
    df = df[~excluded_mask].copy()

    # Map to binary
    for col in SURGEON_COLS:
        df[col + "_bin"] = df[col].map(LABEL_MAP)

    # Remove scans flagged as unusable in notes
    unusable_notes = {"Unusable with same HU range"}
    notes_mask = df["notes"].isin(unusable_notes)
    n_unusable = notes_mask.sum()
    if n_unusable > 0:
        print(f"  Dropped {n_unusable} rows flagged as unusable in notes.")
    df = df[~notes_mask].copy()

    # Flag and remove rows with clearly anomalous CT values
    anomaly_mask = pd.Series(False, index=df.index)

    bmd_cols = [c for c in CT_FEATURE_COLS if "bmd" in c]
    for col in bmd_cols:
        anomaly_mask |= (df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD))

    anomaly_mask |= (df["cortical_area_mm2"].notna()
                     & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN))
    anomaly_mask |= (df["cortical_thickness_mm"].notna()
                     & (df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN))

    n_anom = anomaly_mask.sum()
    if n_anom > 0:
        print(f"  Dropped {n_anom} rows with anomalous CT feature values.")
    df = df[~anomaly_mask].copy()

    print(f"  Remaining: {len(df)} rows, "
          f"{df['Anonymize_ID'].nunique()} patients.\n")
    return df


def build_ground_truth(df):
    """Create multiple ground truth columns from surgeon labels."""
    bin_cols = [c + "_bin" for c in SURGEON_COLS]
    valid_all = df[bin_cols].notna().all(axis=1)

    cemented_votes = df.loc[valid_all, bin_cols].sum(axis=1)

    # Strategy 1: original surgeon label only
    df["gt_original"] = df["Cement_vs_noCement_original_bin"]

    # Strategy 2: majority vote (at least 2 of 3 agree)
    df["gt_majority"] = np.nan
    df.loc[valid_all, "gt_majority"] = (cemented_votes >= 2).astype(int)

    # Strategy 3: vote fraction for probabilistic training
    df["gt_vote_fraction"] = np.nan
    df.loc[valid_all, "gt_vote_fraction"] = cemented_votes / 3.0

    # Agreement category for stratification and analysis
    df["agreement_category"] = "incomplete"
    df.loc[valid_all & (cemented_votes == 3), "agreement_category"] = "unanimous_cemented"
    df.loc[valid_all & (cemented_votes == 0), "agreement_category"] = "unanimous_noncemented"
    df.loc[valid_all & (cemented_votes == 2), "agreement_category"] = "majority_cemented"
    df.loc[valid_all & (cemented_votes == 1), "agreement_category"] = "majority_noncemented"

    print("Ground truth strategies created:")
    print(f"  gt_original:      {df['gt_original'].notna().sum()} valid")
    print(f"  gt_majority:      {df['gt_majority'].notna().sum()} valid")
    print(f"  gt_vote_fraction: {df['gt_vote_fraction'].notna().sum()} valid")
    print(f"\n  gt_majority class balance:")
    if df["gt_majority"].notna().any():
        vc = df["gt_majority"].value_counts()
        for label, cnt in vc.items():
            desc = "cemented" if label == 1 else "non-cemented"
            print(f"    {desc}: {cnt}")
    print()

    return df


def patient_grouped_split(df, target_col="gt_majority"):
    """
    Split into train/test by patient group, stratified by target + cohort.
    All hips from the same patient stay in the same split.
    """
    usable = df[target_col].notna()
    df_usable = df[usable].copy()

    # Build a patient-level stratification key
    patient_info = (df_usable.groupby("Anonymize_ID")
                    .agg({target_col: "first", "Cohort_group": "first"})
                    .reset_index())
    patient_info["strat_key"] = (patient_info[target_col].astype(int).astype(str)
                                 + "_" + patient_info["Cohort_group"])

    # True patient-level stratification. Fallback progressively if strata are sparse.
    split_strategy = "target_plus_cohort"
    try:
        train_pat, test_pat = train_test_split(
            patient_info,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=patient_info["strat_key"],
        )
    except ValueError:
        split_strategy = "target_only_fallback"
        try:
            train_pat, test_pat = train_test_split(
                patient_info,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=patient_info[target_col].astype(int),
            )
        except ValueError:
            split_strategy = "unstratified_fallback"
            train_pat, test_pat = train_test_split(
                patient_info,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

    train_patient_ids = set(train_pat["Anonymize_ID"])
    test_patient_ids = set(test_pat["Anonymize_ID"])

    train_mask = df_usable["Anonymize_ID"].isin(train_patient_ids)
    test_mask = df_usable["Anonymize_ID"].isin(test_patient_ids)

    df_train = df_usable[train_mask].copy()
    df_test = df_usable[test_mask].copy()

    print(f"Train/test split (by patient, {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}):")
    print(f"  Train: {len(df_train)} hips from "
          f"{df_train['Anonymize_ID'].nunique()} patients")
    print(f"  Test:  {len(df_test)} hips from "
          f"{df_test['Anonymize_ID'].nunique()} patients")
    print(f"\n  Train cohort distribution:")
    print(f"    {df_train['Cohort_group'].value_counts().to_dict()}")
    print(f"  Test cohort distribution:")
    print(f"    {df_test['Cohort_group'].value_counts().to_dict()}")
    print(f"\n  Train target distribution (gt_majority):")
    print(f"    {df_train['gt_majority'].value_counts().to_dict()}")
    print(f"  Test target distribution (gt_majority):")
    print(f"    {df_test['gt_majority'].value_counts().to_dict()}")
    print(f"\n  Split stratification mode used: {split_strategy}")
    print()

    return df_train, df_test, split_strategy


def cohort_specific_knn_impute(df_train, df_test, feature_cols):
    """
    Impute missing values using KNN, fitted per-cohort on training data only.
    Training imputer is then applied to the corresponding cohort in test.
    """
    imputers = {}

    for cohort in sorted(df_train["Cohort_group"].unique()):
        mask_train = df_train["Cohort_group"] == cohort
        mask_test = df_test["Cohort_group"] == cohort

        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance")
        df_train.loc[mask_train, feature_cols] = imputer.fit_transform(
            df_train.loc[mask_train, feature_cols]
        )

        if mask_test.any():
            df_test.loc[mask_test, feature_cols] = imputer.transform(
                df_test.loc[mask_test, feature_cols]
            )

        imputers[cohort] = imputer
        n_train = mask_train.sum()
        n_test = mask_test.sum()
        print(f"  KNN imputed {cohort}: {n_train} train, {n_test} test rows")

    print()
    return df_train, df_test, imputers


def engineer_features(df):
    """Add derived features recommended in the handoff guide."""
    df = df.copy()

    df["zones_1_7_avg"] = df[["zone_1_bmd", "zone_7_bmd"]].mean(axis=1)
    df["zones_2_6_avg"] = df[["zone_2_bmd", "zone_6_bmd"]].mean(axis=1)

    # Ratios to cortical features
    for zone_avg in ["zones_1_7_avg", "zones_2_6_avg"]:
        for geom in ["cortical_thickness_mm", "cortical_area_mm2"]:
            ratio_name = f"{zone_avg}_to_{geom}"
            df[ratio_name] = df[zone_avg] / df[geom].replace(0, np.nan)

    # BMD summary features across all 7 zones
    bmd_cols = [f"zone_{i}_bmd" for i in range(1, 8)]
    df["bmd_mean"] = df[bmd_cols].mean(axis=1)
    df["bmd_std"] = df[bmd_cols].std(axis=1)
    df["bmd_range"] = df[bmd_cols].max(axis=1) - df[bmd_cols].min(axis=1)

    return df


def get_all_feature_cols(df):
    """Return the full list of feature columns after engineering."""
    engineered = [
        "zones_1_7_avg", "zones_2_6_avg",
        "zones_1_7_avg_to_cortical_thickness_mm",
        "zones_1_7_avg_to_cortical_area_mm2",
        "zones_2_6_avg_to_cortical_thickness_mm",
        "zones_2_6_avg_to_cortical_area_mm2",
        "bmd_mean", "bmd_std", "bmd_range",
    ]
    return CT_FEATURE_COLS + engineered


def scale_features(df_train, df_test, feature_cols):
    """Fit scaler on training data, transform both sets."""
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    print(f"  Scaled {len(feature_cols)} features using StandardScaler "
          f"(fit on train only).\n")
    return df_train, df_test, scaler


def save_outputs(df_train, df_test, feature_cols, scaler, split_strategy):
    """Save processed datasets and metadata."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Columns to save: identifiers + features + all ground truth variants
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group", "agreement_category"]
    gt_cols = ["gt_original", "gt_majority", "gt_vote_fraction"]
    save_cols = id_cols + feature_cols + gt_cols

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    df_train[save_cols].to_csv(train_path, index=False)
    df_test[save_cols].to_csv(test_path, index=False)
    print(f"Saved train set to {train_path}  ({len(df_train)} rows)")
    print(f"Saved test set to {test_path}  ({len(df_test)} rows)")

    # Save feature column list and metadata
    meta = {
        "feature_columns": feature_cols,
        "ct_base_features": CT_FEATURE_COLS,
        "ground_truth_columns": gt_cols,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "knn_neighbors": KNN_NEIGHBORS,
        "n_train_rows": len(df_train),
        "n_test_rows": len(df_test),
        "n_train_patients": int(df_train["Anonymize_ID"].nunique()),
        "n_test_patients": int(df_test["Anonymize_ID"].nunique()),
        "split_stratification_mode": split_strategy,
    }
    meta_path = os.path.join(OUTPUT_DIR, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}\n")


def main():
    print("=" * 60)
    print("STEP 3: PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    df = load_and_clean()
    df = build_ground_truth(df)

    df_train, df_test, split_strategy = patient_grouped_split(df, target_col="gt_majority")

    print("Cohort-specific KNN imputation (training-only fit):")
    df_train, df_test, _ = cohort_specific_knn_impute(
        df_train, df_test, CT_FEATURE_COLS
    )

    print("Feature engineering:")
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    all_features = get_all_feature_cols(df_train)
    print(f"  Created {len(all_features)} total features.\n")

    print("Scaling:")
    df_train, df_test, scaler = scale_features(df_train, df_test, all_features)

    save_outputs(df_train, df_test, all_features, scaler, split_strategy)

    print("=" * 60)
    print("PREPROCESSING COMPLETE — proceed to Step 4")
    print("=" * 60)


if __name__ == "__main__":
    main()
