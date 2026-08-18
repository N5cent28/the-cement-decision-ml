"""
Step 3B: Leakage-Safe Preprocessing Pipeline (Vote Fraction Ground Truth)
=========================================================================
This pipeline mirrors Step 3 but uses gt_vote_fraction as the primary target
for splitting/stratification metadata and writes outputs to a separate folder.

Run:  python 03b_preprocessing_vote_fraction.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Raw_data_03.03.2026.csv"
OUTPUT_DIR = "processed_data_vote_fraction"

SURGEON_COLS = [
    "Cement_vs_noCement_original",
    "Halldor_decision",
    "3d_surg",
]

LABEL_MAP = {
    "Cemented": 1,
    "cemented": 1,
    "Non-cemented": 0,
    "non-cemented": 0,
    "Uncemented": 0,
    "uncemented": 0,
}

EXCLUDED_LABELS = {"Already operated", "already operated"}

CT_FEATURE_COLS = [
    "zone_1_bmd",
    "zone_2_bmd",
    "zone_3_bmd",
    "zone_4_bmd",
    "zone_5_bmd",
    "zone_6_bmd",
    "zone_7_bmd",
    "cortical_area_mm2",
    "cortical_thickness_mm",
    "avg_outer_radius_mm",
    "ray_inner_radius_mm",
    "geometric_inner_R_mm",
    "inner_radius_std_mm",
]

BMD_ANOMALY_THRESHOLD = 0.5
CORTICAL_AREA_MIN = 50.0
CORTICAL_THICKNESS_MIN = 2.0

TEST_SIZE = 0.20
RANDOM_STATE = 42
KNN_NEIGHBORS = 5


def load_and_clean():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows.\n")

    for col in SURGEON_COLS:
        df[col] = df[col].astype(str).str.strip()

    excluded_mask = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded_mask |= df[col].isin(EXCLUDED_LABELS)
    if excluded_mask.sum() > 0:
        print(f"  Dropped {excluded_mask.sum()} rows with unavailable rater labels.")
    df = df[~excluded_mask].copy()

    for col in SURGEON_COLS:
        df[col + "_bin"] = df[col].map(LABEL_MAP)

    unusable_notes = {"Unusable with same HU range"}
    notes_mask = df["notes"].isin(unusable_notes)
    if notes_mask.sum() > 0:
        print(f"  Dropped {notes_mask.sum()} rows flagged as unusable in notes.")
    df = df[~notes_mask].copy()

    anomaly_mask = pd.Series(False, index=df.index)
    bmd_cols = [c for c in CT_FEATURE_COLS if "bmd" in c]
    for col in bmd_cols:
        anomaly_mask |= (df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD))
    anomaly_mask |= df["cortical_area_mm2"].notna() & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    anomaly_mask |= df["cortical_thickness_mm"].notna() & (
        df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN
    )
    if anomaly_mask.sum() > 0:
        print(f"  Dropped {anomaly_mask.sum()} rows with anomalous CT feature values.")
    df = df[~anomaly_mask].copy()

    print(f"  Remaining: {len(df)} rows, {df['Anonymize_ID'].nunique()} patients.\n")
    return df


def build_ground_truth(df):
    bin_cols = [c + "_bin" for c in SURGEON_COLS]
    valid_all = df[bin_cols].notna().all(axis=1)
    cemented_votes = df.loc[valid_all, bin_cols].sum(axis=1)

    df["gt_original"] = df["Cement_vs_noCement_original_bin"]
    df["gt_majority"] = np.nan
    df.loc[valid_all, "gt_majority"] = (cemented_votes >= 2).astype(int)
    df["gt_vote_fraction"] = np.nan
    df.loc[valid_all, "gt_vote_fraction"] = cemented_votes / 3.0

    df["agreement_category"] = "incomplete"
    df.loc[valid_all & (cemented_votes == 3), "agreement_category"] = "unanimous_cemented"
    df.loc[valid_all & (cemented_votes == 0), "agreement_category"] = "unanimous_noncemented"
    df.loc[valid_all & (cemented_votes == 2), "agreement_category"] = "majority_cemented"
    df.loc[valid_all & (cemented_votes == 1), "agreement_category"] = "majority_noncemented"

    print("Ground truth strategies created:")
    print(f"  gt_vote_fraction (primary for this run): {df['gt_vote_fraction'].notna().sum()} valid")
    print(f"  vote-fraction distribution: {df['gt_vote_fraction'].value_counts().sort_index().to_dict()}\n")
    return df


def patient_grouped_split(df, target_col="gt_vote_fraction"):
    usable = df[target_col].notna()
    df_usable = df[usable].copy()

    patient_info = (
        df_usable.groupby("Anonymize_ID")
        .agg({target_col: "first", "Cohort_group": "first"})
        .reset_index()
    )
    # Convert target to compact stratification bins (0.00, 0.33, 0.67, 1.00)
    patient_info["target_bin"] = patient_info[target_col].round(2).astype(str)
    patient_info["strat_key"] = patient_info["target_bin"] + "_" + patient_info["Cohort_group"]

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
                stratify=patient_info["target_bin"],
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

    df_train = df_usable[df_usable["Anonymize_ID"].isin(train_patient_ids)].copy()
    df_test = df_usable[df_usable["Anonymize_ID"].isin(test_patient_ids)].copy()

    print(f"Train/test split (by patient, {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}):")
    print(f"  Train: {len(df_train)} hips from {df_train['Anonymize_ID'].nunique()} patients")
    print(f"  Test:  {len(df_test)} hips from {df_test['Anonymize_ID'].nunique()} patients")
    print(f"  Train vote-fraction dist: {df_train[target_col].value_counts().sort_index().to_dict()}")
    print(f"  Test vote-fraction dist:  {df_test[target_col].value_counts().sort_index().to_dict()}\n")
    print(f"  Split stratification mode used: {split_strategy}\n")
    return df_train, df_test, split_strategy


def cohort_specific_knn_impute(df_train, df_test, feature_cols):
    for cohort in sorted(df_train["Cohort_group"].unique()):
        train_mask = df_train["Cohort_group"] == cohort
        test_mask = df_test["Cohort_group"] == cohort

        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance")
        df_train.loc[train_mask, feature_cols] = imputer.fit_transform(df_train.loc[train_mask, feature_cols])
        if test_mask.any():
            df_test.loc[test_mask, feature_cols] = imputer.transform(df_test.loc[test_mask, feature_cols])

        print(f"  KNN imputed {cohort}: {train_mask.sum()} train, {test_mask.sum()} test rows")
    print()
    return df_train, df_test


def engineer_features(df):
    df = df.copy()
    df["zones_1_7_avg"] = df[["zone_1_bmd", "zone_7_bmd"]].mean(axis=1)
    df["zones_2_6_avg"] = df[["zone_2_bmd", "zone_6_bmd"]].mean(axis=1)

    for zone_avg in ["zones_1_7_avg", "zones_2_6_avg"]:
        for geom in ["cortical_thickness_mm", "cortical_area_mm2"]:
            ratio_name = f"{zone_avg}_to_{geom}"
            df[ratio_name] = df[zone_avg] / df[geom].replace(0, np.nan)

    bmd_cols = [f"zone_{i}_bmd" for i in range(1, 8)]
    df["bmd_mean"] = df[bmd_cols].mean(axis=1)
    df["bmd_std"] = df[bmd_cols].std(axis=1)
    df["bmd_range"] = df[bmd_cols].max(axis=1) - df[bmd_cols].min(axis=1)
    return df


def get_all_feature_cols():
    engineered = [
        "zones_1_7_avg",
        "zones_2_6_avg",
        "zones_1_7_avg_to_cortical_thickness_mm",
        "zones_1_7_avg_to_cortical_area_mm2",
        "zones_2_6_avg_to_cortical_thickness_mm",
        "zones_2_6_avg_to_cortical_area_mm2",
        "bmd_mean",
        "bmd_std",
        "bmd_range",
    ]
    return CT_FEATURE_COLS + engineered


def scale_features(df_train, df_test, feature_cols):
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    print(f"  Scaled {len(feature_cols)} features using StandardScaler (fit on train only).\n")
    return df_train, df_test


def save_outputs(df_train, df_test, feature_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group", "agreement_category"]
    gt_cols = ["gt_original", "gt_majority", "gt_vote_fraction"]
    save_cols = id_cols + feature_cols + gt_cols

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    df_train[save_cols].to_csv(train_path, index=False)
    df_test[save_cols].to_csv(test_path, index=False)
    print(f"Saved train set to {train_path}  ({len(df_train)} rows)")
    print(f"Saved test set to {test_path}  ({len(df_test)} rows)")

    meta = {
        "analysis_variant": "vote_fraction",
        "primary_target": "gt_vote_fraction",
        "feature_columns": feature_cols,
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
    print("=" * 70)
    print("STEP 3B: PREPROCESSING PIPELINE (VOTE FRACTION)")
    print("=" * 70 + "\n")

    df = load_and_clean()
    df = build_ground_truth(df)

    df_train, df_test, split_strategy = patient_grouped_split(df, target_col="gt_vote_fraction")
    print("Cohort-specific KNN imputation (training-only fit):")
    df_train, df_test = cohort_specific_knn_impute(df_train, df_test, CT_FEATURE_COLS)

    print("Feature engineering:")
    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    all_features = get_all_feature_cols()
    print(f"  Created {len(all_features)} total features.\n")

    print("Scaling:")
    df_train, df_test = scale_features(df_train, df_test, all_features)

    save_outputs(df_train, df_test, all_features, split_strategy)

    print("=" * 70)
    print("PREPROCESSING (VOTE FRACTION) COMPLETE — proceed to Step 4B")
    print("=" * 70)


if __name__ == "__main__":
    main()
