"""
Scenario 5 Preprocessing: Halldor+3D agreement only, CT features only.

Run:
  python 03e_preprocessing_agree_ct_only.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Raw_data_03.03.2026.csv"
OUTPUT_DIR = "processed_data_scenario5_agree_ct_only"

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
    for col in SURGEON_COLS:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows with unavailable rater labels.
    excluded = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded |= df[col].isin(EXCLUDED_LABELS)
    df = df[~excluded].copy()

    # Binary maps.
    for col in SURGEON_COLS:
        df[col + "_bin"] = df[col].map(LABEL_MAP)

    # Exclude unusable scans.
    df = df[df["notes"] != "Unusable with same HU range"].copy()

    # Exclude obvious CT artifacts.
    bad = pd.Series(False, index=df.index)
    for col in [c for c in CT_FEATURE_COLS if "bmd" in c]:
        bad |= df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD)
    bad |= df["cortical_area_mm2"].notna() & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    bad |= df["cortical_thickness_mm"].notna() & (df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN)
    df = df[~bad].copy()
    return df


def build_targets(df):
    # Agreement-only subset for Halldor + 3D.
    agree = (
        df["Halldor_decision_bin"].notna()
        & df["3d_surg_bin"].notna()
        & (df["Halldor_decision_bin"] == df["3d_surg_bin"])
    )
    df = df[agree].copy()
    df["gt_h3d_agree"] = df["Halldor_decision_bin"].astype(int)
    return df


def grouped_split(df, target_col):
    patient_info = (
        df.groupby("Anonymize_ID")
        .agg({target_col: "first", "Cohort_group": "first"})
        .reset_index()
    )
    patient_info["strat_key"] = patient_info[target_col].astype(int).astype(str) + "_" + patient_info["Cohort_group"]

    split_strategy = "target_plus_cohort"
    try:
        tr, te = train_test_split(
            patient_info,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=patient_info["strat_key"],
        )
    except ValueError:
        split_strategy = "target_only_fallback"
        try:
            tr, te = train_test_split(
                patient_info,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=patient_info[target_col].astype(int),
            )
        except ValueError:
            split_strategy = "unstratified_fallback"
            tr, te = train_test_split(
                patient_info,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )
    train_ids = set(tr["Anonymize_ID"])
    test_ids = set(te["Anonymize_ID"])

    train = df[df["Anonymize_ID"].isin(train_ids)].copy()
    test = df[df["Anonymize_ID"].isin(test_ids)].copy()
    return train, test, split_strategy


def cohort_knn_impute(train, test):
    for cohort in sorted(train["Cohort_group"].unique()):
        tr_mask = train["Cohort_group"] == cohort
        te_mask = test["Cohort_group"] == cohort
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance")
        train.loc[tr_mask, CT_FEATURE_COLS] = imputer.fit_transform(train.loc[tr_mask, CT_FEATURE_COLS])
        if te_mask.any():
            test.loc[te_mask, CT_FEATURE_COLS] = imputer.transform(test.loc[te_mask, CT_FEATURE_COLS])
    return train, test


def engineer(df):
    df = df.copy()
    df["zones_1_7_avg"] = df[["zone_1_bmd", "zone_7_bmd"]].mean(axis=1)
    df["zones_2_6_avg"] = df[["zone_2_bmd", "zone_6_bmd"]].mean(axis=1)
    for avg in ["zones_1_7_avg", "zones_2_6_avg"]:
        for geom in ["cortical_thickness_mm", "cortical_area_mm2"]:
            df[f"{avg}_to_{geom}"] = df[avg] / df[geom].replace(0, np.nan)
    bmd_cols = [f"zone_{i}_bmd" for i in range(1, 8)]
    df["bmd_mean"] = df[bmd_cols].mean(axis=1)
    df["bmd_std"] = df[bmd_cols].std(axis=1)
    df["bmd_range"] = df[bmd_cols].max(axis=1) - df[bmd_cols].min(axis=1)
    return df


def all_features():
    return CT_FEATURE_COLS + [
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


def scale(train, test, feat_cols):
    scaler = StandardScaler()
    train[feat_cols] = scaler.fit_transform(train[feat_cols])
    test[feat_cols] = scaler.transform(test[feat_cols])
    return train, test


def save(train, test, feat_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group"]
    target_cols = ["gt_h3d_agree"]
    save_cols = id_cols + feat_cols + target_cols
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, columns=save_cols)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, columns=save_cols)

    meta = {
        "analysis_variant": "scenario5_h3d_agreement_ct_only",
        "primary_target": "gt_h3d_agree",
        "feature_columns": feat_cols,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_patients": int(train["Anonymize_ID"].nunique()),
        "n_test_patients": int(test["Anonymize_ID"].nunique()),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "split_stratification_mode": split_strategy,
    }
    with open(os.path.join(OUTPUT_DIR, "preprocessing_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    df = load_and_clean()
    df = build_targets(df)
    train, test, split_strategy = grouped_split(df, target_col="gt_h3d_agree")
    train, test = cohort_knn_impute(train, test)
    train = engineer(train)
    test = engineer(test)
    feat_cols = all_features()
    train, test = scale(train, test, feat_cols)
    save(train, test, feat_cols, split_strategy)
    print(f"Split stratification mode used: {split_strategy}")
    print(f"Saved scenario 5 processed data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
