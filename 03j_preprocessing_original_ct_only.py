"""
Scenario 10 Preprocessing: Original operating surgeon's decision, CT features only.

This establishes the base patient-grouped split for the "original surgeon"
ground-truth family; scenario 11 (demo-only) reuses this same split. Scenario
6 (03f/04f, original surgeon + CT+demographics) is left as originally run —
it uses an independently computed split, but since it applies identical
cleaning, target, random_state, and stratification logic to the same
underlying dataframe, that split is deterministically identical to this one
(verified at runtime below).

Run:
  python 03j_preprocessing_original_ct_only.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Raw_data_03.03.2026.csv"
OUTPUT_DIR = "processed_data_original_ct_only"
REFERENCE_SPLIT_DIR = "processed_data_scenario6_original_ct_plus_demo"

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

    excluded = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded |= df[col].isin(EXCLUDED_LABELS)
    df = df[~excluded].copy()

    for col in SURGEON_COLS:
        df[col + "_bin"] = df[col].map(LABEL_MAP)

    df = df[df["notes"] != "Unusable with same HU range"].copy()

    bad = pd.Series(False, index=df.index)
    for col in [c for c in CT_FEATURE_COLS if "bmd" in c]:
        bad |= df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD)
    bad |= df["cortical_area_mm2"].notna() & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    bad |= df["cortical_thickness_mm"].notna() & (df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN)
    df = df[~bad].copy()

    df["gt_original"] = df["Cement_vs_noCement_original_bin"].astype(int)
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


def verify_matches_scenario6_split(train, test):
    """Confirm this split is identical to scenario 6's, so scenarios 6/10/11
    are a genuinely isolated feature-set comparison despite being built by
    separate scripts."""
    ref_train_path = os.path.join(REFERENCE_SPLIT_DIR, "train.csv")
    ref_test_path = os.path.join(REFERENCE_SPLIT_DIR, "test.csv")
    if not (os.path.exists(ref_train_path) and os.path.exists(ref_test_path)):
        print("  (Scenario 6 outputs not found yet — skipping split-consistency check.)")
        return
    ref_train = pd.read_csv(ref_train_path)
    ref_test = pd.read_csv(ref_test_path)
    this_train_ids = set(zip(train["Anonymize_ID"], train["hip_side"]))
    this_test_ids = set(zip(test["Anonymize_ID"], test["hip_side"]))
    ref_train_ids = set(zip(ref_train["Anonymize_ID"], ref_train["hip_side"]))
    ref_test_ids = set(zip(ref_test["Anonymize_ID"], ref_test["hip_side"]))
    if this_train_ids == ref_train_ids and this_test_ids == ref_test_ids:
        print("  Split matches scenario 6 (results_scenario6_original_ct_plus_demo) exactly.")
    else:
        print("  WARNING: split does NOT match scenario 6's split. "
              "Scenarios 6, 10, and 11 would not be directly comparable.")


def save(train, test, feat_cols, split_strategy):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    id_cols = ["Anonymize_ID", "hip_side", "Cohort_group"]
    target_cols = ["gt_original"]
    save_cols = id_cols + feat_cols + target_cols
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, columns=save_cols)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, columns=save_cols)

    meta = {
        "analysis_variant": "scenario10_original_ct_only",
        "primary_target": "gt_original",
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
    train, test, split_strategy = grouped_split(df, target_col="gt_original")
    verify_matches_scenario6_split(train, test)
    train, test = cohort_knn_impute(train, test)
    train = engineer(train)
    test = engineer(test)
    feat_cols = all_features()
    train, test = scale(train, test, feat_cols)
    save(train, test, feat_cols, split_strategy)
    print(f"Split stratification mode used: {split_strategy}")
    print(f"Saved scenario 10 processed data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
