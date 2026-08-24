"""
Shared preprocessing library for the THA cement-decision sensitivity matrix.
=============================================================================

WHY THIS FILE EXISTS
---------------------
Every scenario in this project (see SCENARIO_INDEX.md for the full 13-cell
ground-truth x feature-set matrix) starts from the same raw CSV and applies
the same cleaning rules, ground-truth definitions, patient-grouped split
logic, imputation, feature engineering, and scaling. Before this file
existed, each of the 03*_preprocessing*.py scripts (and 07's repeated-split
evaluator) reimplemented this logic independently — 13+ near-identical
copies of the same ~150 lines. That duplication is how a real bug shipped
in an earlier revision: `03e_preprocessing_agree_ct_only.py`'s ground truth
was named `gt_h3d_agree` and actually implements 2-of-3 rater agreement
(Halldor + 3D surgeon, excluding the original surgeon), which is a
different and weaker claim than "all surgeons agree" — a mislabeling that
was only caught by manual review, not by any test, because there was no
single place to check the definition against.

Every scenario script now imports the functions below instead of
reimplementing them. If you need to change a cleaning rule, an exclusion
threshold, or a target definition, change it here ONCE — every scenario
picks up the change automatically the next time its script is run. If you
add a 14th scenario, you should almost never need to touch this file; you
should only need to call these functions with a different `target_col`,
`use_ct`/`use_demo` combination, or output path.

WHAT THIS FILE DOES NOT DO
----------------------------
It does not decide which ground truth or feature set a given scenario uses,
does not know about output directories or file names, and does not run any
models. Each `03*` script still owns: which target column it filters on,
which feature set it assembles, where it writes its output, and what goes
in its metadata JSON. This keeps the shared module a pure, reusable data
pipeline rather than a second layer of scenario-specific branching that
would just move the duplication problem instead of removing it.

GROUND TRUTH DEFINITIONS (all built by `load_and_clean_raw`)
---------------------------------------------------------------
- `gt_majority`      : >=2 of 3 raters (original surgeon, Halldor, 3D surgeon)
                       vote cemented. The study's primary/default label.
- `gt_original`      : the original operating surgeon's real-time decision
                       alone (`Cement_vs_noCement_original`).
- `gt_vote_fraction` : continuous fraction of the 3 raters voting cemented,
                       in {0, 1/3, 2/3, 1}. A regression target.
- `gt_h3d_agree`     : Halldor and the 3D surgeon agree with each other.
                       This is 2-of-3 agreement and DOES NOT require the
                       original surgeon's vote to match. Named `h3d` (not
                       `unanimous`) deliberately to keep this distinction
                       visible in every scenario that uses it (scenario 5).
- `gt_unanimous`     : all 3 raters agree with each other (true 3-way
                       unanimous consensus). This is the correct target for
                       any "all surgeons agree" ground truth (scenarios 7-9).
- `agreement_category`: descriptive-only label (unanimous_cemented /
                       unanimous_noncemented / majority_cemented /
                       majority_noncemented / incomplete) used for reporting,
                       never as a model target.

Rows where a given target's inputs are incomplete get NaN for that column
only — cleaning and exclusion happen once, up front, for all rows, and each
scenario script then does `df[df[target_col].notna()]` to get its own
usable subset. This is why the same cleaned base dataframe can serve every
scenario: no scenario-specific filtering happens before this shared step.

LEAKAGE SAFETY
---------------
Every function that fits something (KNN imputer, StandardScaler, median/mode
fill for demographics) is called separately on train and test: fit on train,
transform (never re-fit) on test. `cohort_knn_impute` additionally fits a
separate imputer per Cohort_group (Philips vs. Toshiba), because the two
scanners have different value distributions and imputing across cohorts
would let information leak between them. This matches the leakage-safety
description in methods.md Section 5.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants shared by every scenario. Change a value here, not in a script.
# ---------------------------------------------------------------------------

DATA_FILE = "Raw_data_03.03.2026.csv"

SURGEON_COLS = ["Cement_vs_noCement_original", "Halldor_decision", "3d_surg"]

LABEL_MAP = {
    "Cemented": 1,
    "cemented": 1,
    "Non-cemented": 0,
    "non-cemented": 0,
    "Uncemented": 0,
    "uncemented": 0,
}

# Rows where any rater's label is one of these are dropped entirely, because
# no ground truth (majority, original, vote-fraction, agreement) can be
# computed without a real cemented/non-cemented opinion from all 3 raters.
EXCLUDED_LABELS = {"Already operated", "already operated"}

# Scans explicitly flagged as unusable are dropped; "Questionable Quality"
# scans are deliberately KEPT (see methods.md Section 4.1) — excluding them
# is a distinct, not-yet-implemented sensitivity analysis (see
# SCENARIO_INDEX.md), not something this cleaning step should do silently.
UNUSABLE_NOTES = {"Unusable with same HU range"}

# Base CT-derived feature columns, before feature engineering.
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

DEMO_FEATURES = ["patient_age_years", "sex_binary", "weight", "height", "BMI"]

# Physiologically-implausible-value thresholds used to drop failed-
# segmentation rows (see methods.md Section 4.2 for examples of the actual
# artifact values these catch, e.g. cortical_area_mm2 = 0.47).
BMD_ANOMALY_THRESHOLD = 0.5
CORTICAL_AREA_MIN = 50.0
CORTICAL_THICKNESS_MIN = 2.0

TEST_SIZE = 0.20
RANDOM_STATE = 42
KNN_NEIGHBORS = 5


# ---------------------------------------------------------------------------
# Step 1: load, clean, and build every ground-truth column.
# ---------------------------------------------------------------------------


def parse_age_to_years(age_val):
    """Parse age strings like '064Y', '60Y', '57Y' into numeric years."""
    if pd.isna(age_val):
        return np.nan
    text = str(age_val).strip()
    m = re.search(r"(\d+)", text)
    return float(m.group(1)) if m else np.nan


def load_and_clean_raw(data_file: str = DATA_FILE) -> pd.DataFrame:
    """
    Load the raw CSV, apply every exclusion rule, and compute every ground
    truth column and demographic transform used anywhere in this project.

    This is the ONE function that defines "what counts as a usable hip" and
    "what does each ground truth mean" for the entire sensitivity matrix.
    Every 03*_preprocessing*.py script calls this and then subsets to its
    own `target_col` — it does not re-derive cleaning or labels itself.

    Exclusion order (matches methods.md Section 4, applied once, in order):
      1. Drop rows where any rater has an unusable label (EXCLUDED_LABELS).
      2. Map the 3 surgeon columns to binary cemented/non-cemented.
      3. Drop rows flagged unusable in `notes` (UNUSABLE_NOTES).
      4. Drop rows with physiologically-implausible CT values (anomaly
         thresholds above) — failed segmentation, not real bone signal.

    Returns a dataframe with columns added:
      Cement_vs_noCement_original_bin, Halldor_decision_bin, 3d_surg_bin
      gt_original, gt_majority, gt_vote_fraction, gt_h3d_agree, gt_unanimous
      agreement_category
      patient_age_years, sex_binary
      split_key  (== "<Anonymize_ID>|<hip_side>", used by "same split" scripts
                  to merge onto another scenario's train/test membership)
    """
    df = pd.read_csv(data_file)

    for col in SURGEON_COLS:
        df[col] = df[col].astype(str).str.strip()

    excluded_mask = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded_mask |= df[col].isin(EXCLUDED_LABELS)
    df = df[~excluded_mask].copy()

    for col in SURGEON_COLS:
        df[col + "_bin"] = df[col].map(LABEL_MAP)

    df = df[~df["notes"].isin(UNUSABLE_NOTES)].copy()

    anomaly_mask = pd.Series(False, index=df.index)
    bmd_cols = [c for c in CT_FEATURE_COLS if "bmd" in c]
    for col in bmd_cols:
        anomaly_mask |= df[col].notna() & (df[col] < BMD_ANOMALY_THRESHOLD)
    anomaly_mask |= df["cortical_area_mm2"].notna() & (df["cortical_area_mm2"] < CORTICAL_AREA_MIN)
    anomaly_mask |= df["cortical_thickness_mm"].notna() & (
        df["cortical_thickness_mm"] < CORTICAL_THICKNESS_MIN
    )
    df = df[~anomaly_mask].copy()

    df = _build_ground_truth_columns(df)

    df["patient_age_years"] = df["patient_age"].apply(parse_age_to_years)
    df["sex_binary"] = df["sex"].map({"F": 0.0, "M": 1.0})

    df["split_key"] = df["Anonymize_ID"].astype(str) + "|" + df["hip_side"].astype(str)

    return df


def _build_ground_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Internal: compute every gt_* column. See module docstring for definitions."""
    df = df.copy()
    bin_cols = [c + "_bin" for c in SURGEON_COLS]
    valid_all = df[bin_cols].notna().all(axis=1)
    cemented_votes = df.loc[valid_all, bin_cols].sum(axis=1)

    df["gt_original"] = df["Cement_vs_noCement_original_bin"].astype(int)

    df["gt_majority"] = np.nan
    df.loc[valid_all, "gt_majority"] = (cemented_votes >= 2).astype(int)

    df["gt_vote_fraction"] = np.nan
    df.loc[valid_all, "gt_vote_fraction"] = cemented_votes / 3.0

    # gt_unanimous: all 3 raters must agree. Distinct from gt_h3d_agree below.
    df["gt_unanimous"] = np.nan
    unanimous_mask = valid_all & cemented_votes.reindex(df.index).isin([0, 3])
    df.loc[unanimous_mask, "gt_unanimous"] = (cemented_votes.reindex(df.index) == 3).astype(int)

    # gt_h3d_agree: only Halldor and the 3D surgeon need to agree with each
    # other. The original surgeon's vote is NOT required to match — this is
    # 2-of-3 agreement, not unanimity. Kept as its own target (scenario 5)
    # rather than renamed, to preserve continuity with prior reporting.
    h3d_agree_mask = (
        df["Halldor_decision_bin"].notna()
        & df["3d_surg_bin"].notna()
        & (df["Halldor_decision_bin"] == df["3d_surg_bin"])
    )
    df["gt_h3d_agree"] = np.nan
    df.loc[h3d_agree_mask, "gt_h3d_agree"] = df.loc[h3d_agree_mask, "Halldor_decision_bin"].astype(int)

    # Descriptive-only agreement category (reporting / stratification aid;
    # never used as a model target).
    df["agreement_category"] = "incomplete"
    df.loc[valid_all & (cemented_votes == 3), "agreement_category"] = "unanimous_cemented"
    df.loc[valid_all & (cemented_votes == 0), "agreement_category"] = "unanimous_noncemented"
    df.loc[valid_all & (cemented_votes == 2), "agreement_category"] = "majority_cemented"
    df.loc[valid_all & (cemented_votes == 1), "agreement_category"] = "majority_noncemented"

    return df


# ---------------------------------------------------------------------------
# Step 2: patient-grouped, target-stratified train/test split.
# ---------------------------------------------------------------------------


def grouped_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    is_regression: bool = False,
):
    """
    Split `df` into train/test by patient (`Anonymize_ID`), so all hips from
    one patient land in the same split — this is what prevents leakage for
    bilateral patients, and is the split logic described in methods.md
    Section 5.1.

    Stratifies on `target_col` + `Cohort_group` where feasible, falling back
    progressively (target_plus_cohort -> target_only_fallback ->
    unstratified_fallback) if a stratum is too sparse for sklearn to split.
    The mode actually used is returned so callers can record it in their
    metadata JSON, matching every scenario's existing `preprocessing_metadata.json`.

    `is_regression=True` rounds the continuous target to 2 decimals to form
    stratification bins (used for `gt_vote_fraction`, which only takes
    values in {0, 1/3, 2/3, 1} but is stored as a float) instead of casting
    to int, which would collapse every value to 0.
    """
    patient_info = (
        df.groupby("Anonymize_ID").agg({target_col: "first", "Cohort_group": "first"}).reset_index()
    )
    if is_regression:
        patient_info["strat_key_col"] = patient_info[target_col].round(2).astype(str)
    else:
        patient_info["strat_key_col"] = patient_info[target_col].astype(int).astype(str)
    patient_info["strat_key"] = patient_info["strat_key_col"] + "_" + patient_info["Cohort_group"]

    split_strategy = "target_plus_cohort"
    try:
        train_pat, test_pat = train_test_split(
            patient_info, test_size=test_size, random_state=random_state, stratify=patient_info["strat_key"]
        )
    except ValueError:
        split_strategy = "target_only_fallback"
        try:
            train_pat, test_pat = train_test_split(
                patient_info,
                test_size=test_size,
                random_state=random_state,
                stratify=patient_info["strat_key_col"],
            )
        except ValueError:
            split_strategy = "unstratified_fallback"
            train_pat, test_pat = train_test_split(patient_info, test_size=test_size, random_state=random_state)

    train_ids = set(train_pat["Anonymize_ID"])
    test_ids = set(test_pat["Anonymize_ID"])
    train = df[df["Anonymize_ID"].isin(train_ids)].copy()
    test = df[df["Anonymize_ID"].isin(test_ids)].copy()
    return train, test, split_strategy


# ---------------------------------------------------------------------------
# Step 3: CT feature engineering, imputation, and scaling (all leakage-safe:
# every `fit` happens on train only; test is only ever `transform`-ed).
# ---------------------------------------------------------------------------


def engineer_ct_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 9 derived CT features described in methods.md Section 5.3."""
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


def ct_feature_list() -> list[str]:
    """Full CT feature list (13 base + 9 engineered = 22), post-engineering."""
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


def cohort_knn_impute(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str] = CT_FEATURE_COLS,
    k_neighbors: int = KNN_NEIGHBORS,
):
    """
    KNN-impute `feature_cols`, fitting a SEPARATE imputer per Cohort_group on
    training data only (see methods.md Section 5.2 for why: Philips and
    Toshiba scanners have different value distributions, so imputing across
    cohorts would let information leak between them). The same
    per-cohort-fitted imputer is then used to transform (never re-fit) the
    matching cohort's test rows.
    """
    train = train.copy()
    test = test.copy()
    for cohort in sorted(train["Cohort_group"].unique()):
        train_mask = train["Cohort_group"] == cohort
        test_mask = test["Cohort_group"] == cohort
        imputer = KNNImputer(n_neighbors=k_neighbors, weights="distance")
        train.loc[train_mask, feature_cols] = imputer.fit_transform(train.loc[train_mask, feature_cols])
        if test_mask.any():
            test.loc[test_mask, feature_cols] = imputer.transform(test.loc[test_mask, feature_cols])
    return train, test


def impute_and_scale_demographics(
    train: pd.DataFrame, test: pd.DataFrame, demo_cols: list[str] = DEMO_FEATURES
):
    """
    Fill missing demographics using TRAIN-ONLY medians (age/weight/height/
    BMI) and mode (sex), then scale with a StandardScaler fit on train only.
    Mirrors the CT imputation's leakage-safety, adapted for demographics
    (which are not cohort-specific, unlike CT features).
    """
    train = train.copy()
    test = test.copy()
    numeric_cols = [c for c in demo_cols if c != "sex_binary"]
    medians = train[numeric_cols].median()
    for col in numeric_cols:
        train[col] = train[col].fillna(medians[col])
        test[col] = test[col].fillna(medians[col])

    if "sex_binary" in demo_cols:
        sex_mode = train["sex_binary"].mode(dropna=True)
        sex_fill = sex_mode.iloc[0] if len(sex_mode) > 0 else 0.0
        train["sex_binary"] = train["sex_binary"].fillna(sex_fill)
        test["sex_binary"] = test["sex_binary"].fillna(sex_fill)

    scaler = StandardScaler()
    train[demo_cols] = scaler.fit_transform(train[demo_cols])
    test[demo_cols] = scaler.transform(test[demo_cols])
    return train, test


def scale_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]):
    """Fit a StandardScaler on train only; transform both train and test."""
    train = train.copy()
    test = test.copy()
    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    return train, test, scaler


# ---------------------------------------------------------------------------
# Step 4: helper for "same split" scripts that reuse another scenario's
# train/test patient membership and attach demographics onto it.
# ---------------------------------------------------------------------------


def attach_demographics_by_split_key(
    base_df: pd.DataFrame, raw_clean_df: pd.DataFrame, demo_cols: list[str] = DEMO_FEATURES
) -> pd.DataFrame:
    """
    Merge demographic columns from `raw_clean_df` (output of
    `load_and_clean_raw`) onto `base_df` (another scenario's already-built
    train or test split) by `split_key`. Used by every "*_same_split.py"
    script that adds or isolates demographics without re-deriving a split.

    Raises if any row fails to find a match — a silent partial merge here
    would be a leakage/correctness bug, not a recoverable warning.
    """
    if "split_key" not in base_df.columns:
        base_df = base_df.copy()
        base_df["split_key"] = base_df["Anonymize_ID"].astype(str) + "|" + base_df["hip_side"].astype(str)

    keep_cols = ["split_key"] + demo_cols
    merged = base_df.merge(raw_clean_df[keep_cols], on="split_key", how="left", validate="one_to_one")
    missing = merged[demo_cols].isna().all(axis=1).sum()
    if missing > 0:
        raise ValueError(f"Merge failed for {missing} rows: no demographics found by split_key.")
    return merged
