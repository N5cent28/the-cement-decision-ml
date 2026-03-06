"""
Step 2: Inter-Surgeon Agreement Analysis
=========================================
Harmonize surgeon labels, compute pairwise percent agreement,
Fleiss' kappa for 3 raters, and categorize each hip by agreement
level (unanimous / majority-only / no majority).

Run:  python 02_surgeon_agreement.py
"""

import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import os

DATA_FILE = "Raw_data_03.03.2026.csv"
REPORT_DIR = "reports"

SURGEON_COLS = [
    "Cement_vs_noCement_original",
    "Halldor_decision",
    "3d_surg",
]

# Canonical binary mapping: 1 = cemented, 0 = non-cemented
LABEL_MAP = {
    "Cemented":      1,
    "cemented":      1,
    "Non-cemented":  0,
    "non-cemented":  0,
    "Uncemented":    0,
    "uncemented":    0,
}

# Values that indicate this rater's label is unavailable for this hip
EXCLUDED_VALUES = {"Already operated", "already operated"}


def load_and_harmonize():
    df = pd.read_csv(DATA_FILE)

    for col in SURGEON_COLS:
        df[col] = df[col].astype(str).str.strip()

    # Flag rows where any rater label is excluded/unavailable
    excluded_mask = pd.Series(False, index=df.index)
    for col in SURGEON_COLS:
        excluded_mask |= df[col].isin(EXCLUDED_VALUES)

    n_excluded = excluded_mask.sum()
    if n_excluded > 0:
        print(f"Excluding {n_excluded} rows with unavailable rater labels "
              f"(e.g. 'Already operated').\n")

    df = df[~excluded_mask].copy()

    # Map to binary
    binary_cols = []
    for col in SURGEON_COLS:
        bcol = col + "_bin"
        df[bcol] = df[col].map(LABEL_MAP)
        unmapped = df[bcol].isna() & df[col].notna()
        if unmapped.any():
            bad = df.loc[unmapped, col].unique()
            print(f"WARNING: unmapped values in {col}: {bad}")
        binary_cols.append(bcol)

    return df, binary_cols


def pairwise_agreement(df, binary_cols):
    print("=" * 60)
    print("PAIRWISE PERCENT AGREEMENT")
    print("=" * 60)
    col_names = [c.replace("_bin", "") for c in binary_cols]
    results = []
    for i in range(len(binary_cols)):
        for j in range(i + 1, len(binary_cols)):
            valid = df[binary_cols[i]].notna() & df[binary_cols[j]].notna()
            sub = df[valid]
            agree = (sub[binary_cols[i]] == sub[binary_cols[j]]).sum()
            total = len(sub)
            pct = 100 * agree / total if total > 0 else 0
            pair = f"{col_names[i]}  vs  {col_names[j]}"
            print(f"  {pair:<55s}  {agree}/{total}  ({pct:.1f}%)")
            results.append({"pair": pair, "agree": agree, "total": total,
                            "pct_agree": round(pct, 2)})
    print()
    return pd.DataFrame(results)


def compute_fleiss_kappa(df, binary_cols):
    print("=" * 60)
    print("FLEISS' KAPPA (3 raters)")
    print("=" * 60)

    # Build matrix: each row is a hip, each column is a rater's binary label
    valid = df[binary_cols].notna().all(axis=1)
    sub = df.loc[valid, binary_cols].astype(int).values

    table, _ = aggregate_raters(sub)
    kappa = fleiss_kappa(table, method="fleiss")
    print(f"  Fleiss' kappa = {kappa:.4f}")
    print(f"  (computed on {len(sub)} hips where all 3 raters provided labels)")
    print()

    interpretation = (
        "almost perfect" if kappa > 0.80 else
        "substantial" if kappa > 0.60 else
        "moderate" if kappa > 0.40 else
        "fair" if kappa > 0.20 else
        "slight" if kappa > 0.00 else
        "poor"
    )
    print(f"  Landis & Koch interpretation: {interpretation} agreement")
    print()
    return kappa


def case_level_agreement(df, binary_cols):
    print("=" * 60)
    print("CASE-LEVEL AGREEMENT CATEGORIES")
    print("=" * 60)

    valid = df[binary_cols].notna().all(axis=1)
    sub = df[valid].copy()
    votes = sub[binary_cols].astype(int)

    cemented_votes = votes.sum(axis=1)  # count of "cemented" across 3 raters

    sub["agreement_category"] = "no_majority"
    sub.loc[cemented_votes == 3, "agreement_category"] = "unanimous_cemented"
    sub.loc[cemented_votes == 0, "agreement_category"] = "unanimous_noncemented"
    sub.loc[cemented_votes == 2, "agreement_category"] = "majority_cemented"
    sub.loc[cemented_votes == 1, "agreement_category"] = "majority_noncemented"

    # Majority vote label
    sub["majority_vote"] = (cemented_votes >= 2).astype(int)

    # Vote proportion for probabilistic weighting
    sub["cemented_vote_fraction"] = cemented_votes / 3.0

    print("  Distribution of agreement categories:")
    vc = sub["agreement_category"].value_counts()
    for cat, cnt in vc.items():
        print(f"    {cat:<30s}  {cnt:>4d}  ({100*cnt/len(sub):.1f}%)")
    print()

    unanimous = sub["agreement_category"].str.startswith("unanimous").sum()
    majority_only = sub["agreement_category"].str.startswith("majority").sum()
    print(f"  Unanimous (3/3): {unanimous}  ({100*unanimous/len(sub):.1f}%)")
    print(f"  Majority  (2/3): {majority_only}  ({100*majority_only/len(sub):.1f}%)")
    print()

    return sub


def save_agreement_data(df_agreement, pairwise_df, kappa):
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Save per-hip agreement annotations
    out_cols = (["Anonymize_ID", "hip_side", "Cohort_group"]
                + [c + "_bin" for c in
                   ["Cement_vs_noCement_original", "Halldor_decision", "3d_surg"]]
                + ["agreement_category", "majority_vote", "cemented_vote_fraction"])
    agreement_path = os.path.join(REPORT_DIR, "02_hip_agreement.csv")
    df_agreement[out_cols].to_csv(agreement_path, index=False)
    print(f"Saved per-hip agreement data to {agreement_path}")

    # Save pairwise summary
    pairwise_path = os.path.join(REPORT_DIR, "02_pairwise_agreement.csv")
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"Saved pairwise agreement to {pairwise_path}")

    # Save kappa
    kappa_path = os.path.join(REPORT_DIR, "02_fleiss_kappa.txt")
    with open(kappa_path, "w") as f:
        f.write(f"Fleiss' kappa (3 raters, binary cemented/non-cemented): {kappa:.4f}\n")
    print(f"Saved Fleiss' kappa to {kappa_path}")
    print()


def main():
    df, binary_cols = load_and_harmonize()
    pairwise_df = pairwise_agreement(df, binary_cols)
    kappa = compute_fleiss_kappa(df, binary_cols)
    df_agreement = case_level_agreement(df, binary_cols)
    save_agreement_data(df_agreement, pairwise_df, kappa)

    print("=" * 60)
    print("AGREEMENT ANALYSIS COMPLETE — proceed to Step 3")
    print("=" * 60)


if __name__ == "__main__":
    main()
