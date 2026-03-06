"""
Step 5: Compare All Four ML Training Exercises
===============================================
Builds comparative tables/figures across:
  1) Majority vote, CT-only                         -> results/
  2) Vote-fraction target (regression framing)      -> results_vote_fraction/
  3) Majority vote, CT + demographics (same split)  -> results_majority_with_demographics_same_split/
  4) Majority vote, demographics-only (same split)  -> results_majority_demographics_only_same_split/

Outputs:
  results_comparison_all_experiments/
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

OUT_DIR = "results_comparison_all_experiments"
FIG_DIR = os.path.join(OUT_DIR, "figures")

PATHS = {
    "majority_ct_only": "results/model_results.json",
    "vote_fraction": "results_vote_fraction/model_results.json",
    "majority_ct_plus_demo": "results_majority_with_demographics_same_split/model_results.json",
    "majority_demo_only": "results_majority_demographics_only_same_split/model_results.json",
}


def read_json(path):
    with open(path) as f:
        return json.load(f)


def build_majority_table(data):
    rows = []
    label_map = {
        "majority_ct_only": "Majority: CT-only",
        "majority_ct_plus_demo": "Majority: CT + demographics",
        "majority_demo_only": "Majority: demographics-only",
    }
    for key in ["majority_ct_only", "majority_ct_plus_demo", "majority_demo_only"]:
        test_set = data[key]["test_set"]
        for model, metrics in test_set.items():
            rows.append(
                {
                    "experiment_key": key,
                    "experiment": label_map[key],
                    "model": model,
                    "accuracy": metrics["accuracy"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                }
            )
    return pd.DataFrame(rows)


def build_vote_fraction_table(data):
    rows = []
    for model, metrics in data["vote_fraction"]["test_set"].items():
        rows.append({"experiment": "Vote-fraction target", "model": model, **metrics})
    return pd.DataFrame(rows)


def plot_majority_comparison(df):
    # Fixed model order for readability
    model_order = ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM (RBF)"]
    df = df.copy()
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["model", "experiment"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric, title in zip(
        axes,
        ["roc_auc", "accuracy", "pr_auc"],
        ["ROC-AUC", "Accuracy", "PR-AUC"],
    ):
        sns.barplot(data=df, x="model", y=metric, hue="experiment", ax=ax)
        ax.set_title(f"Majority-Vote Comparison: {title}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        if ax is not axes[0]:
            ax.get_legend().remove()
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "majority_experiments_comparison.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_vote_fraction(df):
    metrics = ["rmse", "mae", "r2", "binary_accuracy_at_0p5", "binary_roc_auc_vs_thresholded_truth"]
    plot_df = df.melt(id_vars=["experiment", "model"], value_vars=metrics, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model", ax=ax)
    ax.set_title("Vote-Fraction Exercise: Regression and Thresholded Classification Metrics")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "vote_fraction_experiment_metrics.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def build_summary(majority_df, vote_df):
    # Best logistic regression row across majority experiments (for direct feature-set comparison)
    lr_rows = majority_df[majority_df["model"] == "Logistic Regression"].copy()
    lr_best = lr_rows.sort_values("roc_auc", ascending=False).iloc[0]
    lr_ct = lr_rows[lr_rows["experiment_key"] == "majority_ct_only"].iloc[0]
    lr_ct_demo = lr_rows[lr_rows["experiment_key"] == "majority_ct_plus_demo"].iloc[0]
    lr_demo_only = lr_rows[lr_rows["experiment_key"] == "majority_demo_only"].iloc[0]

    # Best model per majority experiment (by ROC-AUC)
    best_rows = (
        majority_df.sort_values("roc_auc", ascending=False)
        .groupby("experiment", as_index=False)
        .first()
        .loc[:, ["experiment", "model", "roc_auc", "accuracy", "pr_auc", "f1"]]
    )

    # Vote-fraction best regression by lowest RMSE
    vote_best = vote_df.sort_values("rmse", ascending=True).iloc[0]

    lines = []
    lines.append("# Comparative Summary Across All Four Exercises")
    lines.append("")
    lines.append("## Majority-vote family (directly comparable)")
    lines.append(
        f"- Logistic Regression ROC-AUC: CT-only {lr_ct['roc_auc']:.4f} -> "
        f"CT + demographics {lr_ct_demo['roc_auc']:.4f} "
        f"(delta {lr_ct_demo['roc_auc'] - lr_ct['roc_auc']:+.4f})"
    )
    lines.append(
        f"- Logistic Regression ROC-AUC: CT-only {lr_ct['roc_auc']:.4f} -> "
        f"demographics-only {lr_demo_only['roc_auc']:.4f} "
        f"(delta {lr_demo_only['roc_auc'] - lr_ct['roc_auc']:+.4f})"
    )
    lines.append(
        f"- Logistic Regression Accuracy: CT-only {lr_ct['accuracy']:.4f} -> "
        f"CT + demographics {lr_ct_demo['accuracy']:.4f}"
    )
    lines.append("")
    lines.append("Best model by ROC-AUC in each majority experiment:")
    for _, r in best_rows.iterrows():
        lines.append(
            f"- {r['experiment']}: {r['model']} (ROC-AUC {r['roc_auc']:.4f}, "
            f"Accuracy {r['accuracy']:.4f}, PR-AUC {r['pr_auc']:.4f})"
        )
    lines.append("")
    lines.append("## Vote-fraction exercise (different target, not directly equivalent)")
    lines.append(
        f"- Best RMSE: {vote_best['model']} (RMSE {vote_best['rmse']:.4f}, "
        f"MAE {vote_best['mae']:.4f}, R2 {vote_best['r2']:.4f})"
    )
    lines.append(
        "- Because vote-fraction uses a continuous target, it is best compared via RMSE/MAE/R2 "
        "rather than directly against majority-vote ROC-AUC."
    )
    lines.append("")
    return "\n".join(lines), best_rows, vote_best


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = {k: read_json(v) for k, v in PATHS.items()}

    majority_df = build_majority_table(data)
    vote_df = build_vote_fraction_table(data)

    majority_csv = os.path.join(OUT_DIR, "majority_experiments_metrics_by_model.csv")
    vote_csv = os.path.join(OUT_DIR, "vote_fraction_metrics_by_model.csv")
    majority_df.to_csv(majority_csv, index=False)
    vote_df.to_csv(vote_csv, index=False)

    fig_majority = plot_majority_comparison(majority_df)
    fig_vote = plot_vote_fraction(vote_df)

    summary_text, best_rows, vote_best = build_summary(majority_df, vote_df)
    summary_md = os.path.join(OUT_DIR, "comparison_summary.md")
    with open(summary_md, "w") as f:
        f.write(summary_text)

    best_rows_csv = os.path.join(OUT_DIR, "majority_best_models_by_experiment.csv")
    best_rows.to_csv(best_rows_csv, index=False)

    vote_best_csv = os.path.join(OUT_DIR, "vote_fraction_best_model.csv")
    pd.DataFrame([vote_best]).to_csv(vote_best_csv, index=False)

    print("Saved comparison artifacts:")
    print(f"  {majority_csv}")
    print(f"  {vote_csv}")
    print(f"  {best_rows_csv}")
    print(f"  {vote_best_csv}")
    print(f"  {summary_md}")
    print(f"  {fig_majority}")
    print(f"  {fig_vote}")


if __name__ == "__main__":
    main()
