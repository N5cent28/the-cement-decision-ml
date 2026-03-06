"""
Step 4C: Model Training & Evaluation (Majority Ground Truth + Demographics)
============================================================================
Runs the same model family as the original majority-vote analysis, but with
demographic features included. Uses the exact same row split as `processed_data`.

Input:
  processed_data_demographics_same_split/

Output:
  results_majority_with_demographics_same_split/

Run: python 04c_train_evaluate_majority_with_demographics.py
"""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC

matplotlib.use("Agg")

PROCESSED_DIR = "processed_data_demographics_same_split"
RESULTS_DIR = "results_majority_with_demographics_same_split"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TARGET_COL = "gt_majority"
N_CV_FOLDS = 5
RANDOM_STATE = 42


def load_processed():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    return train, test, meta["feature_columns"], meta


def prepare_xy(df, feature_cols, target_col=TARGET_COL):
    valid = df[target_col].notna()
    X = df.loc[valid, feature_cols].values
    y = df.loc[valid, target_col].astype(int).values
    groups = df.loc[valid, "Anonymize_ID"].values
    cohorts = df.loc[valid, "Cohort_group"].values
    return X, y, groups, cohorts


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def cross_validate_models(models, X, y, groups):
    print("=" * 72)
    print("GROUPED CROSS-VALIDATION (majority + demographics)")
    print("=" * 72)
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="roc_auc")
        results[name] = {
            "mean_auc": round(scores.mean(), 4),
            "std_auc": round(scores.std(), 4),
            "fold_scores": scores.tolist(),
        }
        print(f"  {name:<25s} AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
    print()
    return results


def evaluate_on_test(models, X_train, y_train, X_test, y_test):
    print("=" * 72)
    print("TEST SET EVALUATION (majority + demographics)")
    print("=" * 72)
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        }
        results[name] = {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}

        print(f"\n  {name}:")
        for k, v in metrics.items():
            print(f"    {k:<12s} = {v}")
    print()
    return results


def evaluate_by_cohort(models, X_test, y_test, cohorts_test):
    print("=" * 72)
    print("COHORT-SPECIFIC TEST PERFORMANCE")
    print("=" * 72)
    out = {}
    for cohort in sorted(np.unique(cohorts_test)):
        mask = cohorts_test == cohort
        if mask.sum() < 2:
            continue
        X_c = X_test[mask]
        y_c = y_test[mask]
        out[cohort] = {}

        print(f"\n  {cohort} ({mask.sum()} test hips):")
        for name, model in models.items():
            y_pred = model.predict(X_c)
            y_proba = model.predict_proba(X_c)[:, 1]
            acc = accuracy_score(y_c, y_pred)
            auc = roc_auc_score(y_c, y_proba) if len(np.unique(y_c)) > 1 else float("nan")
            out[cohort][name] = {
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4) if not np.isnan(auc) else "N/A",
                "n_samples": int(mask.sum()),
            }
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A (single class)"
            print(f"    {name:<25s} Acc={acc:.4f}  AUC={auc_str}")
    print()
    return out


def plot_roc(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_title("ROC Curves — Majority + Demographics")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curves to {path}")


def plot_pr(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(r, p, linewidth=2, label=f"{name} (PR-AUC={res['metrics']['pr_auc']:.3f})")
    ax.set_title("PR Curves — Majority + Demographics")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pr_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved PR curves to {path}")


def plot_confusions(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-cem.", "Cemented"],
            yticklabels=["Non-cem.", "Cemented"],
            ax=ax,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.suptitle("Confusion Matrices — Majority + Demographics", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices to {path}")


def plot_metric_bars(results):
    rows = []
    for name, res in results.items():
        for metric, val in res["metrics"].items():
            rows.append({"Model": name, "Metric": metric.upper().replace("_", "-"), "Value": val})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Metrics — Majority + Demographics")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved metric comparison to {path}")


def save_results(cv_results, test_results, cohort_results, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "analysis_variant": "majority_with_demographics_same_split",
        "cross_validation": cv_results,
        "test_set": {k: v["metrics"] for k, v in test_results.items()},
        "cohort_specific": cohort_results,
        "preprocessing": {
            "n_train": meta["n_train_rows"],
            "n_test": meta["n_test_rows"],
            "n_features_total": len(meta["feature_columns"]),
            "n_demographic_features": len(meta.get("demographic_feature_columns", [])),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "model_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved numeric results to {json_path}")

    summary_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("THA Cement Decision — Majority + Demographics Summary\n")
        f.write("=" * 58 + "\n\n")
        f.write("Cross-validation (grouped ROC-AUC):\n")
        for name, cv in cv_results.items():
            f.write(f"  {name}: {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}\n")
        f.write("\nTest set metrics:\n")
        for name, res in test_results.items():
            f.write(f"\n  {name}:\n")
            for k, v in res["metrics"].items():
                f.write(f"    {k}: {v}\n")
    print(f"  Saved summary to {summary_path}\n")


def main():
    print("=" * 72)
    print("STEP 4C: MAJORITY-VOTE TRAINING WITH DEMOGRAPHICS (SAME SPLIT)")
    print("=" * 72 + "\n")

    train, test, feature_cols, meta = load_processed()
    X_train, y_train, groups_train, _ = prepare_xy(train, feature_cols)
    X_test, y_test, _, cohorts_test = prepare_xy(test, feature_cols)

    print(f"Features used: {len(feature_cols)}")
    print(f"Train samples: {len(y_train)} | Test samples: {len(y_test)}")
    print(f"Demographic features included: {meta.get('demographic_feature_columns', [])}\n")

    models = build_models()
    cv_results = cross_validate_models(models, X_train, y_train, groups_train)
    test_results = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    cohort_results = evaluate_by_cohort(models, X_test, y_test, cohorts_test)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating plots:")
    plot_roc(test_results, y_test)
    plot_pr(test_results, y_test)
    plot_confusions(test_results, y_test)
    plot_metric_bars(test_results)
    print()

    save_results(cv_results, test_results, cohort_results, meta)
    print("=" * 72)
    print("TRAINING COMPLETE (MAJORITY + DEMOGRAPHICS)")
    print("=" * 72)


if __name__ == "__main__":
    main()
