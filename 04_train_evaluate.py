"""
Step 4: Model Training & Evaluation
====================================
Train four classifiers on the preprocessed training set:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. Gradient Boosting
  4. SVM (RBF kernel)

Use patient-group-aware cross-validation for model selection.
Evaluate on the held-out test set.
Generate comparative plots and a concise results summary.

Run:  python 04_train_evaluate.py
"""

import pandas as pd
import numpy as np
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = "processed_data"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

TARGET_COL = "gt_majority"
N_CV_FOLDS = 5
RANDOM_STATE = 42


def load_processed():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    return train, test, feature_cols, meta


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
            n_estimators=200, max_depth=None, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
    }


def cross_validate_models(models, X, y, groups):
    """Grouped k-fold CV to get training-phase performance estimates."""
    print("=" * 60)
    print("GROUPED CROSS-VALIDATION (training set)")
    print("=" * 60)

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=gkf, groups=groups, scoring="roc_auc"
        )
        cv_results[name] = {
            "mean_auc": round(scores.mean(), 4),
            "std_auc": round(scores.std(), 4),
            "fold_scores": scores.tolist(),
        }
        print(f"  {name:<25s}  AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    print()
    return cv_results


def evaluate_on_test(models, X_train, y_train, X_test, y_test):
    """Fit on full training set, evaluate on held-out test set."""
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

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
        results[name] = {
            "metrics": metrics,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        print(f"\n  {name}:")
        for k, v in metrics.items():
            print(f"    {k:<12s} = {v}")

    print()
    return results


def evaluate_by_cohort(models, X_test, y_test, cohorts_test, feature_cols):
    """Report metrics broken out by cohort on the test set."""
    print("=" * 60)
    print("COHORT-SPECIFIC TEST PERFORMANCE")
    print("=" * 60)

    cohort_results = {}
    for cohort in sorted(np.unique(cohorts_test)):
        mask = cohorts_test == cohort
        if mask.sum() < 2:
            continue

        X_c = X_test[mask]
        y_c = y_test[mask]
        print(f"\n  {cohort} ({mask.sum()} test hips):")

        cohort_results[cohort] = {}
        for name, model in models.items():
            y_pred = model.predict(X_c)
            y_proba = model.predict_proba(X_c)[:, 1]

            n_classes = len(np.unique(y_c))
            acc = accuracy_score(y_c, y_pred)
            if n_classes > 1:
                auc = roc_auc_score(y_c, y_proba)
            else:
                auc = float("nan")

            cohort_results[cohort][name] = {
                "accuracy": round(acc, 4),
                "roc_auc": round(auc, 4) if not np.isnan(auc) else "N/A",
                "n_samples": int(mask.sum()),
            }
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A (single class)"
            print(f"    {name:<25s}  Acc={acc:.4f}  AUC={auc_str}")

    print()
    return cohort_results


def plot_roc_curves(results, y_test):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = res["metrics"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Test Set", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC curves to {path}")


def plot_confusion_matrices(results, y_test):
    """Grid of confusion matrices."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    labels = ["Non-cem.", "Cemented"]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.suptitle("Confusion Matrices — Test Set", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices to {path}")


def plot_metric_comparison(results):
    """Bar chart comparing key metrics across models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    model_names = list(results.keys())

    data = []
    for name in model_names:
        for metric in metrics_to_plot:
            data.append({
                "Model": name,
                "Metric": metric.upper().replace("_", "-"),
                "Value": results[name]["metrics"][metric],
            })
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Metric", y="Value", hue="Model", ax=ax)
    ax.set_ylim([0, 1.05])
    ax.set_title("Model Performance Comparison — Test Set", fontsize=14)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "metric_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved metric comparison to {path}")


def plot_pr_curves(results, y_test):
    """Precision-Recall curves (useful if class imbalance exists)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
        pr_auc = res["metrics"]["pr_auc"]
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})", linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Test Set", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "pr_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved PR curves to {path}")


def save_results(cv_results, test_results, cohort_results, meta):
    """Save all results to JSON and a summary text file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON with numeric results
    output = {
        "cross_validation": cv_results,
        "test_set": {
            name: res["metrics"] for name, res in test_results.items()
        },
        "cohort_specific": cohort_results,
        "preprocessing": {
            "n_train": meta["n_train_rows"],
            "n_test": meta["n_test_rows"],
            "n_features": len(meta["feature_columns"]),
        },
    }
    json_path = os.path.join(RESULTS_DIR, "model_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved numeric results to {json_path}")

    # Human-readable summary
    summary_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("THA Cement Decision — Model Results Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Cross-Validation (Grouped K-Fold, training set):\n")
        for name, cv in cv_results.items():
            f.write(f"  {name}: AUC = {cv['mean_auc']:.4f} "
                    f"+/- {cv['std_auc']:.4f}\n")

        f.write("\nTest Set Performance:\n")
        for name, res in test_results.items():
            m = res["metrics"]
            f.write(f"\n  {name}:\n")
            for k, v in m.items():
                f.write(f"    {k}: {v}\n")

        if cohort_results:
            f.write("\nCohort-Specific Test Performance:\n")
            for cohort, models in cohort_results.items():
                f.write(f"\n  {cohort}:\n")
                for name, cm in models.items():
                    f.write(f"    {name}: Acc={cm['accuracy']}, "
                            f"AUC={cm['roc_auc']}\n")

    print(f"  Saved summary to {summary_path}\n")


def print_feature_importance(models, feature_cols):
    """Print feature importances for tree-based models."""
    print("=" * 60)
    print("FEATURE IMPORTANCE (tree-based models)")
    print("=" * 60)

    for name in ["Random Forest", "Gradient Boosting"]:
        if name in models:
            model = models[name]
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print(f"\n  {name} — Top 10 features:")
            for rank, idx in enumerate(sorted_idx[:10], 1):
                print(f"    {rank:>2d}. {feature_cols[idx]:<40s}  "
                      f"{importances[idx]:.4f}")
    print()


def main():
    print("=" * 60)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 60 + "\n")

    train, test, feature_cols, meta = load_processed()

    X_train, y_train, groups_train, cohorts_train = prepare_xy(train, feature_cols)
    X_test, y_test, groups_test, cohorts_test = prepare_xy(test, feature_cols)

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(y_train)} samples, "
          f"class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test:  {len(y_test)} samples, "
          f"class balance: {dict(zip(*np.unique(y_test, return_counts=True)))}\n")

    models = build_models()

    cv_results = cross_validate_models(models, X_train, y_train, groups_train)
    test_results = evaluate_on_test(models, X_train, y_train, X_test, y_test)
    cohort_results = evaluate_by_cohort(
        models, X_test, y_test, cohorts_test, feature_cols
    )

    print_feature_importance(models, feature_cols)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating plots:")
    plot_roc_curves(test_results, y_test)
    plot_confusion_matrices(test_results, y_test)
    plot_metric_comparison(test_results)
    plot_pr_curves(test_results, y_test)
    print()

    save_results(cv_results, test_results, cohort_results, meta)

    print("=" * 60)
    print("TRAINING & EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
