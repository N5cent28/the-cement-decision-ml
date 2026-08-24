"""
Scenario 10 Training/Evaluation: Original operating surgeon's decision, CT-only.

Run:
  python 04j_train_evaluate_original_ct_only.py
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

PROCESSED_DIR = "processed_data_original_ct_only"
RESULTS_DIR = "results_original_ct_only"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TARGET_COL = "gt_original"
N_CV_FOLDS = 5
RANDOM_STATE = 42


def load_processed():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    with open(os.path.join(PROCESSED_DIR, "preprocessing_metadata.json")) as f:
        meta = json.load(f)
    return train, test, meta["feature_columns"], meta


def prepare_xy(df, feature_cols):
    valid = df[TARGET_COL].notna()
    X = df.loc[valid, feature_cols].values
    y = df.loc[valid, TARGET_COL].astype(int).values
    groups = df.loc[valid, "Anonymize_ID"].values
    cohorts = df.loc[valid, "Cohort_group"].values
    return X, y, groups, cohorts


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def cross_validate(models, X, y, groups):
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    out = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="roc_auc")
        out[name] = {
            "mean_auc": round(scores.mean(), 4),
            "std_auc": round(scores.std(), 4),
            "fold_scores": scores.tolist(),
        }
    return out


def evaluate(models, X_train, y_train, X_test, y_test):
    out = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        out[name] = {
            "metrics": {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
                "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            },
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    return out


def eval_by_cohort(models, X_test, y_test, cohorts):
    out = {}
    for cohort in sorted(np.unique(cohorts)):
        mask = cohorts == cohort
        X_c = X_test[mask]
        y_c = y_test[mask]
        if len(y_c) < 2:
            continue
        out[cohort] = {}
        for name, model in models.items():
            y_pred = model.predict(X_c)
            y_proba = model.predict_proba(X_c)[:, 1]
            auc = roc_auc_score(y_c, y_proba) if len(np.unique(y_c)) > 1 else float("nan")
            out[cohort][name] = {
                "accuracy": round(accuracy_score(y_c, y_pred), 4),
                "roc_auc": round(auc, 4) if not np.isnan(auc) else "N/A",
                "n_samples": int(len(y_c)),
            }
    return out


def plot_outputs(results, y_test):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Scenario 10 ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(r, p, label=f"{name} (PR-AUC={res['metrics']['pr_auc']:.3f})")
    ax.set_title("Scenario 10 PR Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pr_curves.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, res in results.items():
        for metric, val in res["metrics"].items():
            rows.append({"Model": name, "Metric": metric.upper().replace("_", "-"), "Value": val})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Metric", y="Value", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Scenario 10 Metric Comparison")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "metric_comparison.png"), dpi=150)
    plt.close(fig)


def save_results(cv_res, test_res, cohort_res, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "analysis_variant": meta.get("analysis_variant", "scenario10_original_ct_only"),
        "cross_validation": cv_res,
        "test_set": {k: v["metrics"] for k, v in test_res.items()},
        "cohort_specific": cohort_res,
        "preprocessing": {
            "n_train": meta["n_train_rows"],
            "n_test": meta["n_test_rows"],
            "n_features": len(meta["feature_columns"]),
            "split_stratification_mode": meta.get("split_stratification_mode", "not_recorded"),
        },
    }
    with open(os.path.join(RESULTS_DIR, "model_results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "results_summary.txt"), "w") as f:
        f.write("Scenario 10 (Original surgeon target, CT-only) Results\n")
        f.write("=" * 57 + "\n\n")
        for name, cv in cv_res.items():
            f.write(f"{name} CV AUC: {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}\n")
        f.write("\nTest metrics:\n")
        for name, res in test_res.items():
            f.write(f"\n{name}:\n")
            for k, v in res["metrics"].items():
                f.write(f"  {k}: {v}\n")


def main():
    train, test, feat_cols, meta = load_processed()
    X_train, y_train, groups_train, _ = prepare_xy(train, feat_cols)
    X_test, y_test, _, cohorts_test = prepare_xy(test, feat_cols)
    models = build_models()
    cv_res = cross_validate(models, X_train, y_train, groups_train)
    test_res = evaluate(models, X_train, y_train, X_test, y_test)
    cohort_res = eval_by_cohort(models, X_test, y_test, cohorts_test)
    plot_outputs(test_res, y_test)
    save_results(cv_res, test_res, cohort_res, meta)
    print(f"Saved scenario 10 outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
