"""Supervised machine learning engine for virality prediction in ViralGraph (Audited Version).

Scientific Audit Enhancements:
1. Grouped train-test splitting: Test cascades occur exclusively on entirely unseen graph topologies.
2. Leakage-free inner-fold target thresholding: Virality threshold tau is learned independently
   inside each inner cross-validation fold using ONLY inner-training-fold reach observations.
   Validation and test folds NEVER influence tau.
3. Grouped K-Fold cross validation on the training split, reporting mean and std CV ROC-AUC and F1.
4. Separate reporting of outer test generalization performance vs. inner grouped CV metrics.
5. Dummy baseline model inclusion (majority class predictor).
6. Comprehensive evaluation metrics: ROC-AUC, Precision, Recall, F1, Confusion Matrix, and Regression R^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelEvaluationReport:
    """Consolidated performance metrics for a trained virality model."""

    model_name: str
    accuracy: float  # Outer test accuracy
    precision: float  # Outer test precision
    recall: float  # Outer test recall
    f1: float  # Outer test F1
    roc_auc: float  # Outer test ROC-AUC
    cv_mean_roc_auc: float  # Inner grouped CV mean ROC-AUC
    cv_std_roc_auc: float  # Inner grouped CV std ROC-AUC
    cv_mean_f1: float  # Inner grouped CV mean F1
    confusion_mat: list[list[int]]  # Outer test confusion matrix [[TN, FP], [FN, TP]]
    feature_importances: dict[str, float]


def evaluate_models_with_leakage_guards(
    df_x: pd.DataFrame,
    s_reach: pd.Series,
    s_groups: pd.Series,
    percentile_threshold: float = 75.0,
    test_graph_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[dict[str, ModelEvaluationReport], dict[str, float]]:
    """Train and evaluate models under strict grouped-splitting and leakage-free thresholding.

    Methodological Guards:
    - Outer Split: GroupShuffleSplit partitions graph topologies into Disjoint Train Graphs and Test Graphs.
    - Inner Grouped CV: Inside each fold, tau_fold is computed strictly on inner-train reaches.
      Neither validation nor outer test cascades contaminate threshold determination.
    - Outer Evaluation: Final models are fit on full train split using tau_train_full and tested on held-out test graphs.

    Args:
        df_x: Feature dataframe including 'graph_id'.
        s_reach: Continuous final cascade reach.
        s_groups: Series of graph IDs.
        percentile_threshold: Percentile of train reach to define viral class (e.g., 75.0).
        test_graph_fraction: Fraction of unique graphs held out for testing.
        seed: Random seed.

    Returns:
        tuple: (classification_reports_dict, regression_metrics_dict)
    """
    feature_cols = [c for c in df_x.columns if c != "graph_id"]
    X_raw = df_x[feature_cols].copy()

    # 1. Grouped Train-Test Split by Graph ID (Outer Split)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_graph_fraction, random_state=seed)
    train_idx, test_idx = next(gss.split(X_raw, s_reach, groups=s_groups))

    x_train = X_raw.iloc[train_idx].copy().reset_index(drop=True)
    x_test = X_raw.iloc[test_idx].copy().reset_index(drop=True)
    reach_train = s_reach.iloc[train_idx].copy().reset_index(drop=True)
    reach_test = s_reach.iloc[test_idx].copy().reset_index(drop=True)
    groups_train = s_groups.iloc[train_idx].copy().reset_index(drop=True)

    # 2. Outer Target Thresholding (Fit exclusively on outer training set)
    tau_train_outer = float(np.percentile(reach_train, percentile_threshold))
    y_train_outer = (reach_train >= tau_train_outer).astype(int)
    y_test_outer = (reach_test >= tau_train_outer).astype(int)

    # 3. Model Suite Definition
    models: dict[str, Any] = {
        "Majority Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=seed, max_iter=1000, C=1.0)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, max_depth=4),
        "Random Forest": RandomForestClassifier(random_state=seed, n_estimators=100, max_depth=5),
    }

    reports: dict[str, ModelEvaluationReport] = {}
    n_splits = min(5, len(groups_train.unique()))
    gkf = GroupKFold(n_splits=n_splits)

    for name, model in models.items():
        # --- Inner Grouped Cross-Validation with Zero Target Leakage ---
        cv_rocs: list[float] = []
        cv_f1s: list[float] = []

        for cv_tr_idx, cv_val_idx in gkf.split(x_train, reach_train, groups=groups_train):
            cv_xtr, cv_xval = x_train.iloc[cv_tr_idx], x_train.iloc[cv_val_idx]
            reach_cv_tr = reach_train.iloc[cv_tr_idx]
            reach_cv_val = reach_train.iloc[cv_val_idx]

            # Re-fit threshold strictly inside this inner training fold!
            tau_fold = float(np.percentile(reach_cv_tr, percentile_threshold))
            cv_ytr = (reach_cv_tr >= tau_fold).astype(int)
            cv_yval = (reach_cv_val >= tau_fold).astype(int)

            if len(np.unique(cv_ytr)) < 2 or len(np.unique(cv_yval)) < 2:
                # If only 1 class in fold, skip AUC calculation for this fold
                continue

            # Clone model for fold evaluation
            if "Logistic" in name:
                model_clone = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(random_state=seed, max_iter=1000, C=1.0)),
                    ]
                )
            elif name == "Majority Baseline":
                model_clone = DummyClassifier(strategy="most_frequent")
            else:
                model_clone = model.__class__(**model.get_params())

            model_clone.fit(cv_xtr, cv_ytr)
            val_preds = model_clone.predict(cv_xval)
            cv_f1s.append(float(f1_score(cv_yval, val_preds, zero_division=0)))

            if hasattr(model_clone, "predict_proba"):
                val_prob = model_clone.predict_proba(cv_xval)[:, 1]
                cv_rocs.append(float(roc_auc_score(cv_yval, val_prob)))
            else:
                cv_rocs.append(0.5)

        cv_mean_roc = float(np.mean(cv_rocs)) if cv_rocs else 0.5
        cv_std_roc = float(np.std(cv_rocs)) if len(cv_rocs) > 1 else 0.0
        cv_mean_f1 = float(np.mean(cv_f1s)) if cv_f1s else 0.0

        # --- Outer Model Training on Full Train Split ---
        model.fit(x_train, y_train_outer)

        # --- Outer Model Evaluation on Held-Out Unseen Test Graphs ---
        y_pred = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
            try:
                roc_auc = float(roc_auc_score(y_test_outer, y_proba))
            except ValueError:
                roc_auc = 0.5
        else:
            roc_auc = 0.5

        acc = float(accuracy_score(y_test_outer, y_pred))
        prec = float(precision_score(y_test_outer, y_pred, zero_division=0))
        rec = float(recall_score(y_test_outer, y_pred, zero_division=0))
        f1 = float(f1_score(y_test_outer, y_pred, zero_division=0))
        cm = confusion_matrix(y_test_outer, y_pred).tolist()

        # Extract feature importances
        feat_imp: dict[str, float] = {}
        if name == "Logistic Regression":
            coefs = model.named_steps["clf"].coef_[0]
            for f_name, c in zip(feature_cols, coefs):
                feat_imp[f_name] = float(abs(c))
        elif hasattr(model, "feature_importances_"):
            for f_name, imp in zip(feature_cols, model.feature_importances_):
                feat_imp[f_name] = float(imp)

        sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

        reports[name] = ModelEvaluationReport(
            model_name=name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=roc_auc,
            cv_mean_roc_auc=cv_mean_roc,
            cv_std_roc_auc=cv_std_roc,
            cv_mean_f1=cv_mean_f1,
            confusion_mat=cm,
            feature_importances=sorted_imp,
        )

    # 4. Continuous Reach Regressor (Grouped evaluation across unseen test graphs)
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed)
    rf_reg.fit(x_train, reach_train)
    reg_preds = rf_reg.predict(x_test)

    reg_mse = float(mean_squared_error(reach_test, reg_preds))
    reg_r2 = float(r2_score(reach_test, reg_preds))

    reg_metrics = {
        "mse": reg_mse,
        "rmse": float(np.sqrt(reg_mse)),
        "r2_score": reg_r2,
        "tau_train_threshold": tau_train_outer,
        "test_graphs_count": len(s_groups.iloc[test_idx].unique()),
        "train_graphs_count": len(s_groups.iloc[train_idx].unique()),
    }

    return reports, reg_metrics
