"""Train and evaluate the two models.

* Classifier  -> performance level (At-Risk / Average / High)
* Regressor   -> final score (0-100)

Several candidates are compared with cross-validation on the training split; the
best one is refit, scored on the held-out test split and saved with joblib.
"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import (GradientBoostingClassifier,  # noqa: E402
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402
from src.preprocessing import build_preprocessor, load_data, split_data  # noqa: E402

RS = config.RANDOM_STATE


def _classifiers(fast: bool) -> dict:
    n = 100 if fast else 300
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=n, random_state=RS,
                                               class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=RS),
    }


def _regressors(fast: bool) -> dict:
    n = 100 if fast else 300
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=n, random_state=RS),
        "GradientBoosting": GradientBoostingRegressor(random_state=RS),
    }


def _pipe(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def _plot_confusion(cm, labels, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_actual_vs_pred(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    ax.plot([0, 100], [0, 100], "r--", lw=1)
    ax.set_xlabel("Actual final score")
    ax.set_ylabel("Predicted final score")
    ax.set_title("Regression: actual vs predicted (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_training(df=None, models_dir=config.MODELS_DIR,
                 reports_dir=config.REPORTS_DIR, fast: bool = False) -> dict:
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    fig_dir = reports_dir / "figures"
    for d in (models_dir, reports_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_data()
    train_df, test_df = split_data(df)
    X_train, X_test = train_df[config.FEATURES], test_df[config.FEATURES]

    metrics = {"n_train": len(train_df), "n_test": len(test_df)}

    # ---------------- classification ----------------
    y_train, y_test = train_df[config.TARGET_LEVEL], test_df[config.TARGET_LEVEL]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    cv_scores = {}
    for name, est in _classifiers(fast).items():
        cv_scores[name] = float(cross_val_score(
            _pipe(est), X_train, y_train, cv=cv, scoring="f1_macro").mean())
    best_name = max(cv_scores, key=cv_scores.get)
    clf = _pipe(_classifiers(fast)[best_name]).fit(X_train, y_train)
    pred = clf.predict(X_test)
    labels = list(clf.classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)
    _plot_confusion(cm, labels, fig_dir / "confusion_matrix.png")
    metrics["classification"] = {
        "cv_f1_macro": cv_scores,
        "best_model": best_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(y_test, pred, output_dict=True),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    joblib.dump(clf, models_dir / "classifier.joblib")

    # ---------------- regression ----------------
    y_train_r, y_test_r = train_df[config.TARGET_SCORE], test_df[config.TARGET_SCORE]
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    cv_rmse = {}
    for name, est in _regressors(fast).items():
        cv_rmse[name] = float(-cross_val_score(
            _pipe(est), X_train, y_train_r, cv=kf,
            scoring="neg_root_mean_squared_error").mean())
    best_r = min(cv_rmse, key=cv_rmse.get)
    reg = _pipe(_regressors(fast)[best_r]).fit(X_train, y_train_r)
    pred_r = np.clip(reg.predict(X_test), 0, 100)
    _plot_actual_vs_pred(y_test_r, pred_r, fig_dir / "regression_actual_vs_pred.png")
    metrics["regression"] = {
        "cv_rmse": cv_rmse,
        "best_model": best_r,
        "mae": float(mean_absolute_error(y_test_r, pred_r)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_r, pred_r))),
        "r2": float(r2_score(y_test_r, pred_r)),
    }
    joblib.dump(reg, models_dir / "regressor.joblib")

    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
