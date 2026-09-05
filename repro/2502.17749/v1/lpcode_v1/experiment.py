"""Fixed-model construction and leakage-safe single-fold evaluation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_model(model_name: str, seed: int) -> Pipeline:
    """Build one of the fixed V1 classifiers behind a standardizing pipeline."""
    if model_name == "mlp":
        model = MLPClassifier(random_state=seed)
    elif model_name == "xgb":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"unknown model name: {model_name!r}")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _validate_features(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 2-D numeric array") from exc
    if (
        array.ndim != 2
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a finite 2-D numeric array")
    try:
        finite = np.isfinite(array).all()
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite 2-D numeric array") from exc
    if not finite:
        raise ValueError(f"{name} must be a finite 2-D numeric array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    return array


def _validate_labels(values: Any, name: str, expected_rows: int) -> np.ndarray:
    try:
        labels = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1-D binary label array") from exc
    if labels.ndim != 1 or labels.shape[0] != expected_rows:
        raise ValueError(f"{name} must be a 1-D binary label array matching its features")
    if np.issubdtype(labels.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real-valued binary labels")
    if labels.dtype != np.dtype(bool) and not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"{name} must be a 1-D binary label array")
    try:
        if not np.isfinite(labels).all():
            raise ValueError(f"{name} must contain finite binary labels")
    except TypeError as exc:
        raise ValueError(f"{name} must be a 1-D binary label array") from exc
    if labels.shape[0] == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return labels


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def evaluate_fold(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    """Fit a fixed model on one training fold and score its held-out test fold."""
    train_features = _validate_features(x_train, "x_train")
    test_features = _validate_features(x_test, "x_test")
    if train_features.shape[1] != test_features.shape[1]:
        raise ValueError("x_train and x_test must have the same number of features")
    train_labels = _validate_labels(y_train, "y_train", train_features.shape[0])
    test_labels = _validate_labels(y_test, "y_test", test_features.shape[0])
    if np.unique(train_labels).size != 2:
        raise ValueError("y_train must contain both binary classes")
    if np.unique(test_labels).size != 2:
        raise ValueError("y_test must contain both binary classes for AUROC")

    model = build_model(model_name, seed)
    fit_started = time.perf_counter()
    model.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)
    predict_seconds = time.perf_counter() - predict_started
    classes = np.asarray(model.classes_)
    positive_index = int(np.flatnonzero(classes == 1)[0])
    positive_probabilities = probabilities[:, positive_index]

    metrics = {
        "f1": float(f1_score(test_labels, predictions, zero_division=0)),
        "precision": float(precision_score(test_labels, predictions, zero_division=0)),
        "recall": float(recall_score(test_labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(test_labels, positive_probabilities)),
        "mcc": float(matthews_corrcoef(test_labels, predictions)),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("model produced a non-finite metric")
    return {
        **metrics,
        "fit_seconds": float(max(0.0, fit_seconds)),
        "predict_seconds": float(max(0.0, predict_seconds)),
        "train_rows": int(train_features.shape[0]),
        "test_rows": int(test_features.shape[0]),
        "train_class_counts": _class_counts(train_labels),
        "test_class_counts": _class_counts(test_labels),
    }
