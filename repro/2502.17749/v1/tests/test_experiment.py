from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lpcode_v1.experiment import build_model, evaluate_fold


X_TRAIN = np.asarray(
    [[0], [0.2], [0.4], [0.6], [3.0], [3.2], [3.4], [3.6]], dtype=float
)
Y_TRAIN = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
X_TEST = np.asarray([[0.1], [0.5], [3.1], [3.5]], dtype=float)
Y_TEST = np.asarray([0, 0, 1, 1])


def test_build_model_mlp_has_locked_pipeline_structure() -> None:
    pipeline = build_model("mlp", seed=17)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "model"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    model = pipeline.named_steps["model"]
    assert isinstance(model, MLPClassifier)
    assert model.random_state == 17


def test_build_model_xgb_has_locked_hyperparameters() -> None:
    pipeline = build_model("xgb", seed=23)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "model"]
    model = pipeline.named_steps["model"]
    assert model.__class__.__name__ == "XGBClassifier"
    params = model.get_params()
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 4
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["eval_metric"] == "logloss"
    assert params["random_state"] == 23
    assert params["n_jobs"] == 1


def test_build_model_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("does-not-exist", seed=1)


@pytest.mark.parametrize("model_name", ["mlp", "xgb"])
def test_evaluate_fold_returns_finite_json_serializable_metrics(model_name: str) -> None:
    with pytest.warns(ConvergenceWarning) if model_name == "mlp" else _no_warnings():
        result = evaluate_fold(
            X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, model_name=model_name, seed=5
        )

    expected_keys = {
        "f1",
        "precision",
        "recall",
        "auroc",
        "mcc",
        "fit_seconds",
        "predict_seconds",
        "train_rows",
        "test_rows",
        "train_class_counts",
        "test_class_counts",
    }
    assert expected_keys <= result.keys()
    for key in ("f1", "precision", "recall", "auroc", "mcc"):
        assert isinstance(result[key], float)
        assert np.isfinite(result[key])
    assert isinstance(result["fit_seconds"], float)
    assert isinstance(result["predict_seconds"], float)
    assert result["fit_seconds"] >= 0.0
    assert result["predict_seconds"] >= 0.0
    assert result["train_rows"] == 8
    assert result["test_rows"] == 4
    assert result["train_class_counts"] == {"0": 4, "1": 4}
    assert result["test_class_counts"] == {"0": 2, "1": 2}
    json.dumps(result)


class _DeterministicModel:
    classes_ = np.asarray([0, 1])

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "_DeterministicModel":
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return np.asarray([0, 1, 1, 0])

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        positive = np.asarray([0.1, 0.4, 0.7, 0.2])
        return np.column_stack([1.0 - positive, positive])


class _no_warnings:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


def test_evaluate_fold_wires_hard_predictions_and_positive_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lpcode_v1.experiment as experiment

    monkeypatch.setattr(experiment, "build_model", lambda model_name, seed: _DeterministicModel())
    result = evaluate_fold(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, model_name="fake", seed=5)

    assert result["f1"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["auroc"] == 0.75
    assert result["mcc"] == 0.0


def test_evaluate_fold_fits_scaler_on_training_data_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import lpcode_v1.experiment as experiment

    captured: list[Pipeline] = []
    original_build_model = experiment.build_model

    def capture_build_model(model_name: str, seed: int) -> Pipeline:
        pipeline = original_build_model(model_name, seed)
        captured.append(pipeline)
        return pipeline

    monkeypatch.setattr(experiment, "build_model", capture_build_model)
    x_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
    y_train = Y_TRAIN.copy()
    x_test = np.asarray([[100.0], [101.0], [110.0], [111.0]])
    evaluate_fold(x_train, y_train, x_test, Y_TEST, model_name="xgb", seed=5)

    assert len(captured) == 1
    np.testing.assert_allclose(captured[0].named_steps["scaler"].mean_, [6.5])


@pytest.mark.parametrize("model_name", ["mlp", "xgb"])
def test_evaluate_fold_same_seed_is_repeatable(model_name: str) -> None:
    context = pytest.warns(ConvergenceWarning) if model_name == "mlp" else _no_warnings()
    with context:
        first = evaluate_fold(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, model_name, seed=13)
    context = pytest.warns(ConvergenceWarning) if model_name == "mlp" else _no_warnings()
    with context:
        second = evaluate_fold(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, model_name, seed=13)

    for key in ("fit_seconds", "predict_seconds"):
        first.pop(key)
        second.pop(key)
    assert first == second


@pytest.mark.parametrize(
    ("x_train", "y_train", "x_test", "y_test"),
    [
        (np.asarray([0.0, 1.0]), Y_TRAIN[:2], X_TEST, Y_TEST),
        (np.asarray([[0.0], [1.0]]), np.asarray([[0], [1]]), X_TEST, Y_TEST),
        (np.asarray([[0.0], [1.0]]), np.asarray([0, 1]), np.asarray([[0.0]]), Y_TEST),
        (np.asarray([[0.0], [np.nan]]), np.asarray([0, 1]), X_TEST, Y_TEST),
        (np.asarray([[0.0], [np.inf]]), np.asarray([0, 1]), X_TEST, Y_TEST),
        (np.asarray([[0.0], [1.0]]), np.asarray([0, 2]), X_TEST, Y_TEST),
        (np.asarray([[0.0], [1.0]]), np.asarray([0, 1]), X_TEST, np.asarray([0, 0, 0, 0])),
        (np.empty((0, 1)), np.asarray([], dtype=int), X_TEST, Y_TEST),
        (X_TRAIN, np.zeros(8, dtype=int), X_TEST, Y_TEST),
        (X_TRAIN, np.asarray([0, np.nan, 0, 1, 1, 1, 1, 1]), X_TEST, Y_TEST),
        (X_TRAIN, Y_TRAIN, np.empty((0, 1)), np.asarray([], dtype=int)),
        (X_TRAIN, Y_TRAIN, np.asarray([0.1, 0.5, 3.1, 3.5]), Y_TEST),
        (X_TRAIN, Y_TRAIN, np.asarray([[0.1, 0.0], [0.5, 0.0], [3.1, 0.0], [3.5, 0.0]]), Y_TEST),
    ],
)
def test_evaluate_fold_rejects_malformed_or_undefined_inputs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        evaluate_fold(x_train, y_train, x_test, y_test, model_name="mlp", seed=1)


def test_evaluate_fold_rejects_non_numeric_features() -> None:
    with pytest.raises(ValueError):
        evaluate_fold(
            np.asarray([["a"], ["b"]], dtype=object),
            np.asarray([0, 1]),
            X_TEST,
            Y_TEST,
            model_name="mlp",
            seed=1,
        )


@pytest.mark.parametrize(
    ("y_train", "y_test"),
    [
        (np.asarray([0 + 0j, 1 + 0j]), Y_TEST),
        (Y_TRAIN, np.asarray([0 + 0j, 0 + 0j, 1 + 0j, 1 + 0j])),
    ],
)
def test_evaluate_fold_rejects_complex_labels_before_model(
    monkeypatch: pytest.MonkeyPatch,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    import lpcode_v1.experiment as experiment

    monkeypatch.setattr(
        experiment,
        "build_model",
        lambda model_name, seed: pytest.fail("complex labels reached model construction"),
    )
    x_train = X_TRAIN if y_train.shape[0] == X_TRAIN.shape[0] else X_TRAIN[:2]
    with pytest.raises(ValueError, match="real-valued"):
        evaluate_fold(x_train, y_train, X_TEST, y_test, model_name="mlp", seed=1)
