"""Tests de src/models.py: clasificadores clasicos, CNN y fusion de features."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models import build_random_forest, build_svm, fuse_features


def _toy_dataset(n_samples: int = 40, n_features: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)  # separable a partir de la 1a caracteristica
    return X, y


# ---------------------------------------------------------------------------
# build_svm
# ---------------------------------------------------------------------------

def test_build_svm_returns_scaler_plus_svc_pipeline():
    pipeline = build_svm()

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["scaler", "svc"]


def test_build_svm_fits_and_predicts_probabilities():
    X, y = _toy_dataset()
    pipeline = build_svm()

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert preds.shape == (len(y),)
    assert probs.shape == (len(y), 2)


# ---------------------------------------------------------------------------
# build_random_forest
# ---------------------------------------------------------------------------

def test_build_random_forest_returns_configured_classifier():
    clf = build_random_forest(n_estimators=50, seed=7)

    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 50
    assert clf.random_state == 7
    assert clf.class_weight == "balanced"


def test_build_random_forest_fits_and_predicts():
    X, y = _toy_dataset()
    clf = build_random_forest(n_estimators=20)

    clf.fit(X, y)
    preds = clf.predict(X)

    assert preds.shape == (len(y),)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------------------------
# build_cnn (requiere TensorFlow, no siempre disponible)
# ---------------------------------------------------------------------------

def test_build_cnn_compiles_model():
    tf = pytest.importorskip("tensorflow")

    from src.models import build_cnn

    model = build_cnn(input_shape=(32, 32, 3), n_classes=2)

    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 2)


# ---------------------------------------------------------------------------
# fuse_features
# ---------------------------------------------------------------------------

def test_fuse_features_concatenates_columns():
    topo = np.zeros((5, 3))
    cnn_embeddings = np.ones((5, 4))

    fused = fuse_features(topo, cnn_embeddings)

    assert fused.shape == (5, 7)
    assert np.array_equal(fused[:, :3], topo)
    assert np.array_equal(fused[:, 3:], cnn_embeddings)
