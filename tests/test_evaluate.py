"""Tests de src/evaluate.py: metricas clinicas y utilidades de visualizacion."""
import numpy as np
import pytest

from src.evaluate import bootstrap_ci, clinical_metrics, paired_test, plot_confusion, plot_persistence_diagram


# ---------------------------------------------------------------------------
# clinical_metrics
# ---------------------------------------------------------------------------

def test_clinical_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    metrics = clinical_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["sensibilidad_recall"] == 1.0
    assert metrics["especificidad"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0


def test_clinical_metrics_matches_hand_computed_confusion_matrix():
    # Matriz de confusion (labels=[0,1]): TN=3, FP=1, FN=1, TP=2 (8 muestras).
    y_true = [0, 0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 1, 1]

    metrics = clinical_metrics(y_true, y_pred)

    # sensibilidad = TP/(TP+FN) = 2/3 ; especificidad = TN/(TN+FP) = 3/4
    assert metrics["sensibilidad_recall"] == pytest.approx(2 / 3)
    assert metrics["especificidad"] == pytest.approx(3 / 4)
    # precision = TP/(TP+FP) = 2/3
    assert metrics["precision"] == pytest.approx(2 / 3)


def test_clinical_metrics_includes_auc_when_score_given():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    metrics = clinical_metrics(y_true, y_pred, y_score=y_score)

    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_clinical_metrics_omits_auc_without_score():
    metrics = clinical_metrics([0, 1], [0, 1])

    assert "auc" not in metrics


def test_clinical_metrics_specificity_zero_when_no_negatives_predicted_correctly():
    # Todas las muestras negativas se predicen como positivas -> TN=0, FP=2.
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 1, 1]

    metrics = clinical_metrics(y_true, y_pred)

    assert metrics["especificidad"] == 0.0


# ---------------------------------------------------------------------------
# Visualizacion (matplotlib, sin backend interactivo)
# ---------------------------------------------------------------------------

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_plot_confusion_returns_axes_with_annotations():
    ax = plot_confusion([0, 1, 0, 1], [0, 1, 1, 1])

    assert ax is not None
    assert len(ax.texts) == 4  # una anotacion por celda de la matriz 2x2


def test_plot_persistence_diagram_handles_3d_giotto_shape():
    # giotto-tda devuelve (1, n_puntos, 3): (birth, death, dimension).
    diagram = np.array([[[0.0, 0.5, 0], [0.1, 0.9, 1]]])

    ax = plot_persistence_diagram(diagram)

    assert ax is not None
    assert len(ax.collections) == 2  # un scatter por dimension de homologia (H0, H1)


# ---------------------------------------------------------------------------
# bootstrap_ci / paired_test (issue #79)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_the_true_mean_on_repeated_sampling():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=200)
    mean, lower, upper = bootstrap_ci(values, n_resamples=500, seed=1)
    assert lower <= mean <= upper
    assert mean == pytest.approx(values.mean())


def test_bootstrap_ci_empty_array_returns_nan():
    mean, lower, upper = bootstrap_ci(np.array([]))
    assert np.isnan(mean) and np.isnan(lower) and np.isnan(upper)


def test_bootstrap_ci_is_narrower_with_more_samples():
    rng = np.random.default_rng(0)
    small = rng.normal(loc=0.5, scale=0.1, size=8)
    large = rng.normal(loc=0.5, scale=0.1, size=800)
    _, lo_small, hi_small = bootstrap_ci(small, n_resamples=1000, seed=1)
    _, lo_large, hi_large = bootstrap_ci(large, n_resamples=1000, seed=1)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_paired_test_identical_arrays_gives_pvalue_one():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    result = paired_test(values, values)
    assert result["p_value"] == 1.0
    assert result["median_diff"] == 0.0


def test_paired_test_detects_a_clear_systematic_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.8, scale=0.02, size=30)
    b = rng.normal(loc=0.2, scale=0.02, size=30)
    result = paired_test(a, b)
    assert result["p_value"] < 0.01
    assert result["median_diff"] > 0
