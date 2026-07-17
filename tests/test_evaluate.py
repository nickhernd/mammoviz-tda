"""Tests de src/evaluate.py: metricas clinicas y utilidades de visualizacion."""
import numpy as np
import pytest

from src.evaluate import clinical_metrics, plot_confusion, plot_persistence_diagram


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
