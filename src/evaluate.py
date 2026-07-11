"""Metricas y visualizacion para la evaluacion de modelos."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clinical_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Calcula metricas relevantes para diagnostico clinico.

    Incluye sensibilidad (recall) y especificidad, criticas en cribado.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensibilidad_recall": recall_score(y_true, y_pred, zero_division=0),
        "especificidad": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    return metrics


def plot_confusion(y_true, y_pred, labels=("Benigno", "Maligno"), ax=None):
    """Dibuja la matriz de confusion."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    return ax


def plot_persistence_diagram(diagram: np.ndarray, ax=None):
    """Dibuja un diagrama de persistencia (birth, death, dim)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    diag = np.asarray(diagram)
    if diag.ndim == 3:
        diag = diag[0]
    for dim in np.unique(diag[:, 2]):
        pts = diag[diag[:, 2] == dim]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, label=f"H{int(dim)}")
    lims = [diag[:, :2].min(), diag[:, :2].max()]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Nacimiento")
    ax.set_ylabel("Muerte")
    ax.legend()
    return ax
