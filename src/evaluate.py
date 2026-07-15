"""Metricas y visualizacion para la evaluacion de modelos.

Quinta y ultima etapa del pipeline. En un problema clinico no basta con la
"exactitud": interesa especialmente NO pasar por alto casos malignos. Por eso
este modulo separa y reporta la sensibilidad y la especificidad, y ofrece
utilidades de visualizacion (matriz de confusion y diagrama de persistencia).

Terminologia sobre la matriz de confusion (clase 1 = maligno):
  - TP (verdadero positivo): maligno detectado como maligno.
  - FN (falso negativo): maligno NO detectado -> el error mas grave en cribado.
  - FP (falso positivo): benigno marcado como maligno -> genera alarmas/biopsias.
  - TN (verdadero negativo): benigno correctamente descartado.
"""
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
    """Calcula las metricas relevantes para diagnostico clinico.

    Parametros
    ----------
    y_true : etiquetas reales (0/1).
    y_pred : etiquetas predichas por el modelo (0/1).
    y_score : probabilidad de la clase maligna (opcional); si se pasa, se
              calcula tambien el AUC-ROC.

    La especificidad no viene en sklearn como funcion directa, asi que la
    derivamos de la matriz de confusion: TN / (TN + FP).
    """
    # ravel() sobre la matriz 2x2 (con labels=[0,1]) desempaqueta en el orden
    # tn, fp, fn, tp -> asi evitamos confusiones de indices.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        # recall == sensibilidad: de todos los malignos, cuantos detectamos.
        "sensibilidad_recall": recall_score(y_true, y_pred, zero_division=0),
        "especificidad": specificity,
        # precision: de los que marcamos como malignos, cuantos lo eran.
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # f1: media armonica de precision y recall (equilibrio entre ambas).
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        # AUC: capacidad de ranking del modelo, independiente del umbral 0.5.
        metrics["auc"] = roc_auc_score(y_true, y_score)
    return metrics


def plot_confusion(y_true, y_pred, labels=("Benigno", "Maligno"), ax=None):
    """Dibuja la matriz de confusion como un mapa de calor anotado."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    # Escribimos el numero de casos en cada celda de la matriz.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    return ax


def plot_persistence_diagram(diagram: np.ndarray, ax=None):
    """Dibuja un diagrama de persistencia (puntos birth vs. death por dimension).

    Interpretacion visual: cada punto es una caracteristica topologica. Cuanto
    MAS LEJOS de la diagonal (death = birth) esta un punto, mas persiste y mas
    relevante es. Los puntos pegados a la diagonal suelen ser ruido.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    diag = np.asarray(diagram)
    # giotto-tda devuelve (1, n, 3); nos quedamos con la primera muestra.
    if diag.ndim == 3:
        diag = diag[0]
    # Dibujamos cada dimension de homologia (H0, H1...) con un color distinto.
    for dim in np.unique(diag[:, 2]):
        pts = diag[diag[:, 2] == dim]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, label=f"H{int(dim)}")
    # Linea diagonal de referencia (birth == death).
    lims = [diag[:, :2].min(), diag[:, :2].max()]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Nacimiento")
    ax.set_ylabel("Muerte")
    ax.legend()
    return ax
