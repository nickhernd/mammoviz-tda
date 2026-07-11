"""Extraccion de descriptores topologicos mediante homologia persistente.

Se emplea homologia persistente cubica (filtracion por sublevel sets de la
intensidad) sobre imagenes en escala de grises, y varias vectorizaciones de
los diagramas de persistencia para su uso como caracteristicas.

Dependencias: giotto-tda (``pip install giotto-tda``).
"""
from __future__ import annotations

import numpy as np

try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import (
        PersistenceImage,
        Scaler,
        Amplitude,
        NumberOfPoints,
    )
except ImportError:  # pragma: no cover
    CubicalPersistence = None


def persistence_diagram(image: np.ndarray, homology_dims=(0, 1)) -> np.ndarray:
    """Calcula el diagrama de persistencia cubico de una imagen 2D.

    Devuelve un array con forma (1, n_puntos, 3): (birth, death, dimension).
    """
    if CubicalPersistence is None:
        raise RuntimeError("giotto-tda es necesario. Instala con: pip install giotto-tda")
    x = np.asarray(image, dtype=np.float32)[None, :, :]
    cubical = CubicalPersistence(homology_dimensions=homology_dims, n_jobs=-1)
    return cubical.fit_transform(x)


def persistence_image_vector(
    diagrams: np.ndarray, n_bins: int = 20
) -> np.ndarray:
    """Vectoriza los diagramas como imagenes de persistencia aplanadas."""
    diagrams = Scaler().fit_transform(diagrams)
    pi = PersistenceImage(n_bins=n_bins, n_jobs=-1)
    images = pi.fit_transform(diagrams)
    return images.reshape(images.shape[0], -1)


def persistence_statistics(diagrams: np.ndarray) -> np.ndarray:
    """Estadisticos simples de persistencia (amplitud y numero de puntos)."""
    amp = Amplitude(metric="wasserstein").fit_transform(diagrams)
    npts = NumberOfPoints().fit_transform(diagrams)
    return np.hstack([amp, npts])


def extract_features(
    image: np.ndarray, method: str = "image", n_bins: int = 20
) -> np.ndarray:
    """Extrae un vector de caracteristicas topologicas de una imagen.

    method:
        'image' -> imagen de persistencia (por defecto)
        'stats' -> estadisticos de persistencia
    """
    diagrams = persistence_diagram(image)
    if method == "image":
        return persistence_image_vector(diagrams, n_bins=n_bins)[0]
    if method == "stats":
        return persistence_statistics(diagrams)[0]
    raise ValueError(f"Metodo desconocido: {method}")
