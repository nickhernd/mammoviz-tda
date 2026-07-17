"""Tests de src/topology.py: homologia persistente y vectorizacion.

Estos tests son AGNOSTICOS de backend: corren contra lo que
``topology.backend()`` reporte como activo en este entorno (giotto-tda o
gudhi), o se saltan si no hay ninguno instalado. Los tests especificos de cada
backend viven en ficheros separados (test_topology_gudhi.py,
test_topology_giotto.py): un ``pytest.importorskip`` a mitad de fichero salta
el MODULO ENTERO, no solo lo que viene despues, asi que no puede mezclarse con
tests genericos en el mismo fichero.
"""
import numpy as np
import pytest

import src.topology as topology

# Se aplica solo a los tests que necesitan un backend REAL instalado; los
# tests de "sin backend" de mas abajo deben correr siempre (monkeypatch).
requires_backend = pytest.mark.skipif(
    topology.backend() == "none",
    reason="no hay ningun backend de homologia persistente instalado (giotto-tda o gudhi)",
)


@requires_backend
def test_backend_reports_a_known_value():
    assert topology.backend() in {"giotto-tda", "gudhi"}


@requires_backend
def test_persistence_diagram_has_expected_shape():
    image = np.random.rand(16, 16).astype(np.float32)

    diagrams = topology.persistence_diagram(image)

    assert diagrams.ndim == 3
    assert diagrams.shape[0] == 1
    assert diagrams.shape[2] == 3  # (birth, death, dimension)
    # homology_dims por defecto es (0, 1): no debe aparecer ninguna otra dimension.
    assert set(diagrams[0][:, 2].astype(int).tolist()) <= {0, 1}


@requires_backend
def test_extract_features_image_method_returns_fixed_size_vector():
    image = np.random.rand(16, 16).astype(np.float32)

    vector = topology.extract_features(image, method="image", n_bins=10)

    assert vector.shape == (10 * 10 * 2,)  # n_bins^2 por cada dimension de homologia (H0, H1)


@requires_backend
def test_extract_features_stats_method_returns_vector():
    image = np.random.rand(16, 16).astype(np.float32)

    vector = topology.extract_features(image, method="stats")

    assert vector.ndim == 1


@requires_backend
def test_extract_features_vector_length_is_independent_of_image_size():
    # Distintas imagenes deben producir vectores COMPARABLES (misma longitud),
    # aunque tengan tamanos de entrada distintos.
    small = np.random.rand(16, 16).astype(np.float32)
    large = np.random.rand(48, 48).astype(np.float32)

    v_small = topology.extract_features(small, method="image", n_bins=12)
    v_large = topology.extract_features(large, method="image", n_bins=12)

    assert v_small.shape == v_large.shape


def test_persistence_diagram_raises_clear_error_without_any_backend(monkeypatch):
    monkeypatch.setattr(topology, "_HAS_GIOTTO", False)
    monkeypatch.setattr(topology, "_HAS_GUDHI", False)
    image = np.random.rand(16, 16)

    with pytest.raises(RuntimeError, match="backend"):
        topology.persistence_diagram(image)


def test_extract_features_unknown_method_raises_value_error(monkeypatch):
    # Sustituimos persistence_diagram para que el test no dependa de que haya
    # un backend instalado: el objetivo es solo el `if/elif` de seleccion de metodo.
    monkeypatch.setattr(topology, "persistence_diagram", lambda image, homology_dims=(0, 1): np.zeros((1, 1, 3)))
    monkeypatch.setattr(topology, "persistence_image_vector", lambda diagrams, n_bins=20: np.zeros((1, n_bins * n_bins)))
    monkeypatch.setattr(topology, "persistence_statistics", lambda diagrams: np.zeros((1, 2)))

    with pytest.raises(ValueError):
        topology.extract_features(np.zeros((8, 8)), method="no-existe")
