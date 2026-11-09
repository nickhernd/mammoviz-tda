"""Tests de scripts/synthetic_longitudinal_validation.py: sobre todo, el
recorte por ``max_points`` de _diagram_by_dim/topological_drift (issue #94),
necesario para que la distancia bottleneck sea tratable sobre diagramas
reales (miles de puntos) en vez de solo sobre los sinteticos de este modulo
(decenas de puntos).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

persim = pytest.importorskip("persim", reason="persim no instalado en este entorno")

from synthetic_longitudinal_validation import _diagram_by_dim, topological_drift  # noqa: E402


def _make_diagram(n_points: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    birth = rng.uniform(0, 0.8, size=n_points)
    death = birth + rng.uniform(0.01, 0.2, size=n_points)
    dims = np.full(n_points, dim, dtype=float)
    points = np.stack([birth, death, dims], axis=1)
    return points[None, :, :]  # forma (1, n, 3)


def test_diagram_by_dim_without_max_points_keeps_everything():
    diagram = _make_diagram(50, dim=0)
    points = _diagram_by_dim(diagram, dim=0)
    assert points.shape == (50, 2)


def test_diagram_by_dim_max_points_keeps_the_most_persistent():
    diagram = _make_diagram(50, dim=0)
    points_full = _diagram_by_dim(diagram, dim=0)
    persistence_full = points_full[:, 1] - points_full[:, 0]

    points_pruned = _diagram_by_dim(diagram, dim=0, max_points=10)

    assert points_pruned.shape == (10, 2)
    persistence_pruned = points_pruned[:, 1] - points_pruned[:, 0]
    # los 10 puntos recortados deben ser, exactamente, los 10 de mayor persistencia
    expected = np.sort(persistence_full)[-10:]
    assert np.allclose(np.sort(persistence_pruned), expected)


def test_diagram_by_dim_max_points_is_noop_when_fewer_points_than_limit():
    diagram = _make_diagram(5, dim=1)
    points = _diagram_by_dim(diagram, dim=1, max_points=100)
    assert points.shape == (5, 2)


def test_topological_drift_with_max_points_approximates_unpruned_when_signal_is_in_top_points():
    # Construimos dos diagramas donde los puntos de alta persistencia son los
    # mismos, rodeados de "ruido" de baja persistencia: recortar a
    # max_points es una APROXIMACION (la distancia bottleneck no es, en
    # general, invariante al quitar puntos de baja persistencia: puede
    # cambiar el emparejamiento optimo con la diagonal), pero para una señal
    # dominante de alta persistencia como esta deberia quedar muy cerca del
    # valor sin recortar, no identica.
    def _signal_plus_noise(seed):
        rng_local = np.random.default_rng(seed)
        signal_birth = np.array([0.1, 0.3])
        signal_death = np.array([0.9, 0.95])
        noise_birth = rng_local.uniform(0, 0.5, size=200)
        noise_death = noise_birth + rng_local.uniform(0.001, 0.02, size=200)
        birth = np.concatenate([signal_birth, noise_birth])
        death = np.concatenate([signal_death, noise_death])
        dims = np.zeros_like(birth)
        return np.stack([birth, death, dims], axis=1)[None, :, :]

    diagram_a = _signal_plus_noise(seed=10)
    diagram_b = _signal_plus_noise(seed=20)

    full = topological_drift(diagram_a, diagram_b)
    pruned = topological_drift(diagram_a, diagram_b, max_points=20)

    assert pruned[0] == pytest.approx(full[0], abs=0.02)
