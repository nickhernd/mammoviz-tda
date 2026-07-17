"""Tests especificos del backend giotto-tda de src/topology.py.

Se saltan automaticamente si giotto-tda no esta instalado (por ejemplo con
Python 3.13+, donde giotto-tda no publica ruedas). Vease tambien
test_topology_gudhi.py y el test_topology.py generico.
"""
import numpy as np
import pytest

gtda = pytest.importorskip("gtda", reason="giotto-tda no instalado en este entorno")

import src.topology as topology


def test_persistence_diagram_shape_with_giotto_tda():
    image = np.random.rand(16, 16).astype(np.float32)

    diagrams = topology.persistence_diagram(image)

    assert diagrams.ndim == 3
    assert diagrams.shape[0] == 1
    assert diagrams.shape[2] == 3
