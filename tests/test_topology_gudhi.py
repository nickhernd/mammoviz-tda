"""Tests especificos del backend gudhi de src/topology.py.

Todo el fichero depende de gudhi, asi que el importorskip a nivel de modulo es
correcto aqui (salta el fichero entero si no esta instalado, que es justo lo
que queremos). No mezclar con tests que no necesiten gudhi en el mismo
fichero: un importorskip a mitad de modulo salta TODO el modulo, no solo lo
que viene despues.
"""
import numpy as np
import pytest

gudhi = pytest.importorskip("gudhi", reason="gudhi no instalado en este entorno")

import src.topology as topology


def test_persistence_diagram_gudhi_caps_infinite_death_at_image_max():
    # En H0 siempre hay una barra infinita (la componente que cubre toda la
    # imagen): debe quedar acotada al maximo de intensidad, nunca inf.
    image = np.random.rand(16, 16).astype(np.float32)

    diagrams = topology._persistence_diagram_gudhi(image, homology_dims=(0, 1))

    assert np.isfinite(diagrams).all()
    assert diagrams[0][:, 1].max() <= float(image.max()) + 1e-9


def test_persistence_image_manual_is_deterministic_and_normalized_grid():
    diagrams = np.array([[[0.2, 0.6, 0.0], [0.1, 0.3, 1.0]]])

    image_a = topology._persistence_image_manual(diagrams, n_bins=16)
    image_b = topology._persistence_image_manual(diagrams, n_bins=16)

    assert image_a.shape == (1, 16 * 16 * 2)  # 2 dimensiones presentes (0 y 1)
    assert np.array_equal(image_a, image_b)
    assert np.all(image_a >= 0.0)


def test_persistence_image_manual_handles_empty_diagram():
    diagrams = np.empty((1, 0, 3))

    image = topology._persistence_image_manual(diagrams, n_bins=8)

    assert np.all(image == 0.0)


def test_persistence_image_manual_keeps_fixed_length_when_a_dimension_is_absent():
    # Regresion: si una imagen concreta no tiene NINGUN punto H1 (plausible en
    # fondos lisos, p.ej. ruido sin estructura), el vector debe seguir
    # teniendo longitud 2*n_bins*n_bins (H0 + H1), con el bloque H1 a cero -
    # no un vector mas corto que el de una imagen con ambas dimensiones, o
    # distintas imagenes del mismo lote quedarian con features de longitud
    # incompatible al apilarlas en una matriz.
    only_h0 = np.array([[[0.2, 0.6, 0.0], [0.1, 0.3, 0.0]]])
    both_dims = np.array([[[0.2, 0.6, 0.0], [0.1, 0.3, 1.0]]])

    image_only_h0 = topology._persistence_image_manual(only_h0, n_bins=8)
    image_both = topology._persistence_image_manual(both_dims, n_bins=8)

    assert image_only_h0.shape == image_both.shape == (1, 8 * 8 * 2)
    assert np.all(image_only_h0[:, 8 * 8:] == 0.0)  # bloque H1 vacio -> todo cero


def test_persistence_statistics_manual_counts_points_per_dimension():
    diagrams = np.array([[[0.1, 0.4, 0.0], [0.2, 0.5, 0.0], [0.0, 0.9, 1.0]]])

    stats = topology._persistence_statistics_manual(diagrams)

    # 2 columnas por dimension (persistencia total, num. puntos); dims 0 y 1.
    assert stats.shape == (1, 4)
    assert stats[0, 1] == 2  # 2 puntos en dimension 0
    assert stats[0, 3] == 1  # 1 punto en dimension 1
