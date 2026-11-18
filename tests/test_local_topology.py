"""Tests de src/local_topology.py con arrays construidos a mano (comprueban
el codigo, no son datos del estudio)."""
from __future__ import annotations

import numpy as np

from src.local_topology import (
    breast_bbox, cell_neighbours, grid_patches, map_point, point_to_cell, window_around,
)


def test_breast_bbox_finds_the_tissue_and_ignores_small_specks():
    img = np.zeros((100, 80))
    img[20:70, 10:50] = 1.0     # la "mama"
    img[90:92, 75:77] = 1.0     # una mota aislada (etiqueta, artefacto)
    assert breast_bbox(img) == (20, 10, 70, 50)


def test_breast_bbox_of_a_flat_image_is_the_whole_image():
    assert breast_bbox(np.zeros((10, 12))) == (0, 0, 10, 12)


def test_grid_patches_cover_the_image_exactly_once():
    img = np.arange(100 * 90).reshape(100, 90)
    patches = grid_patches(img, grid=4)
    assert len(patches) == 16
    assert sum(p.size for p in patches) == img.size
    # orden por filas: el parche 1 empieza en la fila 0, el 4 en la segunda fila de celdas
    assert patches[0][0, 0] == img[0, 0]
    assert patches[4][0, 0] == img[25, 0]


def test_point_to_cell_uses_breast_relative_coordinates():
    bbox = (100, 200, 500, 600)  # mama de 400x400 empezando en (100, 200)
    assert point_to_cell(100, 200, bbox) == 0           # esquina superior izquierda
    assert point_to_cell(499, 599, bbox) == 15          # esquina inferior derecha
    assert point_to_cell(300, 400, bbox) == 2 * 4 + 2   # centro -> celda (2, 2)
    assert point_to_cell(0, 0, bbox) == 0               # fuera de la caja -> borde


def test_cell_neighbours_in_corner_and_centre():
    assert cell_neighbours(0) == {0, 1, 4, 5}
    assert len(cell_neighbours(5)) == 9


def test_map_point_keeps_the_relative_position_in_the_breast():
    # centro de una mama -> centro de la otra, aunque cambien encuadre y tamaño
    assert map_point(300, 400, (100, 200, 500, 600), (0, 0, 200, 100)) == (100.0, 50.0)
    assert map_point(100, 200, (100, 200, 500, 600), (10, 20, 30, 40)) == (10.0, 20.0)


def test_window_around_is_centred_and_shifted_inside_at_the_border():
    img = np.arange(100 * 100).reshape(100, 100)
    w = window_around(img, 50, 50, half=10)
    assert w.shape == (20, 20) and w[0, 0] == img[40, 40]
    corner = window_around(img, 0, 99, half=10)   # se sale por arriba y por la derecha
    assert corner.shape == (20, 20) and corner[0, 0] == img[0, 80]


def test_shift_point_moves_half_the_breast_and_wraps_around():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from embed_lesion_site import shift_point

    bbox = (100, 0, 500, 300)                      # mama de 400 px de alto
    assert shift_point(150, 40, bbox, 0.5) == (350.0, 40)   # mitad superior -> inferior, misma columna
    assert shift_point(450, 40, bbox, 0.5) == (250.0, 40)   # da la vuelta dentro de la caja


def test_chest_wall_side_and_mirroring():
    from src.local_topology import chest_wall_side, mirror_relative

    img = np.zeros((100, 80))
    img[10:90, 0:50] = 1.0                    # mama pegada al borde izquierdo
    assert chest_wall_side(img) == "left"
    assert chest_wall_side(img[:, ::-1]) == "right"
    # un punto a 30 % de profundidad desde la pared, en una mama con la pared a
    # la izquierda, corresponde en la otra (pared a la derecha) a la columna 70 %
    assert mirror_relative((0.4, 0.3), "left", "right") == (0.4, 0.7)
    assert mirror_relative((0.4, 0.3), "left", "left") == (0.4, 0.3)
