"""Tests de src/local_topology.py con arrays construidos a mano (comprueban
el codigo, no son datos del estudio)."""
from __future__ import annotations

import numpy as np

from src.local_topology import breast_bbox, cell_neighbours, grid_patches, point_to_cell


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
