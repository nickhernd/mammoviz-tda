"""Topologia LOCAL de una mamografia: recorte de la mama y rejilla de parches.

La señal de "anticipacion" global (imagen de persistencia de la mamografia
entera) no detecta un precursor en la exploracion previa. Si el precursor
existe, es pequeño frente a la imagen completa y se diluye. Este modulo da
las piezas para mirar LOCALMENTE: recortar la mama (quitando el fondo negro,
que varia entre equipos), dividirla en una rejilla de parches y calcular la
homologia de cada parche por separado.

La rejilla se define sobre el rectangulo envolvente de la mama, no sobre la
imagen: asi la celda (i, j) cubre aproximadamente la misma region
anatomica en dos exploraciones de la misma paciente aunque el tamaño de
imagen o el encuadre cambien (no es un registro elastico, es una
aproximacion gruesa, de ahi que la rejilla sea de pocos parches).
"""
from __future__ import annotations

import numpy as np


def breast_bbox(image: np.ndarray, rel_threshold: float = 0.05) -> tuple[int, int, int, int]:
    """Rectangulo envolvente (r0, c0, r1, c1) de la mama: la mayor region
    conexa por encima de un umbral bajo (el fondo de una mamografia es casi
    negro). ``r1``/``c1`` son exclusivos, estilo slicing de numpy."""
    from scipy import ndimage

    img = np.asarray(image, dtype=np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return 0, 0, img.shape[0], img.shape[1]
    mask = img > lo + rel_threshold * (hi - lo)
    labels, n = ndimage.label(mask)
    if n == 0:
        return 0, 0, img.shape[0], img.shape[1]
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    rows = np.where(largest.any(axis=1))[0]
    cols = np.where(largest.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1


def grid_patches(image: np.ndarray, grid: int = 4) -> list[np.ndarray]:
    """Divide una imagen en ``grid`` x ``grid`` parches, en orden de filas
    (el parche k esta en la fila k // grid, columna k % grid)."""
    rows = np.array_split(np.arange(image.shape[0]), grid)
    cols = np.array_split(np.arange(image.shape[1]), grid)
    return [image[np.ix_(r, c)] for r in rows for c in cols]


def point_to_cell(row: float, col: float, bbox: tuple[int, int, int, int], grid: int = 4) -> int:
    """Celda de la rejilla (indice 0..grid²-1) que contiene el punto (row,
    col), en coordenadas de la imagen original, dada la caja de la mama
    sobre la que se construyo la rejilla. Puntos fuera de la caja se
    asignan a la celda del borde mas cercana."""
    r0, c0, r1, c1 = bbox
    rel_r = (row - r0) / max(r1 - r0, 1)
    rel_c = (col - c0) / max(c1 - c0, 1)
    i = min(max(int(rel_r * grid), 0), grid - 1)
    j = min(max(int(rel_c * grid), 0), grid - 1)
    return i * grid + j


def cell_neighbours(cell: int, grid: int = 4) -> set[int]:
    """La celda y sus vecinas (vecindad 3x3) en la rejilla."""
    i, j = divmod(cell, grid)
    return {
        a * grid + b
        for a in range(max(i - 1, 0), min(i + 2, grid))
        for b in range(max(j - 1, 0), min(j + 2, grid))
    }
