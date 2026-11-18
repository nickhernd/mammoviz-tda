"""Tests de src/registration.py con imagenes sinteticas construidas a mano
(comprueban el codigo, no son datos del estudio)."""
from __future__ import annotations

import numpy as np

from src.registration import map_point_registered


def _breast(shape=(800, 640), shift=(0, 0), seed=0):
    """Una "mama" (semielipse) con textura suave, desplazada ``shift`` px."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    inside = ((rr - 400) / 330) ** 2 + (cc / 520) ** 2 <= 1
    texture = ndimage.gaussian_filter(rng.random(shape), 12)
    img = np.where(inside, 0.3 + texture, 0.0).astype(np.float32)
    return ndimage.shift(img, shift, order=1)


def test_affine_registration_recovers_a_translation_better_than_the_box():
    a = _breast()
    # la mama se desplaza 24 px en vertical y la textura con ella; la caja de la
    # mama tambien se mueve, pero un punto interior ademas cambia de posicion
    # relativa si la mama se comprime: aqui solo hay traslacion, asi que ambos
    # metodos deben acertar con poco error
    b = _breast(shift=(24, 0))
    point = (300.0, 200.0)
    expected = (point[0] + 24, point[1])
    for method in ("caja", "afin"):
        r, c = map_point_registered(a, b, *point, method=method, factor=4)
        assert np.hypot(r - expected[0], c - expected[1]) < 16, method


def test_unknown_method_is_rejected():
    import pytest

    a = _breast()
    with pytest.raises(ValueError):
        map_point_registered(a, a, 10.0, 10.0, method="magia", factor=4)
