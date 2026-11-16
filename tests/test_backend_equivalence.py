"""Issue #88: equivalencia numerica entre los backends giotto-tda y gudhi.

Estos tests solo corren donde AMBOS backends estan instalados a la vez.
En el entorno de desarrollo (Python 3.14) giotto-tda no tiene wheels, asi
que aqui se saltan; el job 'backend-equivalence' de
.github/workflows/tests.yml los ejecuta en CI con Python 3.12, donde si
hay wheels de ambos.

Que se comprueba: para varias imagenes aleatorias y estructuradas, los
diagramas de persistencia (H0 y H1) de gtda.CubicalPersistence y de
gudhi.CubicalComplex (nuestra implementacion en
src/topology.py:_persistence_diagram_gudhi) contienen exactamente los
mismos puntos (birth, death), una vez descartados los puntos de
persistencia cero que giotto-tda usa como relleno (padding en la
diagonal). La vectorizacion posterior (Persistence Image) difiere por
diseño entre backends y NO se compara aqui; lo que debe coincidir es la
homologia persistente en si.
"""
from __future__ import annotations

import numpy as np
import pytest

gtda_homology = pytest.importorskip(
    "gtda.homology", reason="giotto-tda no instalado (solo hay wheels hasta Python 3.12)"
)
pytest.importorskip("gudhi", reason="gudhi no instalado en este entorno")

import src.topology as topology  # noqa: E402


def _points_by_dim(diagram: np.ndarray, dim: int) -> np.ndarray:
    """Puntos (birth, death) de una dimension, sin los de persistencia cero
    (giotto-tda rellena con puntos en la diagonal para igualar tamanos de
    lote), ordenados para poder comparar como multiconjunto."""
    pts = diagram[0]
    pts = pts[pts[:, 2] == dim][:, :2]
    pts = pts[np.abs(pts[:, 1] - pts[:, 0]) > 1e-12]
    return pts[np.lexsort((pts[:, 1], pts[:, 0]))]


def _images():
    rng = np.random.default_rng(42)
    yield "ruido uniforme", rng.random((32, 32))
    yield "gradiente + ruido", np.linspace(0, 1, 32 * 32).reshape(32, 32) + 0.05 * rng.random((32, 32))
    img = np.full((32, 32), 0.6)
    yy, xx = np.mgrid[0:32, 0:32]
    img -= 0.5 * np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / 40.0)  # pozo central
    img += 0.3 * np.exp(-((xx - 8) ** 2 + (yy - 24) ** 2) / 10.0)   # pico lateral
    yield "estructura con pozo y pico", img


@pytest.mark.parametrize("nombre,imagen", list(_images()))
def test_gtda_and_gudhi_produce_the_same_persistence_points(nombre, imagen):
    cubical = gtda_homology.CubicalPersistence(homology_dimensions=(0, 1), n_jobs=1)
    diag_gtda = cubical.fit_transform(imagen.astype(np.float32)[None, :, :])
    diag_gudhi = topology._persistence_diagram_gudhi(imagen, (0, 1))

    for dim in (0, 1):
        pts_gtda = _points_by_dim(np.asarray(diag_gtda, dtype=np.float64), dim)
        pts_gudhi = _points_by_dim(diag_gudhi, dim)
        assert pts_gtda.shape == pts_gudhi.shape, (
            f"[{nombre}] H{dim}: {pts_gtda.shape[0]} puntos (gtda) vs "
            f"{pts_gudhi.shape[0]} (gudhi)"
        )
        np.testing.assert_allclose(
            pts_gtda, pts_gudhi, atol=1e-5,
            err_msg=f"[{nombre}] H{dim}: puntos distintos entre backends",
        )
