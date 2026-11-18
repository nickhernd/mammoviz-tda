"""Registro entre dos mamografias de la misma mama (t -> t-1).

El analisis en el sitio de la lesion necesita llevar un punto de una
exploracion a otra. La aproximacion inicial (misma posicion relativa en la
caja de la mama, ``src.local_topology.map_point``) es gruesa: en CSAW-CC,
con el sitio trasladado asi, la señal de anticipacion desaparece.

Tres metodos, de menos a mas flexible, todos para trasladar UN punto:

- ``"caja"``: posicion relativa en la caja de la mama (la de partida).
- ``"afin"``: transformacion afin que maximiza la correlacion (ECC de
  OpenCV) entre las dos mamografias reducidas, inicializada con la caja.
- ``"elastico"``: la afin seguida de un campo de deformacion denso (flujo
  optico TV-L1 de scikit-image) para corregir la compresion no rigida.

Todo se calcula a baja resolucion (``DOWNSAMPLE``) y el punto se devuelve en
coordenadas de la imagen original.
"""
from __future__ import annotations

import numpy as np

DOWNSAMPLE = 8


def _prep(image: np.ndarray, factor: int = DOWNSAMPLE) -> tuple[np.ndarray, np.ndarray]:
    """Imagen reducida y normalizada a [0, 1], y mascara de la mama."""
    from src.local_topology import breast_bbox

    small = np.asarray(image, dtype=np.float32)[::factor, ::factor]
    lo, hi = np.percentile(small, [1, 99.5])
    small = np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1)
    r0, c0, r1, c1 = breast_bbox(small)
    mask = np.zeros_like(small, dtype=np.uint8)
    mask[r0:r1, c0:c1] = (small[r0:r1, c0:c1] > 0.05).astype(np.uint8)
    return small, mask


def _bbox_affine(bb_from, bb_to) -> np.ndarray:
    """Matriz afin 2x3 (en x=col, y=fila) que lleva la caja ``bb_from`` a ``bb_to``."""
    r0, c0, r1, c1 = bb_from
    s0, d0, s1, d1 = bb_to
    sx = (d1 - d0) / max(c1 - c0, 1)
    sy = (s1 - s0) / max(r1 - r0, 1)
    return np.array([[sx, 0, d0 - sx * c0], [0, sy, s0 - sy * r0]], dtype=np.float32)


def affine_ecc(img_from: np.ndarray, img_to: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    """Afin 2x3 (coordenadas reducidas, x=col, y=fila) de ``img_from`` a ``img_to``.

    Si ECC no converge, devuelve la inicializacion por cajas."""
    import cv2

    from src.local_topology import breast_bbox

    a, ma = _prep(img_from, factor)
    b, mb = _prep(img_to, factor)
    init = _bbox_affine(breast_bbox(a), breast_bbox(b))
    # ECC estima W tal que b(x) ~ a(W x): la inversa de from->to
    warp = cv2.invertAffineTransform(init).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        _, warp = cv2.findTransformECC(b, a, warp, cv2.MOTION_AFFINE, criteria, mb, 5)
    except cv2.error:
        return init
    return cv2.invertAffineTransform(warp)


def map_point_registered(img_from: np.ndarray, img_to: np.ndarray, row: float, col: float,
                         method: str = "afin", factor: int = DOWNSAMPLE) -> tuple[float, float]:
    """Traslada el punto (row, col) de ``img_from`` a ``img_to`` (resolucion completa)."""
    from src.local_topology import breast_bbox, map_point

    if method == "caja":
        f, t = (breast_bbox(np.asarray(x)[::4, ::4]) for x in (img_from, img_to))
        f = tuple(v * 4 for v in f)
        t = tuple(v * 4 for v in t)
        return map_point(row, col, f, t)

    A = affine_ecc(img_from, img_to, factor)
    x, y = col / factor, row / factor
    xt, yt = A @ np.array([x, y, 1.0])
    if method == "afin":
        return yt * factor, xt * factor
    if method != "elastico":
        raise ValueError(f"metodo desconocido: {method}")

    import cv2
    from skimage.registration import optical_flow_tvl1

    a, _ = _prep(img_from, factor)
    b, _ = _prep(img_to, factor)
    a_warped = cv2.warpAffine(a, A, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR)
    # flujo v tal que b(p) ~ a_warped(p + v(p)); buscamos p con p + v(p) = punto afin
    v_r, v_c = optical_flow_tvl1(b, a_warped)
    rr, cc = np.mgrid[0:b.shape[0], 0:b.shape[1]]
    dist = (rr + v_r - yt) ** 2 + (cc + v_c - xt) ** 2
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    return float(i) * factor, float(j) * factor
