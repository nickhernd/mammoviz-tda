"""Preprocesado de mamografias para el pipeline TDA + deep learning."""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - opcional en tiempo de import
    cv2 = None


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen a escala de grises float en [0, 1]."""
    if image.ndim == 3:
        if cv2 is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = image.mean(axis=2)
    image = image.astype(np.float32)
    vmin, vmax = float(image.min()), float(image.max())
    if vmax > vmin:
        image = (image - vmin) / (vmax - vmin)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Realce de contraste adaptativo (CLAHE). Requiere OpenCV.

    Recibe y devuelve una imagen en escala de grises float en [0, 1].
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) es necesario para CLAHE.")
    img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img8).astype(np.float32) / 255.0


def crop_roi(
    image: np.ndarray, mask: np.ndarray, margin: int = 16
) -> np.ndarray:
    """Recorta la region de interes definida por una mascara binaria."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return image
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, image.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, image.shape[1])
    return image[y0:y1, x0:x1]


def resize(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Reescala la imagen al tamano indicado."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) es necesario para el reescalado.")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    size: tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> np.ndarray:
    """Pipeline de preprocesado completo: gris -> ROI -> CLAHE -> resize."""
    img = to_grayscale_float(image)
    if mask is not None:
        img = crop_roi(img, mask)
    if use_clahe:
        img = apply_clahe(img)
    img = resize(img, size)
    return img
