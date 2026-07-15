"""Preprocesado de mamografias para el pipeline TDA + deep learning.

Segunda etapa del pipeline. Transforma una imagen cruda (posiblemente en color,
con rangos de intensidad arbitrarios y con toda la mama) en una imagen limpia,
normalizada y centrada en la lesion, lista para:
  - calcular su homologia persistente (modulo topology.py), y/o
  - alimentar la CNN de referencia (modulo models.py).

Cada funcion hace UNA cosa, de modo que se pueden combinar o probar por separado.
OpenCV (cv2) se importa de forma tolerante: si no esta instalado, el modulo se
puede importar igualmente y solo fallan las funciones que lo necesitan.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 es opcional en tiempo de import
    cv2 = None


def to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen a escala de grises en formato float [0, 1].

    Las mamografias son en escala de grises, pero pueden llegar como RGB o con
    rangos de intensidad muy distintos (0-255, 0-4095 en DICOM...). Aqui:
      1) si viene en color, se pasa a un solo canal;
      2) se normaliza al rango [0, 1] mediante min-max, para que el resto del
         pipeline trabaje siempre con la misma escala.
    """
    # Paso 1: reducir a un canal si la imagen tiene 3 dimensiones (H, W, C).
    if image.ndim == 3:
        if cv2 is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # Fallback sin OpenCV: media de los canales de color.
            image = image.mean(axis=2)

    # Paso 2: normalizacion min-max a [0, 1].
    image = image.astype(np.float32)
    vmin, vmax = float(image.min()), float(image.max())
    if vmax > vmin:  # evita division por cero en imagenes constantes
        image = (image - vmin) / (vmax - vmin)
    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Realce de contraste adaptativo (CLAHE). Requiere OpenCV.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) mejora el contraste
    LOCAL, lo que ayuda a resaltar microcalcificaciones y bordes de masas sin
    saturar el resto de la imagen. Es un realce muy usado en mamografia.

    Recibe y devuelve una imagen en escala de grises float en [0, 1].
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) es necesario para CLAHE.")
    # CLAHE de OpenCV trabaja sobre enteros de 8 bits, asi que convertimos
    # [0,1] -> [0,255] uint8, aplicamos, y volvemos a [0,1] float.
    img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img8).astype(np.float32) / 255.0


def crop_roi(
    image: np.ndarray, mask: np.ndarray, margin: int = 16
) -> np.ndarray:
    """Recorta la region de interes (ROI) definida por una mascara binaria.

    CBIS-DDSM proporciona mascaras que marcan donde esta la lesion. En lugar de
    analizar toda la mama, recortamos un rectangulo ajustado a la lesion (mas un
    pequeno margen), lo que reduce ruido y coste computacional.
    """
    # np.where devuelve las coordenadas (filas ys, columnas xs) de los pixeles
    # de la mascara que estan "encendidos" (> 0).
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:  # mascara vacia: devolvemos la imagen sin recortar
        return image
    # Rectangulo que engloba la lesion, ampliado con 'margin' y recortado a los
    # limites de la imagen para no salirnos de rango.
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, image.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, image.shape[1])
    return image[y0:y1, x0:x1]


def resize(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Reescala la imagen a un tamano comun (por defecto 224x224).

    Un tamano fijo es necesario para: (a) que la CNN reciba entradas uniformes,
    y (b) que los descriptores topologicos sean comparables entre imagenes.
    INTER_AREA es la interpolacion recomendada al reducir tamano.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) es necesario para el reescalado.")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    size: tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> np.ndarray:
    """Pipeline de preprocesado completo, encadenando las funciones anteriores.

    Orden: escala de grises -> recorte de ROI (si hay mascara) -> CLAHE -> resize.
    Devuelve una imagen 2D float [0, 1] del tamano solicitado.
    """
    img = to_grayscale_float(image)
    if mask is not None:
        img = crop_roi(img, mask)
    if use_clahe:
        img = apply_clahe(img)
    img = resize(img, size)
    return img
