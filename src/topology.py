"""Extraccion de descriptores topologicos mediante homologia persistente.

CORAZON del TFG. Aqui es donde entra el Analisis Topologico de Datos (TDA).
La idea: en lugar de mirar los pixeles uno a uno, medimos la "forma" de la
imagen (cuantas componentes conexas, cuantos agujeros, y a que escalas
aparecen y desaparecen). Esa informacion es robusta al ruido y complementaria
a la que aprende una CNN.

Como funciona la homologia persistente CUBICA sobre una imagen en escala de grises:
  - Interpretamos la imagen como un "paisaje" donde la intensidad es la altura.
  - Vamos "inundando" desde los valores mas bajos a los mas altos (filtracion
    por sublevel sets). Segun sube el nivel, aparecen y se fusionan regiones.
  - Registramos el NACIMIENTO (birth) y la MUERTE (death) de cada caracteristica
    topologica -> eso es el "diagrama de persistencia".
  - Las caracteristicas que viven mucho (death - birth grande) son estructura
    real; las de vida corta suelen ser ruido.

Dependencias: giotto-tda (``pip install giotto-tda``).
"""
from __future__ import annotations

import numpy as np

# Importacion tolerante: el modulo se puede importar sin giotto-tda instalado;
# solo fallaran las funciones que realmente lo usan (con un mensaje claro).
try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import (
        PersistenceImage,
        Scaler,
        Amplitude,
        NumberOfPoints,
    )
except ImportError:  # pragma: no cover
    CubicalPersistence = None


def persistence_diagram(image: np.ndarray, homology_dims=(0, 1)) -> np.ndarray:
    """Calcula el diagrama de persistencia cubico de una imagen 2D.

    homology_dims=(0, 1) pide dos tipos de caracteristicas:
      - dimension 0 (H0): componentes conexas (islas de intensidad).
      - dimension 1 (H1): agujeros / ciclos (util para masas espiculadas,
        distribucion de microcalcificaciones...).

    Devuelve un array con forma (1, n_puntos, 3), donde cada punto es
    (birth, death, dimension).
    """
    if CubicalPersistence is None:
        raise RuntimeError("giotto-tda es necesario. Instala con: pip install giotto-tda")
    # giotto-tda espera un lote de imagenes; anadimos un eje de lote -> (1, H, W).
    x = np.asarray(image, dtype=np.float32)[None, :, :]
    # CubicalPersistence es el estimador que calcula la homologia sobre la rejilla
    # de pixeles (complejo cubico). n_jobs=-1 usa todos los nucleos disponibles.
    cubical = CubicalPersistence(homology_dimensions=homology_dims, n_jobs=-1)
    return cubical.fit_transform(x)


def persistence_image_vector(
    diagrams: np.ndarray, n_bins: int = 20
) -> np.ndarray:
    """Vectoriza los diagramas como 'imagenes de persistencia' aplanadas.

    Un diagrama de persistencia es un conjunto de puntos de tamano variable, y
    los algoritmos de ML necesitan vectores de tamano FIJO. La 'persistence
    image' resuelve esto: convierte el diagrama en una rejilla n_bins x n_bins
    (una imagen), que luego aplanamos a un vector.

    - Scaler: reescala los diagramas a un rango comun antes de vectorizar.
    - reshape(len, -1): aplana cada imagen 2D a un vector 1D por muestra.
    """
    diagrams = Scaler().fit_transform(diagrams)
    pi = PersistenceImage(n_bins=n_bins, n_jobs=-1)
    images = pi.fit_transform(diagrams)
    return images.reshape(images.shape[0], -1)


def persistence_statistics(diagrams: np.ndarray) -> np.ndarray:
    """Estadisticos compactos de persistencia (alternativa ligera al vector imagen).

    En vez de una imagen completa, resume cada diagrama con pocos numeros:
      - Amplitude (metrica Wasserstein): "cuanta" persistencia hay en total.
      - NumberOfPoints: cuantas caracteristicas topologicas se han detectado.
    Utiles como baseline rapido o para combinar con otras caracteristicas.
    """
    amp = Amplitude(metric="wasserstein").fit_transform(diagrams)
    npts = NumberOfPoints().fit_transform(diagrams)
    # hstack pega ambos bloques de columnas -> un vector de caracteristicas por muestra.
    return np.hstack([amp, npts])


def extract_features(
    image: np.ndarray, method: str = "image", n_bins: int = 20
) -> np.ndarray:
    """Funcion de alto nivel: de imagen a vector de caracteristicas topologicas.

    Es el punto de entrada que usara el pipeline/notebook. Encapsula los dos
    pasos (calcular diagrama -> vectorizar) y permite elegir la representacion.

    method:
        'image' -> imagen de persistencia (mas rica, por defecto).
        'stats' -> estadisticos de persistencia (mas compacta).
    Devuelve un vector 1D (una sola muestra).
    """
    diagrams = persistence_diagram(image)
    if method == "image":
        return persistence_image_vector(diagrams, n_bins=n_bins)[0]
    if method == "stats":
        return persistence_statistics(diagrams)[0]
    raise ValueError(f"Metodo desconocido: {method}")
