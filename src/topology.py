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

Backends disponibles (se elige automaticamente el primero disponible):
  1. ``giotto-tda`` (``pip install giotto-tda``): implementacion de referencia
     usada en el resto de la memoria. Solo tiene ruedas (*wheels*) publicadas
     hasta Python 3.12, por lo que en entornos con Python mas reciente no se
     puede instalar.
  2. ``gudhi`` (``pip install gudhi persim``): motor de homologia persistente
     alternativo con soporte binario para Python 3.10-3.14. Se usa
     ``gudhi.CubicalComplex`` para el diagrama y una imagen de persistencia
     calculada a mano (nucleo gaussiano ponderado por persistencia) porque las
     imagenes de entrada ya estan normalizadas a [0, 1] por
     ``preprocessing.to_grayscale_float``, lo que permite fijar la rejilla
     birth/persistencia sin tener que ajustarla (*fit*) por lote como hace
     giotto-tda.

Si ninguno de los dos esta instalado, las funciones que dependen de ellos
lanzan ``RuntimeError`` con un mensaje explicito; el resto del modulo se puede
importar sin problema.
"""
from __future__ import annotations

import numpy as np

# Importacion tolerante: el modulo se puede importar sin ningun backend TDA
# instalado; solo fallaran las funciones que realmente lo necesiten.
try:
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import (
        PersistenceImage,
        Scaler,
        Amplitude,
        NumberOfPoints,
    )
    _HAS_GIOTTO = True
except ImportError:  # pragma: no cover
    CubicalPersistence = None
    _HAS_GIOTTO = False

try:
    import gudhi
    _HAS_GUDHI = True
except ImportError:  # pragma: no cover
    gudhi = None
    _HAS_GUDHI = False


def backend() -> str:
    """Nombre del backend de homologia persistente activo en este entorno.

    Devuelve ``"giotto-tda"``, ``"gudhi"`` o ``"none"``. Util para que los
    notebooks informen de que via se esta usando sin duplicar la logica de
    deteccion de dependencias.
    """
    if _HAS_GIOTTO:
        return "giotto-tda"
    if _HAS_GUDHI:
        return "gudhi"
    return "none"


def persistence_diagram(image: np.ndarray, homology_dims=(0, 1)) -> np.ndarray:
    """Calcula el diagrama de persistencia cubico de una imagen 2D.

    homology_dims=(0, 1) pide dos tipos de caracteristicas:
      - dimension 0 (H0): componentes conexas (islas de intensidad).
      - dimension 1 (H1): agujeros / ciclos (util para masas espiculadas,
        distribucion de microcalcificaciones...).

    Devuelve un array con forma (1, n_puntos, 3), donde cada punto es
    (birth, death, dimension). El calculo delega en giotto-tda si esta
    disponible y, si no, en gudhi (vease el modulo).
    """
    if _HAS_GIOTTO:
        # giotto-tda espera un lote de imagenes; anadimos un eje de lote -> (1, H, W).
        x = np.asarray(image, dtype=np.float32)[None, :, :]
        # CubicalPersistence es el estimador que calcula la homologia sobre la
        # rejilla de pixeles (complejo cubico). n_jobs=-1 usa todos los nucleos.
        cubical = CubicalPersistence(homology_dimensions=homology_dims, n_jobs=-1)
        return cubical.fit_transform(x)
    if _HAS_GUDHI:
        return _persistence_diagram_gudhi(image, homology_dims)
    raise RuntimeError(
        "Se necesita un backend de homologia persistente. Instala uno de los "
        "dos: 'pip install giotto-tda' (hasta Python 3.12) o "
        "'pip install gudhi persim' (Python 3.10-3.14)."
    )


def _persistence_diagram_gudhi(image: np.ndarray, homology_dims) -> np.ndarray:
    """Diagrama de persistencia cubico con gudhi.CubicalComplex.

    gudhi representa cada imagen como un complejo cubico donde la altura de
    cada celda es la intensidad del pixel (filtracion por sublevel sets,
    igual que giotto-tda). ``persistence()`` devuelve pares (dimension,
    (birth, death)); la componente conexa "de fondo" en H0 nunca muere
    (death = infinito) porque cubre toda la imagen, asi que se sustituye ese
    infinito por la intensidad maxima de la imagen (convencion habitual: el
    valor de "muerte" para una clase esencial es el final de la filtracion).
    """
    img = np.asarray(image, dtype=np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=img)
    pairs = cc.persistence(homology_coeff_field=2, min_persistence=0.0)

    cap = float(img.max())
    points = [
        (birth, cap if np.isinf(death) else death, float(dim))
        for dim, (birth, death) in pairs
        if dim in homology_dims
    ]
    if not points:
        return np.empty((1, 0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)[None, :, :]


def persistence_image_vector(
    diagrams: np.ndarray, n_bins: int = 20
) -> np.ndarray:
    """Vectoriza los diagramas como 'imagenes de persistencia' aplanadas.

    Un diagrama de persistencia es un conjunto de puntos de tamano variable, y
    los algoritmos de ML necesitan vectores de tamano FIJO. La 'persistence
    image' resuelve esto: convierte el diagrama en una rejilla n_bins x n_bins
    (una imagen), que luego aplanamos a un vector.

    Con giotto-tda se usa su implementacion de referencia (Scaler +
    PersistenceImage). Sin giotto-tda se usa una implementacion propia
    (``_persistence_image_manual``) sobre la rejilla fija [0, 1] x [0, 1].
    """
    if _HAS_GIOTTO:
        # Scaler reescala los diagramas a un rango comun antes de vectorizar.
        diagrams = Scaler().fit_transform(diagrams)
        pi = PersistenceImage(n_bins=n_bins, n_jobs=-1)
        images = pi.fit_transform(diagrams)
        # reshape(len, -1): aplana cada imagen 2D (por dimension) a un vector 1D.
        return images.reshape(images.shape[0], -1)
    return _persistence_image_manual(diagrams, n_bins=n_bins)


def _persistence_image_manual(diagrams: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Imagen de persistencia calculada a mano (nucleo gaussiano ponderado).

    Para cada punto (birth, death) del diagrama se coloca una gaussiana 2D
    centrada en (birth, persistencia = death - birth), ponderada por la propia
    persistencia (las caracteristicas mas duraderas pesan mas), sobre una
    rejilla n_bins x n_bins. Es la misma idea que la 'persistence image'
    original (Adams et al., 2017), simplificada para no depender de una
    libreria de vectorizacion externa.

    Al asumir birth/death en [0, 1] (las imagenes llegan normalizadas por
    ``preprocessing.to_grayscale_float``), la rejilla es FIJA para todas las
    imagenes del conjunto de datos: no hace falta ajustar (fit) el rango por
    lote, y todos los vectores resultantes tienen la misma longitud y son
    directamente comparables entre imagenes.
    """
    diagrams = np.asarray(diagrams, dtype=np.float64)
    n_samples = diagrams.shape[0]
    dims = (
        sorted({int(d) for sample in diagrams for d in sample[:, 2]})
        if diagrams.size
        else [0, 1]
    )

    grid = np.linspace(0.0, 1.0, n_bins)
    birth_grid, pers_grid = np.meshgrid(grid, grid, indexing="ij")
    sigma = 1.0 / n_bins

    out = np.zeros((n_samples, len(dims), n_bins, n_bins))
    for s in range(n_samples):
        points = diagrams[s]
        for k, dim in enumerate(dims):
            selected = points[points[:, 2] == dim] if len(points) else points
            if len(selected) == 0:
                continue
            births = selected[:, 0]
            persistences = selected[:, 1] - selected[:, 0]
            weights = np.clip(persistences, 0.0, None)
            for birth, pers, weight in zip(births, persistences, weights):
                out[s, k] += weight * np.exp(
                    -((birth_grid - birth) ** 2 + (pers_grid - pers) ** 2)
                    / (2 * sigma ** 2)
                )
    return out.reshape(n_samples, -1)


def persistence_statistics(diagrams: np.ndarray) -> np.ndarray:
    """Estadisticos compactos de persistencia (alternativa ligera al vector imagen).

    En vez de una imagen completa, resume cada diagrama con pocos numeros por
    dimension de homologia: cuanta persistencia total hay y cuantas
    caracteristicas topologicas se han detectado. Utiles como baseline rapido
    o para combinar con otras caracteristicas.

    Con giotto-tda se usan sus transformadores de referencia (Amplitude con
    metrica Wasserstein + NumberOfPoints). Sin giotto-tda se usa una
    implementacion propia con la persistencia total (suma de death - birth)
    como aproximacion mas simple a la amplitud.
    """
    if _HAS_GIOTTO:
        amp = Amplitude(metric="wasserstein").fit_transform(diagrams)
        npts = NumberOfPoints().fit_transform(diagrams)
        # hstack pega ambos bloques de columnas -> un vector de caracteristicas por muestra.
        return np.hstack([amp, npts])
    return _persistence_statistics_manual(diagrams)


def _persistence_statistics_manual(diagrams: np.ndarray) -> np.ndarray:
    """Persistencia total y numero de puntos por dimension, sin giotto-tda."""
    diagrams = np.asarray(diagrams, dtype=np.float64)
    n_samples = diagrams.shape[0]
    dims = (
        sorted({int(d) for sample in diagrams for d in sample[:, 2]})
        if diagrams.size
        else [0, 1]
    )

    out = np.zeros((n_samples, 2 * len(dims)))
    for s in range(n_samples):
        points = diagrams[s]
        for k, dim in enumerate(dims):
            selected = points[points[:, 2] == dim] if len(points) else points
            total_persistence = (
                float(np.sum(selected[:, 1] - selected[:, 0])) if len(selected) else 0.0
            )
            out[s, 2 * k] = total_persistence
            out[s, 2 * k + 1] = len(selected)
    return out


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
