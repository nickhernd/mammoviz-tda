"""Valida la metodologia de deteccion anticipada longitudinal con datos
SINTETICOS, mientras se tramita el acceso a un dataset real longitudinal
(CSAW-CC / EMBED, ver issue #92 -- ambos requieren solicitud formal, no hay
descarga directa).

Pregunta que responde este script (issue #94, diseno de la comparacion
topologica): dado un par de "exploraciones" sucesivas de la misma paciente
(t-1 = previa, t = actual), ?aporta la homologia persistente senal capaz de
distinguir:

  (a) SEÑAL DE ANTICIPACION: si ya en t-1 (la exploracion leida como normal)
      hay un precursor sutil, ?se distingue topologicamente de una t-1
      verdaderamente sana (grupo control), SOLO con la informacion de t-1?
  (b) SEÑAL DE DERIVA: ?el cambio topologico entre t-1 y t (distancia
      bottleneck/Wasserstein entre sus diagramas de persistencia) es mayor
      en casos incidentes (precursor -> masa) que en controles sanos
      (ruido -> ruido)?

Generador sintetico (reproducible, seed fija):
  - Fondo: ruido gaussiano suavizado (textura de tejido mamario simulada).
  - Casos incidentes: en t-1 se inserta un precursor PEQUEÑO y de bajo
    contraste; en t, en la MISMA posicion, una masa mucho mas prominente e
    irregular (simula crecimiento real de una lesion).
  - Controles: fondo solo, sin ninguna lesion, en t-1 y en t (con ruido
    independiente en cada visita, como en una paciente sana real).

Uso:
    python scripts/synthetic_longitudinal_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import persim
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topology import persistence_diagram, persistence_image_vector  # noqa: E402

IMAGE_SIZE = 96
CENTER = (48, 48)


def _background(rng: np.random.Generator) -> np.ndarray:
    """Textura de fondo: ruido gaussiano suavizado, normalizado a [0, 1]."""
    noise = rng.normal(size=(IMAGE_SIZE, IMAGE_SIZE))
    bg = gaussian_filter(noise, sigma=IMAGE_SIZE / 8)
    bg -= bg.min()
    bg /= bg.max()
    return bg


def _add_lesion(img: np.ndarray, radius: float, amplitude: float, irregularity: float,
                 rng: np.random.Generator) -> np.ndarray:
    """Anade una lesion radialmente simetrica pero con borde irregular
    (espiculado): el radio efectivo se modula angularmente con una suma de
    armonicos de fase aleatoria, controlada por `irregularity`.
    """
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    dx, dy = xx - CENTER[0], yy - CENTER[1]
    theta = np.arctan2(dy, dx)
    r = np.sqrt(dx ** 2 + dy ** 2)

    modulation = np.ones_like(theta)
    for k in range(2, 6):
        phase = rng.uniform(0, 2 * np.pi)
        modulation += (irregularity / 4) * np.cos(k * theta + phase)

    effective_radius = radius * modulation
    lesion = amplitude * np.exp(-(r / np.clip(effective_radius, 1e-3, None)) ** 2)
    out = img + lesion
    return out / out.max()  # renormaliza a [0, 1] tras anadir la lesion


def make_patient(
    kind: str, seed: int, precursor_radius: float = 4, precursor_amplitude: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Genera (imagen_t_menos_1, imagen_t) para un paciente sintetico.

    kind='incidente': precursor en t-1 (tamano/contraste parametrizable, para
    el analisis de sensibilidad: que tan sutil puede ser antes de perder la
    senal) -> masa prominente en t (misma posicion, simulando crecimiento
    real de una lesion).
    kind='control': solo fondo (ruido independiente) en t-1 y en t.
    """
    rng = np.random.default_rng(seed)
    img_t_minus_1 = _background(rng)
    img_t = _background(rng)  # ruido independiente: misma paciente, visita distinta

    if kind == "incidente":
        img_t_minus_1 = _add_lesion(
            img_t_minus_1, radius=precursor_radius, amplitude=precursor_amplitude,
            irregularity=0.3, rng=rng,
        )
        img_t = _add_lesion(img_t, radius=11, amplitude=0.9, irregularity=1.1, rng=rng)

    return img_t_minus_1, img_t


def _diagram_by_dim(diagram: np.ndarray, dim: int, max_points: int | None = None) -> np.ndarray:
    """Extrae del diagrama (1, n, 3) los puntos (birth, death) de una
    dimension, en el formato (n, 2) que espera persim.bottleneck/wasserstein.

    ``max_points``: si se da, se queda solo con los `max_points` puntos de
    MAYOR persistencia (death - birth). Necesario en la practica: las
    imagenes sinteticas de este modulo producen decenas de puntos (barato
    para persim.bottleneck), pero una mamografia real preprocesada puede
    producir miles (observado: ~1000-2000 en H0, ~1800-2200 en H1), y
    persim.bottleneck escala muy mal con el numero de puntos (medido:
    ~121s para un solo par de diagramas reales sin recortar, issue #94).
    Quedarse con los puntos mas persistentes es la practica habitual en TDA
    para descartar "ruido topologico" de baja persistencia, y hace que el
    calculo sea tratable. Es una APROXIMACION, no una garantia: la
    distancia bottleneck no es en general invariante a quitar puntos de
    baja persistencia (puede cambiar el emparejamiento optimo con la
    diagonal), aunque para diagramas con una señal dominante de alta
    persistencia el cambio es pequeño (test de regresion en
    tests/test_synthetic_longitudinal_validation.py). Por defecto
    (``None``) no se recorta nada, para no alterar los resultados ya
    reportados sobre datos sinteticos.
    """
    points = diagram[0]
    points = points[points[:, 2] == dim][:, :2]
    if max_points is not None and len(points) > max_points:
        persistence = points[:, 1] - points[:, 0]
        top_idx = np.argsort(-persistence)[:max_points]
        points = points[top_idx]
    return points


def topological_drift(
    diagram_a: np.ndarray, diagram_b: np.ndarray, max_points: int | None = None
) -> tuple[float, float]:
    """Distancia bottleneck entre los diagramas H0 y H1 de dos imagenes.

    Ver ``_diagram_by_dim`` sobre ``max_points``: imprescindible pasarlo en
    datos reales (p.ej. ``max_points=100``) o el calculo puede tardar
    minutos por par en vez de milisegundos.
    """
    h0_dist = persim.bottleneck(
        _diagram_by_dim(diagram_a, 0, max_points), _diagram_by_dim(diagram_b, 0, max_points)
    )
    h1_dist = persim.bottleneck(
        _diagram_by_dim(diagram_a, 1, max_points), _diagram_by_dim(diagram_b, 1, max_points)
    )
    return h0_dist, h1_dist


def run_experiment(
    n_per_class: int = 60, n_bins: int = 6, seed: int = 42,
    precursor_radius: float = 4, precursor_amplitude: float = 0.25,
) -> dict[str, float]:
    """Genera el cohorte sintetico y devuelve el AUC (5-fold CV) de las 3
    senales (anticipacion, deriva, combinada). Reutilizada por `main()` (una
    sola corrida, con impresion detallada) y por `sweep_sensitivity()`
    (muchas corridas variando `precursor_amplitude`/`precursor_radius`).
    """
    rng_master = np.random.default_rng(seed)
    kinds = ["incidente"] * n_per_class + ["control"] * n_per_class
    seeds = rng_master.integers(0, 10**6, size=len(kinds))

    feats_t_minus_1, feats_drift, labels = [], [], []
    for kind, s in zip(kinds, seeds):
        img_t_minus_1, img_t = make_patient(
            kind, seed=int(s), precursor_radius=precursor_radius, precursor_amplitude=precursor_amplitude
        )

        diagram_t_minus_1 = persistence_diagram(img_t_minus_1)
        diagram_t = persistence_diagram(img_t)

        feats_t_minus_1.append(persistence_image_vector(diagram_t_minus_1, n_bins=n_bins)[0])
        feats_drift.append(topological_drift(diagram_t_minus_1, diagram_t))
        labels.append(1 if kind == "incidente" else 0)

    X_t_minus_1 = np.asarray(feats_t_minus_1)
    X_drift = np.asarray(feats_drift)
    X_combined = np.hstack([X_t_minus_1, X_drift])
    y = np.asarray(labels)

    def _cv_auc(X):
        scores = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=seed),
                                  X, y, cv=5, scoring="roc_auc")
        return float(scores.mean()), float(scores.std())

    auc_a, std_a = _cv_auc(X_t_minus_1)
    auc_b, std_b = _cv_auc(X_drift)
    auc_c, std_c = _cv_auc(X_combined)
    return {
        "auc_anticipacion": auc_a, "std_anticipacion": std_a,
        "auc_deriva": auc_b, "std_deriva": std_b,
        "auc_combinada": auc_c, "std_combinada": std_c,
    }


def sweep_sensitivity(n_per_class: int = 60, n_bins: int = 6, seed: int = 42) -> None:
    """Analisis de sensibilidad (issue #94): ?que tan sutil puede ser el
    precursor en t-1 antes de que la senal de anticipacion se pierda?
    Informa directamente cuanta antelacion real cabria esperar detectar.
    """
    amplitudes = [0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03]
    print("Barrido de sensibilidad: amplitud del precursor en t-1 (radio fijo=4)")
    print(f"{'amplitud':>10}  {'AUC anticipacion':>18}  {'AUC deriva':>12}  {'AUC combinada':>14}")
    for amplitude in amplitudes:
        result = run_experiment(n_per_class=n_per_class, n_bins=n_bins, seed=seed, precursor_amplitude=amplitude)
        print(f"{amplitude:>10.2f}  "
              f"{result['auc_anticipacion']:.3f} +/- {result['std_anticipacion']:.3f}   "
              f"{result['auc_deriva']:.3f}        "
              f"{result['auc_combinada']:.3f}")
    print("\nAUC ~ 0.5 = la senal de anticipacion ya no es distinguible del azar con ese "
          "nivel de contraste del precursor. El punto donde empieza a caer marca el limite "
          "de sensibilidad del metodo en este escenario sintetico (NO es una cifra clinica).")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", action="store_true",
                         help="Analisis de sensibilidad: barre la amplitud del precursor en vez "
                              "de una sola corrida con los parametros por defecto.")
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sweep:
        sweep_sensitivity(n_per_class=args.n_per_class, n_bins=args.n_bins, seed=args.seed)
        return

    result = run_experiment(n_per_class=args.n_per_class, n_bins=args.n_bins, seed=args.seed)

    print(f"Pacientes sinteticos: {2 * args.n_per_class} ({args.n_per_class} incidentes, {args.n_per_class} control)")
    print("\n=== (a) Señal de ANTICIPACION: ¿t-1 solo distingue incidente vs. control? ===")
    print(f"  AUC (5-fold CV, solo features de t-1): {result['auc_anticipacion']:.3f} +/- {result['std_anticipacion']:.3f}")
    print("\n=== (b) Señal de DERIVA: ¿la distancia topologica t-1 -> t distingue incidente vs. control? ===")
    print(f"  AUC (5-fold CV, solo distancia bottleneck H0+H1): {result['auc_deriva']:.3f} +/- {result['std_deriva']:.3f}")
    print("\n=== (c) Combinada: t-1 + deriva ===")
    print(f"  AUC (5-fold CV, t-1 + deriva): {result['auc_combinada']:.3f} +/- {result['std_combinada']:.3f}")

    print("\nNota: esto es una validacion de METODOLOGIA sobre datos sinteticos "
          "controlados (la lesion incidental es, por construccion, detectable). "
          "Un AUC alto aqui confirma que el PIPELINE es capaz de medir la senal "
          "si existe -- NO implica que la senal exista en pacientes reales. "
          "Eso solo se podra confirmar con datos longitudinales reales (issue #92). "
          "Usa --sweep para ver el limite de sensibilidad del metodo.")


if __name__ == "__main__":
    main()
