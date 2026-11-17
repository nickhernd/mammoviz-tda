"""Evalua la hipotesis de deteccion anticipada SOBRE DATOS REALES por
primera vez (O8, issues #92/#94), usando el piloto longitudinal de EMBED
descargado por scripts/download_embed_longitudinal.py.

Uso:
    python scripts/embed_longitudinal_evaluation.py

Señales:
  - ANTICIPACION: vector de Persistence Image de la exploracion t-1 sola.
  - DERIVA: distancia bottleneck (H0, H1) entre los diagramas de t-1 y t.
  - COMBINADA: concatenacion de ambas.

Aviso metodologico (léase antes de interpretar el AUC): con 40 pacientes
(20 incidentes + 20 control) este es un PILOTO EXPLORATORIO, no un
resultado concluyente -- el mismo aviso que se ha repetido en todo el
proyecto para piloto pequeños se aplica aqui con más fuerza todavia, al
ser la primera vez que se prueba la hipotesis principal del TFG con datos
reales en lugar de sinteticos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.preprocessing import apply_clahe, read_dicom_gray, resize, to_grayscale_float  # noqa: E402
from src.topology import persistence_diagram, persistence_image_vector, topological_drift  # noqa: E402

EMBED_DIR = ROOT / "data" / "embed"
RESULTS_DIR = ROOT / "results"


def _load_and_preprocess(path: Path, size=(224, 224)) -> np.ndarray:
    """Igual que src.preprocessing.preprocess, pero sin recorte de ROI: EMBED
    no trae mascara de segmentacion de la lesion (a diferencia de
    CBIS-DDSM), asi que se usa la mamografia completa."""
    arr = read_dicom_gray(path)  # invierte MONOCHROME1 (Fujifilm)
    gray = to_grayscale_float(arr)
    clahe = apply_clahe(gray)
    return resize(clahe, size)


CACHE_PATH = EMBED_DIR / "features_cache.pkl"
# Se incrementa cuando cambia el preprocesado, para invalidar la cache:
# v2 = lectura con read_dicom_gray (invierte las imagenes MONOCHROME1).
FEATURES_VERSION = 2


def _pair_features(path_t0: Path, path_t1: Path, n_bins: int):
    """Anticipacion (Persistence Image de t-1) y deriva (bottleneck H0/H1)
    de un par. max_points=100: los diagramas reales tienen miles de puntos y
    persim.bottleneck sin recortar tarda minutos por par; recortar a los
    mas persistentes es una aproximacion estandar (ver
    src/topology.py:diagram_points)."""
    diagram_t0 = persistence_diagram(_load_and_preprocess(path_t0))
    diagram_t1 = persistence_diagram(_load_and_preprocess(path_t1))
    return (persistence_image_vector(diagram_t0, n_bins=n_bins)[0],
            np.asarray(topological_drift(diagram_t0, diagram_t1, max_points=100)))


def compute_features(manifest: pd.DataFrame, n_bins: int = 6, verbose: bool = True):
    """Features de todo el manifiesto, con cache en disco por par
    (path_t0, path_t1, n_bins): re-evaluar otra variante del piloto (p.ej.
    el manifiesto emparejado por intervalo, que solo cambia el t de los
    controles) solo recalcula los pares nuevos.

    Devuelve (X_anticipacion, X_deriva, y)."""
    import pickle

    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    feats_t0, feats_drift, labels = [], [], []
    n_new = 0
    for _, row in manifest.iterrows():
        key = (row["path_t0"], row["path_t1"], n_bins, FEATURES_VERSION)
        if key not in cache:
            cache[key] = _pair_features(ROOT / row["path_t0"], ROOT / row["path_t1"], n_bins)
            n_new += 1
            if verbose:
                print(f"  {row['empi_anon']} ({row['kind']}, {row['view']}): procesado", flush=True)
        pi, drift = cache[key]
        feats_t0.append(pi)
        feats_drift.append(drift)
        labels.append(1 if row["kind"] == "incidente" else 0)
    if n_new:
        CACHE_PATH.write_bytes(pickle.dumps(cache))
    if verbose:
        print(f"  ({n_new} pares calculados, {len(manifest) - n_new} desde cache)")
    return np.asarray(feats_t0), np.asarray(feats_drift), np.asarray(labels)


IMAGE_CACHE_PATH = EMBED_DIR / "image_pi_cache.pkl"


def compute_t1_features(manifest: pd.DataFrame, n_bins: int = 6, verbose: bool = True) -> np.ndarray:
    """Persistence Image de la exploracion t (la del diagnostico) de cada par.

    Es el control positivo del diseño: en t el cancer ya esta presente, asi
    que si ni siquiera t sola separa casos de controles, el descriptor no ve
    estas lesiones y no cabe esperar anticipacion desde t-1."""
    import pickle

    cache = pickle.loads(IMAGE_CACHE_PATH.read_bytes()) if IMAGE_CACHE_PATH.exists() else {}
    feats, n_new = [], 0
    for _, row in manifest.iterrows():
        key = (row["path_t1"], n_bins, FEATURES_VERSION)
        if key not in cache:
            diagram = persistence_diagram(_load_and_preprocess(ROOT / row["path_t1"]))
            cache[key] = persistence_image_vector(diagram, n_bins=n_bins)[0]
            n_new += 1
        feats.append(cache[key])
    if n_new:
        IMAGE_CACHE_PATH.write_bytes(pickle.dumps(cache))
    if verbose:
        print(f"  (t: {n_new} imagenes calculadas, {len(manifest) - n_new} desde cache)")
    return np.asarray(feats)


def intervals_days(manifest: pd.DataFrame) -> np.ndarray:
    """Intervalo t-1 -> t de cada par, en dias (requiere acc_t0/acc_t1)."""
    from download_embed_longitudinal import _load_per_study_assessment

    dates = _load_per_study_assessment(EMBED_DIR / "clinical.csv").set_index(
        ["empi_anon", "acc_anon"])["study_date"]
    return np.asarray([
        (dates[(r["empi_anon"], r["acc_t1"])] - dates[(r["empi_anon"], r["acc_t0"])]).days
        for _, r in manifest.iterrows()
    ], dtype=float)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="longitudinal_pilot_manifest.csv",
                        help="Manifiesto dentro de data/embed/ (p.ej. "
                             "longitudinal_pilot_manifest_matched.csv, el emparejado por intervalo).")
    parser.add_argument("--n-bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = EMBED_DIR / args.manifest
    if not manifest_path.exists():
        print(f"No se encuentra {manifest_path}. "
              f"Ejecuta antes scripts/download_embed_longitudinal.py")
        return
    manifest = pd.read_csv(manifest_path)
    n_inc = (manifest["kind"] == "incidente").sum()
    n_ctl = (manifest["kind"] == "control").sum()
    print(f"Piloto longitudinal real de EMBED ({args.manifest}): {len(manifest)} pacientes "
          f"({n_inc} incidentes, {n_ctl} control).")

    X_anticipacion, X_deriva, y = compute_features(manifest, n_bins=args.n_bins)
    X_combinada = np.hstack([X_anticipacion, X_deriva])

    resultados = {}

    def _cv_auc(X, name):
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=300, random_state=args.seed, class_weight="balanced"),
            X, y, cv=5, scoring="roc_auc",
        )
        print(f"  {name:22s}: AUC (5-fold CV) = {scores.mean():.3f} +/- {scores.std():.3f}")
        resultados[name] = (scores.mean(), scores.std())

    print(f"\n=== Resultados sobre el piloto REAL de EMBED (n={len(manifest)}) ===")
    _cv_auc(X_anticipacion, "anticipacion")
    _cv_auc(X_deriva, "deriva")
    _cv_auc(X_combinada, "combinada")

    print("\nControl positivo (t sola, con el cancer ya presente):")
    _cv_auc(compute_t1_features(manifest, n_bins=args.n_bins), "diagnostico (solo t)")

    if {"acc_t0", "acc_t1"} <= set(manifest.columns):
        dias = intervals_days(manifest)
        print(f"\nIntervalo t-1 -> t (mediana): incidentes {np.median(dias[y == 1]):.0f} dias, "
              f"controles {np.median(dias[y == 0]):.0f} dias")
        print("Lineas base de control del sesgo de intervalo (si 'intervalo solo' da un AUC")
        print("alto, el diseño deja que el tiempo transcurrido distinga las clases):")
        _cv_auc(dias[:, None], "intervalo solo")
        _cv_auc(np.hstack([X_deriva, dias[:, None]]), "deriva + intervalo")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = Path(args.manifest).stem.replace("longitudinal_pilot_manifest", "embed_longitudinal_pilot")
    stem = stem.replace("longitudinal_manifest", "longitudinal")
    if not stem.startswith("embed_"):
        stem = f"embed_{stem}"
    out_path = RESULTS_DIR / f"{stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Piloto longitudinal REAL (EMBED, {args.manifest}): {len(manifest)} pacientes "
                f"({n_inc} incidentes, {n_ctl} control)\n")
        f.write(f"AVISO: piloto exploratorio de {len(manifest)} pacientes, no concluyente.\n\n")
        for name, (mean, std) in resultados.items():
            f.write(f"auc_{name.replace(' ', '_').replace('+', 'mas')}: {mean:.4f} +/- {std:.4f}\n")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
