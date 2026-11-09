"""Evalua la hipotesis de deteccion anticipada SOBRE DATOS REALES por
primera vez (O8, issues #92/#94), usando el piloto longitudinal de EMBED
descargado por scripts/download_embed_longitudinal.py.

Uso:
    python scripts/embed_longitudinal_evaluation.py

Reutiliza exactamente las mismas funciones y señales que la validacion
sintetica (scripts/synthetic_longitudinal_validation.py), para que los
resultados sean directamente comparables:
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
import pydicom
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.preprocessing import apply_clahe, resize, to_grayscale_float  # noqa: E402
from src.topology import persistence_diagram, persistence_image_vector  # noqa: E402
from synthetic_longitudinal_validation import topological_drift  # noqa: E402

EMBED_DIR = ROOT / "data" / "embed"
RESULTS_DIR = ROOT / "results"


def _load_and_preprocess(path: Path, size=(224, 224)) -> np.ndarray:
    """Igual que src.preprocessing.preprocess, pero sin recorte de ROI: EMBED
    no trae mascara de segmentacion de la lesion (a diferencia de
    CBIS-DDSM), asi que se usa la mamografia completa."""
    arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    gray = to_grayscale_float(arr)
    clahe = apply_clahe(gray)
    return resize(clahe, size)


def main(n_bins: int = 6, seed: int = 42) -> None:
    manifest_path = EMBED_DIR / "longitudinal_pilot_manifest.csv"
    if not manifest_path.exists():
        print(f"No se encuentra {manifest_path}. "
              f"Ejecuta antes scripts/download_embed_longitudinal.py")
        return
    manifest = pd.read_csv(manifest_path)
    print(f"Piloto longitudinal real de EMBED: {len(manifest)} pacientes "
          f"({(manifest['kind'] == 'incidente').sum()} incidentes, "
          f"{(manifest['kind'] == 'control').sum()} control).")

    feats_t0, feats_drift, labels, patients = [], [], [], []
    for _, row in manifest.iterrows():
        img_t0 = _load_and_preprocess(ROOT / row["path_t0"])
        img_t1 = _load_and_preprocess(ROOT / row["path_t1"])

        diagram_t0 = persistence_diagram(img_t0)
        diagram_t1 = persistence_diagram(img_t1)

        feats_t0.append(persistence_image_vector(diagram_t0, n_bins=n_bins)[0])
        # max_points=100: diagramas reales tienen miles de puntos (vs. decenas
        # en los sinteticos) y persim.bottleneck sin recortar tarda minutos
        # por par en vez de milisegundos. Es una aproximacion (no una
        # igualdad garantizada, ver synthetic_longitudinal_validation.py:
        # _diagram_by_dim), necesaria para que el piloto sea tratable.
        feats_drift.append(topological_drift(diagram_t0, diagram_t1, max_points=100))
        labels.append(1 if row["kind"] == "incidente" else 0)
        patients.append(row["empi_anon"])
        print(f"  {row['empi_anon']} ({row['kind']}, {row['view']}): procesado")

    X_anticipacion = np.asarray(feats_t0)
    X_deriva = np.asarray(feats_drift)
    X_combinada = np.hstack([X_anticipacion, X_deriva])
    y = np.asarray(labels)

    print(f"\nVectores: anticipacion {X_anticipacion.shape}, deriva {X_deriva.shape}")

    def _cv_auc(X, name):
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced"),
            X, y, cv=5, scoring="roc_auc",
        )
        print(f"  {name:12s}: AUC (5-fold CV) = {scores.mean():.3f} +/- {scores.std():.3f}")
        return scores.mean(), scores.std()

    print("\n=== Resultados sobre el piloto REAL de EMBED (n=40, exploratorio) ===")
    auc_a, std_a = _cv_auc(X_anticipacion, "Anticipacion")
    auc_b, std_b = _cv_auc(X_deriva, "Deriva")
    auc_c, std_c = _cv_auc(X_combinada, "Combinada")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "embed_longitudinal_pilot.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Piloto longitudinal REAL (EMBED): {len(manifest)} pacientes "
                f"({(manifest['kind'] == 'incidente').sum()} incidentes, "
                f"{(manifest['kind'] == 'control').sum()} control)\n")
        f.write("AVISO: piloto exploratorio de 40 pacientes, no concluyente.\n\n")
        f.write(f"auc_anticipacion: {auc_a:.4f} +/- {std_a:.4f}\n")
        f.write(f"auc_deriva: {auc_b:.4f} +/- {std_b:.4f}\n")
        f.write(f"auc_combinada: {auc_c:.4f} +/- {std_c:.4f}\n")
    print(f"\nGuardado en {out_path}")


if __name__ == "__main__":
    main()
