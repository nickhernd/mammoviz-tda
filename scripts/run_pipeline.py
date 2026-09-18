"""Ejecuta el pipeline TDA completo de un tiron sobre los casos reales.

Un unico comando, sin ir fichero por fichero ni celda por celda:

    python scripts/run_pipeline.py

Hace, en este orden:
  1. Carga los CSV reales de CBIS-DDSM (data/csv/, ver README) y se queda solo
     con los casos cuyas imagenes YA estan descargadas en data/images/ (ver
     scripts/download_cbis_ddsm.py --add N N; el piloto puede ampliarse en
     cualquier momento sin tocar este script).
  2. Particiona esos casos por PACIENTE (src.data.split_by_patient), para no
     repetir la fuga de datos del Demo_Notebook.ipynb original.
  3. Carga cada imagen DICOM real (pydicom) + su mascara de ROI, y las
     preprocesa (src.preprocessing: recorte de ROI, CLAHE, resize a 224x224).
  4. Calcula el descriptor topologico de cada imagen (src.topology: diagrama
     de persistencia H0/H1 + Persistence Image) con el backend disponible
     (giotto-tda o gudhi, ver `python -c "import src.topology as t; print(t.backend())"`).
  5. Entrena un clasificador clasico (SVM por defecto) sobre las features
     topologicas del split de train, y lo evalua con metricas clinicas sobre
     el split de test (src.evaluate.clinical_metrics).
  6. Guarda en results/: un diagrama de persistencia de ejemplo, la matriz de
     confusion y un resumen en texto (metrics.txt) con las metricas finales.

Piloto pequeno == particiones pequenas: con ~30-60 casos, val/test pueden
tener muy pocos pacientes. Es esperado en esta fase (F1); los numeros no son
concluyentes hasta ampliar el dataset (issue #82, F3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import case_folder_key, load_case_descriptions, split_by_patient  # noqa: E402
from src.evaluate import clinical_metrics, plot_confusion, plot_persistence_diagram  # noqa: E402
from src.models import build_random_forest, build_svm  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.topology import backend, persistence_diagram, persistence_image_vector  # noqa: E402

IMAGES_DIR = ROOT / "data" / "images"
RESULTS_DIR = ROOT / "results"


def _read_dicom_gray(path: Path) -> np.ndarray:
    return pydicom.dcmread(path).pixel_array.astype(np.float32)


def _load_case_images(row) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Carga la mamografia completa + su mascara de ROI para una fila del CSV.

    Devuelve ``None`` si las imagenes de este caso no estan descargadas
    todavia (piloto parcial). La mascara puede ser ``None`` si el caso solo
    tiene el recorte pre-hecho pero no la mascara binaria (variante rara de
    CBIS-DDSM, ver scripts/download_cbis_ddsm.py:download_case).
    """
    prefix = row["_prefix"]
    full_key = case_folder_key(prefix, row["patient_id"], row["left_or_right_breast"], row["image_view"])
    lesion_key = case_folder_key(
        prefix, row["patient_id"], row["left_or_right_breast"], row["image_view"], row["abnormality_id"]
    )

    full_dir = IMAGES_DIR / full_key
    lesion_dir = IMAGES_DIR / lesion_key
    if not full_dir.exists() or not lesion_dir.exists():
        return None

    full_dcms = sorted(full_dir.glob("*.dcm"))
    if not full_dcms:
        return None
    full_image = _read_dicom_gray(full_dcms[0])

    # La carpeta de la lesion puede tener 1 o 2 instancias DICOM repartidas en
    # subcarpetas por SeriesDescription (ver nota en download_cbis_ddsm.py: la
    # mascara y el recorte suelen venir juntos en una misma serie "ROI mask
    # images"). Distinguimos la mascara por su forma: coincide con la de la
    # mamografia completa (la mascara es un overlay a resolucion completa,
    # binario); el recorte es mas pequeno y con rango de grises continuo.
    mask = None
    for dcm_path in lesion_dir.glob("**/*.dcm"):
        arr = _read_dicom_gray(dcm_path)
        if arr.shape == full_image.shape:
            mask = arr
            break

    return full_image, mask


def build_dataset(df, n_bins: int = 20):
    """Preprocesa + extrae features topologicas de todos los casos disponibles."""
    features, labels, kept_rows = [], [], []
    skipped = 0
    for _, row in df.iterrows():
        loaded = _load_case_images(row)
        if loaded is None:
            skipped += 1
            continue
        full_image, mask = loaded
        img = preprocess(full_image, mask=mask)
        diagram = persistence_diagram(img)
        vector = persistence_image_vector(diagram, n_bins=n_bins)[0]
        features.append(vector)
        labels.append(row["label"])
        kept_rows.append(row)

    if skipped:
        print(f"  ({skipped} casos del CSV omitidos: imagenes no descargadas todavia)")
    return np.asarray(features), np.asarray(labels), kept_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", choices=["svm", "rf"], default="svm")
    parser.add_argument("--n-bins", type=int, default=20, help="Resolucion de la Persistence Image (ver issue #77).")
    args = parser.parse_args()

    print(f"Backend TDA activo: {backend()}")

    print("Cargando metadatos reales de CBIS-DDSM (data/csv/)...")
    df = load_case_descriptions(ROOT / "data" / "csv")
    df = df[df["_prefix"].str.startswith("Mass")].reset_index(drop=True)  # el piloto actual solo tiene 'mass'

    print("Particionando por paciente (train/val/test, sin fuga de datos)...")
    partitions = split_by_patient(df)

    print("Cargando imagenes DICOM reales + extrayendo descriptores topologicos...")
    print("  [train]")
    X_train, y_train, _ = build_dataset(partitions["train"], n_bins=args.n_bins)
    print("  [test]")
    X_test, y_test, test_rows = build_dataset(partitions["test"], n_bins=args.n_bins)

    print(f"\nCasos usables: {len(X_train)} train, {len(X_test)} test.")
    if len(X_train) == 0 or len(X_test) == 0:
        print("No hay suficientes casos con imagenes descargadas todavia.")
        print("Amplia el piloto con: python scripts/download_cbis_ddsm.py --add 15 15")
        return

    print(f"Entrenando clasificador ({args.classifier})...")
    clf = build_svm() if args.classifier == "svm" else build_random_forest()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    metrics = clinical_metrics(y_test, y_pred, y_score=y_score)

    print("\nMetricas clinicas sobre el conjunto de test (datos REALES):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ax = plot_confusion(y_test, y_pred)
    ax.figure.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    example_diagram = persistence_diagram(
        preprocess(*_load_case_images(test_rows[0]))
    )
    ax = plot_persistence_diagram(example_diagram)
    ax.figure.savefig(RESULTS_DIR / "persistence_diagram_example.png", dpi=150, bbox_inches="tight")

    with open(RESULTS_DIR / "metrics.txt", "w") as f:
        f.write(f"backend TDA: {backend()}\n")
        f.write(f"clasificador: {args.classifier}\n")
        f.write(f"n_train={len(X_train)} n_test={len(X_test)}\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    print(f"\nResultados guardados en {RESULTS_DIR}/ (confusion_matrix.png, "
          f"persistence_diagram_example.png, metrics.txt)")


if __name__ == "__main__":
    main()
