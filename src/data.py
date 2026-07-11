"""Carga de metadatos de CBIS-DDSM y particion por paciente.

IMPORTANTE sobre los datos disponibles en el repositorio:

* El fichero ``CBIS-DDSM-All-...-nbia-digest.xlsx`` es el *manifiesto* de TCIA:
  describe las series DICOM (identificadores, tamano, etc.) pero **no incluye la
  etiqueta de patologia (benigno/maligno)**. Esa informacion se encuentra en los
  CSV de descripcion de casos de CBIS-DDSM:
  ``mass_case_description_train_set.csv``, ``calc_case_description_*_set.csv``, etc.
  Descargalos de TCIA y colocalos en ``data/csv/``.

* El campo ``Patient ID`` del manifiesto tiene la forma
  ``Mass-Training_P_01009_RIGHT_CC_1``. El identificador real de paciente es el
  segmento ``P_01009``; se usa para particionar por paciente y evitar fuga de
  datos.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# Etiquetas de patologia de CBIS-DDSM -> binaria (0 = benigno, 1 = maligno).
_PATHOLOGY_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}

_PATIENT_RE = re.compile(r"(P_\d+)")


def load_manifest(xlsx_path: str | Path) -> pd.DataFrame:
    """Carga el manifiesto TCIA (.xlsx) y extrae el id de paciente real."""
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip() for c in df.columns]
    df["patient"] = df["Patient ID"].str.extract(_PATIENT_RE.pattern)
    return df


def load_case_descriptions(csv_dir: str | Path) -> pd.DataFrame:
    """Carga y concatena los CSV de descripcion de casos de CBIS-DDSM.

    Cada CSV incluye, entre otras, las columnas ``patient_id``,
    ``pathology`` y las rutas a la imagen y a la mascara de ROI.
    """
    csv_dir = Path(csv_dir)
    frames = [pd.read_csv(p) for p in sorted(csv_dir.glob("*case_description*.csv"))]
    if not frames:
        raise FileNotFoundError(
            f"No se encontraron CSV de descripcion en {csv_dir}. "
            "Descargalos de CBIS-DDSM (TCIA)."
        )
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df["label"] = to_binary_label(df["pathology"])
    return df


def to_binary_label(pathology: pd.Series) -> pd.Series:
    """Convierte la etiqueta de patologia en binaria (0/1)."""
    return pathology.str.upper().str.strip().map(_PATHOLOGY_MAP)


def split_by_patient(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Particiona el DataFrame por paciente en train/val/test.

    Ningun paciente aparece en mas de una particion. La asignacion se
    estratifica por la etiqueta mayoritaria de cada paciente para mantener
    la proporcion de clases entre particiones.
    """
    rng = np.random.default_rng(seed)

    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: int(round(s.mean())))
        .reset_index()
    )

    partitions: dict[str, list] = {"train": [], "val": [], "test": []}
    for _label_value, group in per_patient.groupby(label_col):
        patients = group[patient_col].to_numpy()
        rng.shuffle(patients)
        n = len(patients)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        partitions["test"].extend(patients[:n_test])
        partitions["val"].extend(patients[n_test : n_test + n_val])
        partitions["train"].extend(patients[n_test + n_val :])

    return {
        name: df[df[patient_col].isin(ids)].reset_index(drop=True)
        for name, ids in partitions.items()
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    xlsx = root / "CBIS-DDSM-All-doiJNLP-zzWs5zfZ-nbia-digest.xlsx"
    manifest = load_manifest(xlsx)
    print("Filas del manifiesto:", len(manifest))
    print("Pacientes unicos:", manifest["patient"].nunique())
    print(manifest[["Patient ID", "patient"]].head())
