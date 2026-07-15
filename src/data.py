"""Carga de metadatos de CBIS-DDSM y particion por paciente.

Este modulo resuelve la PRIMERA etapa del pipeline: preparar los datos de forma
correcta y reproducible. Su responsabilidad mas importante es **evitar la fuga
de datos (data leakage)**, un error que inflaba artificialmente los resultados
del notebook original (accuracy = 1.0 desde la primera epoca).

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


# ---------------------------------------------------------------------------
# Constantes de configuracion
# ---------------------------------------------------------------------------

# Mapa de las 3 etiquetas de patologia de CBIS-DDSM a una etiqueta binaria.
# "BENIGN_WITHOUT_CALLBACK" son casos benignos que no requirieron seguimiento:
# clinicamente son benignos, por lo que se agrupan con la clase 0.
_PATHOLOGY_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT": 1,
}

# Expresion regular que extrae el id real de paciente (p.ej. "P_01009")
# del identificador largo del manifiesto ("Mass-Training_P_01009_RIGHT_CC_1").
_PATIENT_RE = re.compile(r"(P_\d+)")


# ---------------------------------------------------------------------------
# Carga de ficheros
# ---------------------------------------------------------------------------

def load_manifest(xlsx_path: str | Path) -> pd.DataFrame:
    """Carga el manifiesto TCIA (.xlsx) y extrae el id de paciente real.

    Parametros
    ----------
    xlsx_path : ruta al fichero .xlsx del manifiesto.

    Devuelve
    --------
    DataFrame con todas las columnas originales mas una columna nueva
    ``patient`` con el identificador de paciente normalizado.
    """
    df = pd.read_excel(xlsx_path)
    # Limpiamos espacios sobrantes en los nombres de columna (vienen de Excel).
    df.columns = [c.strip() for c in df.columns]
    # Extraemos "P_XXXXX" del identificador largo. .str.extract con un grupo
    # de captura devuelve una Serie con el primer grupo encontrado por fila.
    df["patient"] = df["Patient ID"].str.extract(_PATIENT_RE.pattern)
    return df


def load_case_descriptions(csv_dir: str | Path) -> pd.DataFrame:
    """Carga y concatena los CSV de descripcion de casos de CBIS-DDSM.

    Estos CSV (a diferencia del manifiesto) SI contienen la etiqueta clinica.
    Cada fila describe una lesion e incluye, entre otras, las columnas
    ``patient_id``, ``pathology`` y las rutas a la imagen y a la mascara de ROI.

    Lanza FileNotFoundError si el directorio no contiene ningun CSV esperado,
    para avisar cuanto antes de que faltan datos por descargar.
    """
    csv_dir = Path(csv_dir)
    # Buscamos todos los ficheros cuyo nombre contenga "case_description".
    frames = [pd.read_csv(p) for p in sorted(csv_dir.glob("*case_description*.csv"))]
    if not frames:
        raise FileNotFoundError(
            f"No se encontraron CSV de descripcion en {csv_dir}. "
            "Descargalos de CBIS-DDSM (TCIA)."
        )
    # Unimos los CSV de masas y calcificaciones (train y test) en un unico DataFrame.
    df = pd.concat(frames, ignore_index=True)
    # Normalizamos nombres de columna: sin espacios y con guion bajo
    # (CBIS-DDSM usa "patient id" con espacio en algunos ficheros).
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # Anadimos la etiqueta binaria derivada de la patologia.
    df["label"] = to_binary_label(df["pathology"])
    return df


def to_binary_label(pathology: pd.Series) -> pd.Series:
    """Convierte la columna de patologia (texto) en etiqueta binaria (0/1).

    Normaliza a mayusculas y sin espacios antes de mapear, para tolerar
    variaciones de formato entre ficheros.
    """
    return pathology.str.upper().str.strip().map(_PATHOLOGY_MAP)


# ---------------------------------------------------------------------------
# Particion de datos (nucleo de la prevencion de fuga de datos)
# ---------------------------------------------------------------------------

def split_by_patient(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Particiona el DataFrame por paciente en train/val/test.

    Por que POR PACIENTE y no por imagen: un mismo paciente puede tener varias
    vistas (CC/MLO, izquierda/derecha) de la misma lesion. Si esas imagenes
    cayeran en particiones distintas, el modelo "veria" en test informacion muy
    parecida a la de entrenamiento -> fuga de datos y metricas infladas.

    La asignacion se estratifica por la etiqueta mayoritaria de cada paciente,
    de modo que la proporcion benigno/maligno se mantiene similar en las tres
    particiones.

    Devuelve un diccionario {"train": df, "val": df, "test": df}.
    """
    # Generador aleatorio reproducible (misma semilla -> misma particion).
    rng = np.random.default_rng(seed)

    # Etiqueta representativa por paciente: redondeamos la media de sus lesiones.
    # Asi cada paciente entra entero en una sola particion (evita la fuga) y
    # ademas conocemos su clase para poder estratificar el reparto.
    per_patient = (
        df.groupby(patient_col)[label_col]
        .agg(lambda s: int(round(s.mean())))
        .reset_index()
    )

    partitions: dict[str, list] = {"train": [], "val": [], "test": []}
    # Repartimos DENTRO de cada clase por separado para mantener la proporcion.
    for _label_value, group in per_patient.groupby(label_col):
        patients = group[patient_col].to_numpy()
        rng.shuffle(patients)  # barajamos los pacientes de esta clase
        n = len(patients)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        # Primeros -> test, siguientes -> val, resto -> train.
        partitions["test"].extend(patients[:n_test])
        partitions["val"].extend(patients[n_test : n_test + n_val])
        partitions["train"].extend(patients[n_test + n_val :])

    # Finalmente, seleccionamos las filas del DataFrame original cuyos pacientes
    # pertenecen a cada particion.
    return {
        name: df[df[patient_col].isin(ids)].reset_index(drop=True)
        for name, ids in partitions.items()
    }


# ---------------------------------------------------------------------------
# Ejecucion directa: comprobacion rapida sobre el manifiesto real
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Permite ejecutar `python src/data.py` para inspeccionar el manifiesto
    # sin necesidad de tener aun descargados los CSV ni las imagenes.
    root = Path(__file__).resolve().parents[1]
    xlsx = root / "CBIS-DDSM-All-doiJNLP-zzWs5zfZ-nbia-digest.xlsx"
    manifest = load_manifest(xlsx)
    print("Filas del manifiesto:", len(manifest))
    print("Pacientes unicos:", manifest["patient"].nunique())
    print(manifest[["Patient ID", "patient"]].head())
