"""Descarga incremental de imagenes DICOM reales de CBIS-DDSM desde TCIA.

Este script resuelve el problema de que CBIS-DDSM (163.5 GB, 6775 series) es
demasiado grande para descargar entero de golpe: en su lugar, descarga un
PILOTO balanceado (benigno/maligno) de N pacientes nuevos cada vez que se
ejecuta, y se puede volver a lanzar mas adelante para ampliarlo sin repetir
descargas ya hechas (issue #53: piloto de F1; issue #82: ampliacion en F3).

Como funciona:
  1. Los CSV de descripcion de casos (data/csv/*.csv) tienen las etiquetas
     benigno/maligno, pero sus columnas de ruta ("image file path", etc.)
     contienen UIDs OBSOLETOS que ya no existen en el TCIA actual (problema
     conocido de CBIS-DDSM). Lo unico fiable de esas rutas es el PRIMER
     segmento (p.ej. "Mass-Training_P_00001_LEFT_CC_1"), que SI coincide con
     el campo "PatientID" que devuelve la API de TCIA para cada serie DICOM.
  2. Se descarga (y cachea localmente) el indice completo de series de la
     coleccion via la API REST v1 de TCIA (getSeries), y se construye un
     diccionario {PatientID-carpeta -> series DICOM disponibles}.
  3. Para cada caso del CSV se buscan sus 2-3 series necesarias: la mamografia
     completa ("full mammogram images"), el recorte de la lesion
     ("cropped images") y la mascara de ROI ("ROI mask images").
  4. Cada serie se descarga como ZIP (endpoint getImage) y se descomprime en
     data/images/<PatientID-carpeta>/.

Uso:
    python scripts/download_cbis_ddsm.py --add 15 15
    python scripts/download_cbis_ddsm.py --add 20 20 --category mass
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import case_folder_key, load_case_descriptions  # noqa: E402

CSV_DIR = ROOT / "data" / "csv"
IMAGES_DIR = ROOT / "data" / "images"
SERIES_INDEX_CACHE = IMAGES_DIR / "_series_index.json"

API_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "CBIS-DDSM"


def fetch_series_index(force_refresh: bool = False) -> dict[str, list[dict]]:
    """Descarga (o carga de cache local) el indice PatientID -> series NBIA.

    El indice completo (6775 series) tarda unos segundos en descargar y no
    cambia entre ejecuciones, asi que se cachea en disco para no repetir la
    llamada cada vez que se amplia el piloto.
    """
    if SERIES_INDEX_CACHE.exists() and not force_refresh:
        return json.loads(SERIES_INDEX_CACHE.read_text())

    resp = requests.get(f"{API_BASE}/getSeries", params={"Collection": COLLECTION}, timeout=120)
    resp.raise_for_status()
    series = resp.json()

    index: dict[str, list[dict]] = {}
    for s in series:
        index.setdefault(s["PatientID"], []).append(s)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SERIES_INDEX_CACHE.write_text(json.dumps(index))
    return index


def _folder_key(prefix: str, row: pd.Series, with_abnormality: bool) -> str:
    """Reconstruye el nombre de carpeta NBIA a partir de una fila del CSV."""
    return case_folder_key(
        prefix,
        row["patient_id"],
        row["left_or_right_breast"],
        row["image_view"],
        abnormality_id=row["abnormality_id"] if with_abnormality else None,
    )


def _download_series(series_uid: str, dest_dir: Path, retries: int = 3) -> int:
    """Descarga una serie DICOM (ZIP) y la descomprime en dest_dir.

    Devuelve el numero de bytes descargados. Si dest_dir ya existe y tiene
    contenido, no vuelve a descargar (soporta relanzar el script sin repetir
    trabajo). La API de TCIA corta conexiones de vez en cuando en series
    grandes (mamografias completas): reintenta con backoff antes de rendirse,
    para que un timeout puntual no tire abajo una descarga de 30 casos.
    """
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return 0

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{API_BASE}/getImage", params={"SeriesInstanceUID": series_uid}, timeout=(15, 90))
            resp.raise_for_status()
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(dest_dir)
            return len(resp.content)
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo descargar la serie {series_uid} tras {retries} intentos: {last_error}")


def download_case(row: pd.Series, prefix: str, index: dict[str, list[dict]]) -> dict:
    """Descarga las series de un caso: mamografia completa + recorte/mascara.

    IMPORTANTE (particularidad real de CBIS-DDSM, verificada sobre el indice
    de TCIA): en ~97% de los casos, el recorte de la lesion y la mascara de
    ROI NO son series separadas, sino DOS INSTANCIAS DICOM dentro de la MISMA
    serie "ROI mask images" (una es la mascara binaria, la otra el recorte en
    escala de grises; no hay garantia de orden). Solo ~3% de los casos las
    tienen como series distintas ("cropped images" + "ROI mask images"). Por
    eso aqui se descargan TODAS las series encontradas bajo la carpeta de la
    lesion, sin asumir una descripcion concreta; la distincion mascara vs.
    recorte se resuelve despues, al cargar los DICOM (ver scripts/run_pipeline.py).

    Devuelve un resumen {"patient_id", "ok": bool, "bytes": int, "missing": [...]}.
    """
    full_key = _folder_key(prefix, row, with_abnormality=False)
    lesion_key = _folder_key(prefix, row, with_abnormality=True)

    missing = []
    total_bytes = 0

    full_entries = index.get(full_key)
    if not full_entries:
        missing.append(full_key)
    else:
        series = _series_by_description(full_entries, "full mammogram images") or full_entries[0]
        total_bytes += _download_series(series["SeriesInstanceUID"], IMAGES_DIR / full_key)

    lesion_entries = index.get(lesion_key)
    if not lesion_entries:
        missing.append(lesion_key)
    else:
        for series in lesion_entries:
            sub_dir = IMAGES_DIR / lesion_key / series["SeriesDescription"].replace(" ", "_")
            total_bytes += _download_series(series["SeriesInstanceUID"], sub_dir)

    return {"patient_id": row["patient_id"], "ok": not missing, "bytes": total_bytes, "missing": missing}


def _series_by_description(entries: list[dict], description: str) -> dict | None:
    return next((e for e in entries if e["SeriesDescription"] == description), None)


def _load_cases(category: str) -> pd.DataFrame:
    """Carga los CSV reales (con la columna ``_prefix`` de src.data) y filtra
    por categoria (mass/calc/both) segun ese mismo prefijo."""
    df = load_case_descriptions(CSV_DIR)
    if category == "both":
        return df
    wanted = "Mass" if category == "mass" else "Calc"
    return df[df["_prefix"].str.startswith(wanted)].reset_index(drop=True)


_PATIENT_IN_FOLDER_RE = re.compile(r"(P_\d+)")


def already_downloaded_patients() -> set[str]:
    """Pacientes cuyas imagenes ya estan presentes en disco (para no repetir).

    p.ej. "Mass-Training_P_00001_LEFT_CC" -> "P_00001". OJO: un split("_")
    ingenuo rompe "P_00001" en dos trozos ("P", "00001"); hay que capturar el
    patron completo con regex.
    """
    if not IMAGES_DIR.exists():
        return set()
    downloaded = set()
    for entry in IMAGES_DIR.iterdir():
        if entry.is_dir() and entry.name.startswith(("Mass-", "Calc-")):
            match = _PATIENT_IN_FOLDER_RE.search(entry.name)
            if match:
                downloaded.add(match.group(1))
    return downloaded


def select_new_cases(df: pd.DataFrame, n_benign: int, n_malignant: int, seed: int) -> pd.DataFrame:
    """Elige N pacientes nuevos por clase, evitando los ya descargados."""
    done = already_downloaded_patients()
    df = df[~df["patient_id"].isin(done)]

    # Un paciente por caso (evita elegir 2 vistas del mismo paciente en la
    # misma tanda; split_by_patient ya se encarga de que no se mezclen fases).
    one_per_patient = df.drop_duplicates(subset="patient_id")

    benign_mask = one_per_patient["pathology"].str.upper().str.startswith("BENIGN")
    malignant_mask = one_per_patient["pathology"].str.upper() == "MALIGNANT"

    picked_benign = one_per_patient[benign_mask].sample(
        n=min(n_benign, benign_mask.sum()), random_state=seed
    )
    picked_malignant = one_per_patient[malignant_mask].sample(
        n=min(n_malignant, malignant_mask.sum()), random_state=seed
    )
    picked_patients = pd.concat([picked_benign, picked_malignant])["patient_id"]

    # Recuperamos TODAS las vistas (CC/MLO, izq/dcha) de los pacientes elegidos.
    return df[df["patient_id"].isin(picked_patients)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", nargs=2, type=int, metavar=("N_BENIGN", "N_MALIGNANT"), default=[15, 15],
                         help="Pacientes NUEVOS a anadir por clase en esta ejecucion (por defecto 15 15).")
    parser.add_argument("--category", choices=["mass", "calc", "both"], default="mass",
                         help="Tipo de caso a descargar (por defecto 'mass', el piloto recomendado en #53).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-index", action="store_true",
                         help="Forzar re-descarga del indice de series de TCIA (normalmente no hace falta).")
    args = parser.parse_args()

    print(f"Cargando indice de series de {COLLECTION} (cache: {SERIES_INDEX_CACHE.exists()})...")
    index = fetch_series_index(force_refresh=args.refresh_index)
    print(f"  {len(index)} claves de paciente/vista indexadas.")

    df = _load_cases(args.category)
    n_benign, n_malignant = args.add
    cases = select_new_cases(df, n_benign, n_malignant, args.seed)
    print(f"Descargando {cases['patient_id'].nunique()} pacientes nuevos "
          f"({len(cases)} vistas/casos) de categoria '{args.category}'...")

    total_bytes = 0
    failures = []
    for i, (_, row) in enumerate(cases.iterrows(), start=1):
        prefix = row["_prefix"]
        try:
            result = download_case(row, prefix, index)
        except RuntimeError as exc:
            # Un fallo de red persistente en un caso no debe tirar abajo el
            # resto del lote: se registra y se continua con el siguiente.
            print(f"  [{i}/{len(cases)}] {row['patient_id']} ({row['left_or_right_breast']} {row['image_view']}): ERROR ({exc})", flush=True)
            failures.append({"patient_id": row["patient_id"], "missing": [str(exc)]})
            continue
        total_bytes += result["bytes"]
        status = "OK" if result["ok"] else f"INCOMPLETO (falta: {result['missing']})"
        print(f"  [{i}/{len(cases)}] {result['patient_id']} ({row['left_or_right_breast']} {row['image_view']}): {status}", flush=True)
        if not result["ok"]:
            failures.append(result)

    print()
    print(f"Descarga completada: {total_bytes / 1e6:.1f} MB nuevos.")
    print(f"Total de pacientes ya en disco: {len(already_downloaded_patients())}.")
    if failures:
        print(f"AVISO: {len(failures)} casos con series incompletas (revisar nombres de carpeta en TCIA).")


if __name__ == "__main__":
    main()
