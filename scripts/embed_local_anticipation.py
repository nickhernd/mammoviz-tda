"""Anticipacion LOCAL: ¿habia ya señal topologica en la exploracion previa
(t-1) EN EL SITIO donde despues cambia la mama? (issue #95, O8).

Uso:
    python scripts/embed_local_anticipation.py

La anticipacion global (imagen de persistencia de la mamografia entera) da
AUC ~0.49 sobre el diseño emparejado: si hay un precursor, se diluye en la
imagen completa. Aqui se mira localmente, sin necesitar anotaciones:

  1. Cada mamografia (t-1 y t) se recorta a la mama y se divide en una
     rejilla de 4x4 parches (src/local_topology.py); se calcula el diagrama
     de persistencia de cada parche.
  2. El SITIO DEL CAMBIO de cada paciente es el parche con mayor deriva
     topologica (distancia bottleneck H0) entre t-1 y t. En un incidente
     deberia coincidir con la lesion (se valida con las ROI anotadas de
     EMBED cuando existen).
  3. Variantes de la señal de anticipacion (siempre con la imagen t-1):
       - sitio propio: el parche del sitio del cambio de cada paciente.
       - sitio del incidente emparejado: para cada control, el MISMO parche
         que el sitio de su incidente emparejado. Es la prueba mas limpia:
         la ubicacion no depende del contenido del control, asi que una
         diferencia no puede venir de "el procedimiento elige otro tipo de
         tejido en cada grupo".
       - parche aleatorio: control negativo (deberia dar azar).
     Cada variante en absoluto y en relativo (parche menos la mediana de los
     demas parches de la MISMA mamografia, lo que neutraliza diferencias de
     equipo/procesado/epoca entre pacientes).

Es una pregunta retrospectiva (el sitio se elige mirando t): no es un
predictor desplegable, es la comprobacion de si el precursor era visible en
la exploracion previa, que es lo que se pregunta en los estudios de
"canceres de intervalo".
"""
from __future__ import annotations

import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.local_topology import breast_bbox, cell_neighbours, grid_patches, point_to_cell  # noqa: E402
from src.preprocessing import apply_clahe, read_dicom_gray, resize, to_grayscale_float  # noqa: E402
from src.topology import persistence_diagram, persistence_image_vector, topological_drift  # noqa: E402

EMBED_DIR = ROOT / "data" / "embed"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = EMBED_DIR / "local_features_cache.pkl"
GRID = 4
CROP_SIZE = 512      # mama recortada -> 512x512, parches de 128x128
DOWNSAMPLE = 4       # la imagen original (~4096x3328) se reduce 4x antes de recortar
N_BINS = 6
VERSION = 1


def local_diagrams(path: Path):
    """Diagramas de persistencia de los GRID x GRID parches de la mama, y la
    caja de la mama en coordenadas de la imagen ORIGINAL (para mapear ROI)."""
    arr = read_dicom_gray(path)
    small = arr[::DOWNSAMPLE, ::DOWNSAMPLE]
    r0, c0, r1, c1 = breast_bbox(small)
    crop = resize(apply_clahe(to_grayscale_float(small[r0:r1, c0:c1])), (CROP_SIZE, CROP_SIZE))
    diagrams = [persistence_diagram(p) for p in grid_patches(crop, GRID)]
    bbox_full = (r0 * DOWNSAMPLE, c0 * DOWNSAMPLE, r1 * DOWNSAMPLE, c1 * DOWNSAMPLE)
    return diagrams, bbox_full


def pair_local_features(path_t0: Path, path_t1: Path) -> dict:
    """Por par: imagen de persistencia de cada parche de t-1 (GRID², 2·N_BINS²),
    deriva bottleneck por parche (GRID², 2) y la caja de la mama en t."""
    d0, _ = local_diagrams(path_t0)
    d1, bbox_t1 = local_diagrams(path_t1)
    pi_t0 = np.stack([persistence_image_vector(d, n_bins=N_BINS)[0] for d in d0])
    drift = np.array([topological_drift(a, b, max_points=100) for a, b in zip(d0, d1)])
    return {"pi_t0": pi_t0, "drift": drift, "bbox_t1": bbox_t1}


def compute_local_features(manifest: pd.DataFrame, verbose: bool = True, batch: int = 32) -> list[dict]:
    """Features locales de todo el manifiesto, en paralelo (~11 s/par en un
    nucleo) y con cache en disco por par, guardada tras cada lote."""
    from joblib import Parallel, delayed

    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    keys = [(r.path_t0, r.path_t1, GRID, N_BINS, VERSION) for r in manifest.itertuples()]
    pending = [k for k in dict.fromkeys(keys) if k not in cache]
    for start in range(0, len(pending), batch):
        chunk = pending[start:start + batch]
        results = Parallel(n_jobs=-1)(
            delayed(pair_local_features)(ROOT / k[0], ROOT / k[1]) for k in chunk)
        cache.update(zip(chunk, results))
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        if verbose:
            print(f"  {min(start + batch, len(pending))}/{len(pending)} pares calculados", flush=True)
    return [cache[k] for k in keys]


def anticipation_variants(manifest: pd.DataFrame, feats: list[dict], seed: int = 42) -> dict[str, np.ndarray]:
    """Matrices de features para cada variante de anticipacion local."""
    rng = np.random.default_rng(seed)
    site = np.array([int(np.argmax(f["drift"][:, 0])) for f in feats])
    # sitio del incidente emparejado (solo existe en el diseño emparejado)
    site_of = dict(zip(manifest["empi_anon"], site))
    paired_site = site.copy()
    if "matched_to" in manifest.columns:
        for k, row in enumerate(manifest.itertuples()):
            if row.kind == "control" and pd.notna(row.matched_to):
                paired_site[k] = site_of[int(row.matched_to)]
    random_site = rng.integers(0, GRID * GRID, size=len(feats))

    def _abs(sites):
        return np.stack([f["pi_t0"][s] for f, s in zip(feats, sites)])

    def _rel(sites):
        return np.stack([
            f["pi_t0"][s] - np.median(np.delete(f["pi_t0"], s, axis=0), axis=0)
            for f, s in zip(feats, sites)
        ])

    return {
        "sitio propio (abs)": _abs(site),
        "sitio propio (rel)": _rel(site),
        "sitio del incidente emparejado (abs)": _abs(paired_site),
        "sitio del incidente emparejado (rel)": _rel(paired_site),
        "parche aleatorio (abs)": _abs(random_site),
        "parche aleatorio (rel)": _rel(random_site),
        "_site": site,
    }


def roi_validation(manifest: pd.DataFrame, feats: list[dict]) -> dict:
    """¿Cae el sitio del cambio sobre la lesion? Solo para los incidentes
    cuya imagen t descargada tiene ROI anotada en EMBED (ymin, xmin, ymax,
    xmax en coordenadas de la imagen original)."""
    meta = pd.read_csv(
        EMBED_DIR / "metadata_reduced.csv",
        usecols=["acc_anon", "FinalImageType", "ImageLateralityFinal", "ViewPosition", "num_ROI", "ROI_coords"],
    )
    meta = meta[meta["FinalImageType"] == "2D"].assign(
        view=lambda d: d["ImageLateralityFinal"] + "_" + d["ViewPosition"])
    exact = near = total = 0
    for row, f in zip(manifest.itertuples(), feats):
        if row.kind != "incidente":
            continue
        imgs = meta[(meta["acc_anon"] == row.acc_t1) & (meta["view"] == row.view)]
        if imgs.empty or imgs.iloc[0]["num_ROI"] <= 0:
            continue  # la imagen descargada es la primera de esa vista
        ymin, xmin, ymax, xmax = ast.literal_eval(imgs.iloc[0]["ROI_coords"])[0]
        roi_cell = point_to_cell((ymin + ymax) / 2, (xmin + xmax) / 2, f["bbox_t1"], GRID)
        change_cell = int(np.argmax(f["drift"][:, 0]))
        total += 1
        exact += change_cell == roi_cell
        near += change_cell in cell_neighbours(roi_cell, GRID)
    n_cells = GRID * GRID
    chance_near = np.mean([len(cell_neighbours(c, GRID)) for c in range(n_cells)]) / n_cells
    return {"n": total, "exact": exact, "near": near,
            "chance_exact": 1 / n_cells, "chance_near": chance_near}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="longitudinal_pilot_manifest_matched.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(EMBED_DIR / args.manifest)
    y = (manifest["kind"] == "incidente").astype(int).values
    print(f"{args.manifest}: {len(manifest)} pacientes ({y.sum()} incidentes, {(1 - y).sum()} controles)")

    feats = compute_local_features(manifest)
    variants = anticipation_variants(manifest, feats, seed=args.seed)

    lines = [f"Anticipacion LOCAL ({args.manifest}, {len(manifest)} pacientes, rejilla {GRID}x{GRID})", ""]
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=args.seed, class_weight="balanced")
    for name, X in variants.items():
        if name.startswith("_"):
            continue
        s = cross_val_score(rf(), X, y, cv=5, scoring="roc_auc")
        lines.append(f"  {name:40s}: AUC (5-fold CV) = {s.mean():.3f} +/- {s.std():.3f}")
    drift_local = np.stack([np.concatenate([f["drift"].max(axis=0), f["drift"].mean(axis=0)]) for f in feats])
    s = cross_val_score(rf(), drift_local, y, cv=5, scoring="roc_auc")
    lines.append(f"  {'deriva local (max y media por parche)':40s}: AUC (5-fold CV) = {s.mean():.3f} +/- {s.std():.3f}")

    v = roi_validation(manifest, feats)
    lines += ["", f"Validacion con ROI anotada (incidentes con ROI en la vista usada: n={v['n']}):",
              f"  sitio del cambio = celda de la lesion:          {v['exact']}/{v['n']} "
              f"(azar {v['chance_exact']:.0%})",
              f"  sitio del cambio en la vecindad 3x3 de la lesion: {v['near']}/{v['n']} "
              f"(azar {v['chance_near']:.0%})"]

    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"embed_local_anticipation_{Path(args.manifest).stem}.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
