"""Topologia EN EL SITIO DE LA LESION (cohorte de cribado, issue #95, O8).

Uso:
    python scripts/embed_lesion_site.py [--manifest longitudinal_manifest_cribado.csv]

Sobre la cohorte de cribado, el descriptor global (mamografia entera a
224x224) no separa casos de controles ni siquiera en t, cuando el cancer ya
esta presente (AUC ~0.53): a esa escala una lesion de cribado ocupa unos
pocos pixeles. Un resultado nulo de anticipacion con un descriptor que no ve
la lesion no dice nada sobre el precursor.

Aqui se mira donde esta la lesion, a resolucion completa:

  1. Solo los incidentes cuya imagen t tiene ROI anotada en EMBED, y su
     control emparejado.
  2. Centro de la ROI en t -> posicion RELATIVA dentro de la caja de la mama.
  3. Esa misma posicion relativa se lleva a las otras tres imagenes del
     par emparejado: t-1 del incidente, t y t-1 del control. La ubicacion la
     fija la anotacion del incidente y se aplica igual a su control, asi que
     no hay sesgo de seleccion del sitio (el problema de la variante "sitio
     propio" de embed_local_anticipation.py).
  4. Ventana de 2*HALF px (~3.3 cm con pixel de 0.065 mm), reducida a la
     mitad, CLAHE y homologia cubica.

Señales:
  - control positivo (t):   ventana en la lesion en t, incidente vs control.
                            Si esto no separa, el descriptor no ve la lesion.
  - anticipacion (t-1):     la misma ventana en t-1. La pregunta del TFG.
  - deriva en el sitio:     bottleneck H0/H1 entre las ventanas de t-1 y t.

La validacion cruzada agrupa por par emparejado (GroupKFold): un incidente y
su control nunca quedan uno en entrenamiento y otro en test.
"""
from __future__ import annotations

import ast
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_topology import breast_bbox, map_point, window_around  # noqa: E402
from src.preprocessing import apply_clahe, read_dicom_gray, resize, to_grayscale_float  # noqa: E402
from src.topology import persistence_diagram, persistence_image_vector, topological_drift  # noqa: E402

EMBED_DIR = ROOT / "data" / "embed"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = EMBED_DIR / "lesion_site_cache.pkl"
HALF = 256          # ventana de 512x512 px a resolucion completa
SHIFT = 0.0         # 0 = sitio de la lesion; 0.5 = sitio alejado (control de localizacion)
WINDOW_SIZE = 256   # reducida a 256x256 antes de la homologia
BBOX_DOWNSAMPLE = 4
N_BINS = 6
VERSION = 1


def _full_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = breast_bbox(arr[::BBOX_DOWNSAMPLE, ::BBOX_DOWNSAMPLE])
    return r0 * BBOX_DOWNSAMPLE, c0 * BBOX_DOWNSAMPLE, r1 * BBOX_DOWNSAMPLE, c1 * BBOX_DOWNSAMPLE


def _window_diagram(arr: np.ndarray, row: float, col: float, half: int):
    win = window_around(arr, row, col, half)
    return persistence_diagram(resize(apply_clahe(to_grayscale_float(win)), (WINDOW_SIZE, WINDOW_SIZE)))


def lesion_rois(manifest: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Centro (fila, columna) de la primera ROI de la imagen t de cada
    incidente que la tenga, en coordenadas de la imagen original."""
    meta = pd.read_csv(
        EMBED_DIR / "metadata_reduced.csv",
        usecols=["acc_anon", "FinalImageType", "ImageLateralityFinal", "ViewPosition", "num_ROI", "ROI_coords"],
    )
    meta = meta[meta["FinalImageType"] == "2D"].assign(
        view=lambda d: d["ImageLateralityFinal"] + "_" + d["ViewPosition"])
    rois = {}
    for row in manifest[manifest["kind"] == "incidente"].itertuples():
        imgs = meta[(meta["acc_anon"] == row.acc_t1) & (meta["view"] == row.view)]
        if imgs.empty or imgs.iloc[0]["num_ROI"] <= 0:
            continue
        ymin, xmin, ymax, xmax = ast.literal_eval(imgs.iloc[0]["ROI_coords"])[0]
        rois[int(row.empi_anon)] = ((ymin + ymax) / 2, (xmin + xmax) / 2)
    return rois


def shift_point(row: float, col: float, bbox: tuple[int, int, int, int], shift: float) -> tuple[float, float]:
    """Desplaza un punto ``shift`` veces la altura de la caja de la mama en
    vertical, dando la vuelta dentro de la caja (misma profundidad, otra zona
    de la mama). Con shift=0.5 se obtiene un sitio alejado de la lesion."""
    r0, _, r1, _ = bbox
    h = max(r1 - r0, 1)
    return r0 + (((row - r0) / h + shift) % 1.0) * h, col


def pair_site_features(path_t0: Path, path_t1: Path, roi_t1_ref: tuple[float, float],
                       bbox_ref: tuple[int, int, int, int] | None, half: int = HALF,
                       shift: float = 0.0) -> dict:
    """Ventanas en el sitio de la lesion de t-1 y t de UNA paciente.

    ``roi_t1_ref`` es el centro de la lesion en la imagen t del incidente y
    ``bbox_ref`` la caja de la mama de esa imagen (None si la paciente es el
    propio incidente: entonces la referencia es su propia t). Con ``shift``
    distinto de 0 la ventana se mueve a un sitio ALEJADO de la lesion
    (control negativo de localizacion), igual en casos y controles."""
    a0, a1 = read_dicom_gray(path_t0), read_dicom_gray(path_t1)
    b0, b1 = _full_bbox(a0), _full_bbox(a1)
    ref_bbox = b1 if bbox_ref is None else bbox_ref
    if shift:
        roi_t1_ref = shift_point(*roi_t1_ref, ref_bbox, shift)
    r1, c1 = map_point(*roi_t1_ref, ref_bbox, b1)
    r0, c0 = map_point(*roi_t1_ref, ref_bbox, b0)
    d0, d1 = _window_diagram(a0, r0, c0, half), _window_diagram(a1, r1, c1, half)
    return {
        "pi_t0": persistence_image_vector(d0, n_bins=N_BINS)[0],
        "pi_t1": persistence_image_vector(d1, n_bins=N_BINS)[0],
        "drift": np.asarray(topological_drift(d0, d1, max_points=100)),
        "bbox_t1": b1,
    }


def compute_site_features(manifest: pd.DataFrame, verbose: bool = True):
    """Features en el sitio de la lesion para cada incidente con ROI y su
    control. Devuelve (filas del manifiesto usadas, lista de dicts, grupos)."""
    from joblib import Parallel, delayed

    rois = lesion_rois(manifest)
    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    incidents = manifest[(manifest["kind"] == "incidente") & manifest["empi_anon"].isin(rois)]
    controls = manifest[manifest["kind"] == "control"].set_index("matched_to")

    # 1) incidentes: la referencia es su propia t (hace falta su caja de mama)
    def _key(r, ref):
        return (r.path_t0, r.path_t1, ref, HALF, WINDOW_SIZE, N_BINS, VERSION)

    inc_rows = list(incidents.itertuples())
    pending = [r for r in inc_rows if _key(r, r.empi_anon) not in cache]
    out = Parallel(n_jobs=-1)(delayed(pair_site_features)(
        ROOT / r.path_t0, ROOT / r.path_t1, rois[int(r.empi_anon)], None, HALF) for r in pending)
    cache.update({_key(r, r.empi_anon): f for r, f in zip(pending, out)})
    CACHE_PATH.write_bytes(pickle.dumps(cache))
    if verbose:
        print(f"  incidentes: {len(pending)} calculados, {len(inc_rows) - len(pending)} desde cache", flush=True)

    # 2) controles: la referencia es la t de SU incidente emparejado
    ctl_rows = []
    for r in inc_rows:
        if float(r.empi_anon) in controls.index:
            c = controls.loc[float(r.empi_anon)]
            ctl_rows.append((r, c))
    pending = [(r, c) for r, c in ctl_rows if _key(c, r.empi_anon) not in cache]
    out = Parallel(n_jobs=-1)(delayed(pair_site_features)(
        ROOT / c.path_t0, ROOT / c.path_t1, rois[int(r.empi_anon)],
        cache[_key(r, r.empi_anon)]["bbox_t1"], HALF) for r, c in pending)
    cache.update({_key(c, r.empi_anon): f for (r, c), f in zip(pending, out)})
    CACHE_PATH.write_bytes(pickle.dumps(cache))
    if verbose:
        print(f"  controles: {len(pending)} calculados, {len(ctl_rows) - len(pending)} desde cache", flush=True)

    feats, labels, groups = [], [], []
    for g, (r, c) in enumerate(ctl_rows):
        feats += [cache[_key(r, r.empi_anon)], cache[_key(c, r.empi_anon)]]
        labels += [1, 0]
        groups += [g, g]
    return feats, np.asarray(labels), np.asarray(groups)


def compute_lesion_cohort_features(manifest: pd.DataFrame, verbose: bool = True):
    """Lo mismo que ``compute_site_features`` para el manifiesto de la
    cohorte de lesion (``--lesion``): una fila por (caso, vista) con la ROI
    en ``roi_r``/``roi_c`` y varios controles por ``set_id``. El grupo de la
    validacion cruzada es la PACIENTE del caso, para que sus dos vistas (y
    sus controles) caigan siempre en el mismo fold.

    Devuelve (lista de dicts, etiquetas, grupos, set_ids)."""
    from joblib import Parallel, delayed

    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}

    def _key(r):
        base = (r.path_t0, r.path_t1, r.set_id, HALF, WINDOW_SIZE, N_BINS, VERSION)
        return base + (("shift", SHIFT),) if SHIFT else base

    case_rows = {r.set_id: r for r in manifest[manifest["kind"] == "incidente"].itertuples()}
    pending = [r for r in case_rows.values() if _key(r) not in cache]
    out = Parallel(n_jobs=-1)(delayed(pair_site_features)(
        ROOT / r.path_t0, ROOT / r.path_t1, (r.roi_r, r.roi_c), None, HALF, SHIFT) for r in pending)
    cache.update({_key(r): f for r, f in zip(pending, out)})
    CACHE_PATH.write_bytes(pickle.dumps(cache))
    if verbose:
        print(f"  casos: {len(pending)} calculados", flush=True)

    ctl_rows = list(manifest[manifest["kind"] == "control"].itertuples())
    pending = [r for r in ctl_rows if _key(r) not in cache]
    for start in range(0, len(pending), 64):
        chunk = pending[start:start + 64]
        res = Parallel(n_jobs=-1)(delayed(pair_site_features)(
            ROOT / r.path_t0, ROOT / r.path_t1,
            (case_rows[r.set_id].roi_r, case_rows[r.set_id].roi_c),
            cache[_key(case_rows[r.set_id])]["bbox_t1"], HALF, SHIFT) for r in chunk)
        cache.update({_key(r): f for r, f in zip(chunk, res)})
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        if verbose:
            print(f"  controles: {min(start + 64, len(pending))}/{len(pending)} calculados", flush=True)

    rows = list(manifest.itertuples())
    feats = [cache[_key(r)] for r in rows]
    labels = np.array([1 if r.kind == "incidente" else 0 for r in rows])
    groups = np.array([case_rows[r.set_id].empi_anon for r in rows])
    sets = np.array([r.set_id for r in rows])
    return feats, labels, groups, sets


def permutation_auc(X, y, groups, sets, n: int = 100, seed: int = 0, n_estimators: int = 200):
    """AUC por CV agrupada (5 folds) y p-valor de una prueba de permutacion
    que, dentro de cada conjunto emparejado (``sets``: un caso y sus
    controles), elige al azar cual de sus miembros hace de caso."""
    rng = np.random.default_rng(seed)
    cv = GroupKFold(n_splits=5)
    rf = lambda: RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                        class_weight="balanced", n_jobs=-1)
    obs = cross_val_score(rf(), X, y, groups=groups, cv=cv, scoring="roc_auc").mean()
    idx_by_set = [np.where(sets == s)[0] for s in np.unique(sets)]
    null = []
    for _ in range(n):
        yp = np.zeros_like(y)
        for idx in idx_by_set:
            yp[rng.choice(idx)] = 1
        null.append(cross_val_score(rf(), X, yp, groups=groups, cv=cv, scoring="roc_auc").mean())
    return obs, (1 + np.sum(np.array(null) >= obs)) / (1 + n)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="longitudinal_manifest_cribado.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half", type=int, default=HALF,
                        help="Semilado de la ventana en px (analisis de sensibilidad al error de alineacion).")
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--shift", type=float, default=0.0,
                        help="Desplaza la ventana esta fraccion de la altura de la mama (0.5 = sitio alejado "
                             "de la lesion: control negativo de localizacion; solo cohorte de lesion).")
    args = parser.parse_args()
    globals()["HALF"] = args.half
    globals()["SHIFT"] = args.shift

    manifest = pd.read_csv(EMBED_DIR / args.manifest)
    if "set_id" in manifest.columns:
        feats, y, groups, sets = compute_lesion_cohort_features(manifest)
        desc = (f"{int(y.sum())} vistas con lesion de {manifest.loc[y == 1, 'empi_anon'].nunique()} "
                f"canceres, {int((1 - y).sum())} controles")
    else:
        feats, y, groups = compute_site_features(manifest)
        sets = groups
        desc = f"{len(y) // 2} pares incidente-control (incidentes con ROI anotada en t)"

    lines = [f"Topologia en el sitio de la lesion ({args.manifest}): {desc}",
             f"Ventana {2 * HALF}x{2 * HALF} px a resolucion completa -> {WINDOW_SIZE}x{WINDOW_SIZE}; "
             f"CV agrupada por paciente del caso; p = permutacion dentro de cada conjunto emparejado "
             f"({args.permutations} permutaciones)"]
    if SHIFT:
        lines.append(f"SITIO ALEJADO: ventana desplazada {SHIFT:.2f} x la altura de la mama "
                     f"(control negativo de localizacion)")
    lines.append("")
    señales = {
        "control positivo: t en la lesion": np.stack([f["pi_t1"] for f in feats]),
        "anticipacion: t-1 en el sitio": np.stack([f["pi_t0"] for f in feats]),
        "deriva en el sitio (t-1 vs t)": np.stack([f["drift"] for f in feats]),
    }
    for name, X in señales.items():
        auc, p = permutation_auc(X, y, groups, sets, n=args.permutations)
        lines.append(f"  {name:36s}: AUC (5-fold CV) = {auc:.3f}, p = {p:.3f}")

    # ¿que dimension de homologia lleva la anticipacion? (PI = [H0 | H1])
    Xa = señales["anticipacion: t-1 en el sitio"]
    mitad = Xa.shape[1] // 2
    cv = GroupKFold(n_splits=5)
    lines.append("")
    for name, cols in [("anticipacion solo H0", slice(0, mitad)), ("anticipacion solo H1", slice(mitad, None))]:
        s = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced",
                                                   n_jobs=-1), Xa[:, cols], y, groups=groups, cv=cv,
                            scoring="roc_auc")
        lines.append(f"  {name:36s}: AUC (5-fold CV) = {s.mean():.3f} +/- {s.std():.3f}")

    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = Path(args.manifest).stem.replace("longitudinal_manifest_", "")
    suffix = "" if args.half == 256 else f"_ventana{2 * args.half}"
    suffix += f"_alejado{args.shift:g}" if args.shift else ""
    out = RESULTS_DIR / f"embed_lesion_site_{stem}{suffix}.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
