"""Validacion externa en CSAW-CC: analisis del plan docs/plan_validacion_csaw.md.

Uso:
    python scripts/csaw_lesion_site.py

Requiere haber ejecutado antes scripts/download_csaw.py. Reutiliza sin
cambios la medida de EMBED (scripts/embed_lesion_site.py): ventana de
512x512 px a resolucion completa, 256x256, CLAHE, homologia cubica e imagen
de persistencia 6x6 por dimension.

Localizacion del sitio (plan, reglas 1-2): en t-1, el punto/mascara del
radiologo si existe; si no, el centroide de la mascara de t trasladado con
la misma posicion relativa en la caja de la mama. En t, al reves. Los
controles usan la posicion relativa del sitio de su caso.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.preprocessing import read_dicom_gray  # noqa: E402
from src.topology import persistence_image_vector, topological_drift  # noqa: E402
import embed_lesion_site as ls  # noqa: E402

CSAW_DIR = ROOT / "data" / "csaw"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = CSAW_DIR / "csaw_site_cache.pkl"
HALF = 256
N_PERM_PRINCIPAL = 1000
N_PERM_SECUNDARIOS = 200


def _rel(point, bbox):
    r0, c0, r1, c1 = bbox
    return (point[0] - r0) / max(r1 - r0, 1), (point[1] - c0) / max(c1 - c0, 1)


def _abs(rel, bbox):
    r0, c0, r1, c1 = bbox
    return r0 + rel[0] * (r1 - r0), c0 + rel[1] * (c1 - c0)


def case_relative_sites(path_t0: Path, path_t1: Path, roi0, roi1) -> tuple[tuple, tuple]:
    """Posicion relativa (en la caja de la mama) del sitio en t-1 y en t de
    un caso, segun las reglas 1-2 del plan. ``roi0``/``roi1`` son el
    centroide de la mascara en t-1/t o None."""
    b0 = ls._full_bbox(read_dicom_gray(path_t0))
    b1 = ls._full_bbox(read_dicom_gray(path_t1))
    rel0 = _rel(roi0, b0) if roi0 is not None else _rel(roi1, b1)
    rel1 = _rel(roi1, b1) if roi1 is not None else _rel(roi0, b0)
    return rel0, rel1


def site_features(path_t0: Path, path_t1: Path, rel0, rel1, half: int = HALF, shift: float = 0.0) -> dict:
    """Ventanas en t-1 y t en las posiciones relativas dadas (opcionalmente
    desplazadas ``shift`` x la altura de la mama: control de localizacion)."""
    a0, a1 = read_dicom_gray(path_t0), read_dicom_gray(path_t1)
    b0, b1 = ls._full_bbox(a0), ls._full_bbox(a1)
    if shift:
        rel0 = ((rel0[0] + shift) % 1.0, rel0[1])
        rel1 = ((rel1[0] + shift) % 1.0, rel1[1])
    d0 = ls._window_diagram(a0, *_abs(rel0, b0), half)
    d1 = ls._window_diagram(a1, *_abs(rel1, b1), half)
    return {"pi_t0": persistence_image_vector(d0, n_bins=ls.N_BINS)[0],
            "pi_t1": persistence_image_vector(d1, n_bins=ls.N_BINS)[0],
            "drift": np.asarray(topological_drift(d0, d1, max_points=100))}


def _roi(r, which):
    v = (getattr(r, f"roi{which}_r"), getattr(r, f"roi{which}_c"))
    return None if pd.isna(v[0]) else v


def compute(manifest: pd.DataFrame, shift: float = 0.0, verbose: bool = True):
    """Features de toda la cohorte. Devuelve (feats, y, grupos, sets)."""
    from joblib import Parallel, delayed

    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    cases = manifest[manifest["kind"] == "incidente"]

    rel_key = lambda r: ("rel", r.path_t0, r.path_t1)
    pending = [r for r in cases.itertuples() if rel_key(r) not in cache]
    out = Parallel(n_jobs=-1)(delayed(case_relative_sites)(
        ROOT / r.path_t0, ROOT / r.path_t1, _roi(r, 0), _roi(r, 1)) for r in pending)
    cache.update({rel_key(r): o for r, o in zip(pending, out)})
    CACHE_PATH.write_bytes(pickle.dumps(cache))
    rels = {r.set_id: cache[rel_key(r)] for r in cases.itertuples()}

    feat_key = lambda r: ("feat", r.path_t0, r.path_t1, r.set_id, HALF, ls.WINDOW_SIZE, ls.N_BINS, shift)
    rows = list(manifest.itertuples())
    pending = [r for r in rows if feat_key(r) not in cache]
    for start in range(0, len(pending), 64):
        chunk = pending[start:start + 64]
        res = Parallel(n_jobs=-1)(delayed(site_features)(
            ROOT / r.path_t0, ROOT / r.path_t1, *rels[r.set_id], HALF, shift) for r in chunk)
        cache.update({feat_key(r): f for r, f in zip(chunk, res)})
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        if verbose:
            print(f"  {min(start + 64, len(pending))}/{len(pending)} filas (shift={shift})", flush=True)

    feats = [cache[feat_key(r)] for r in rows]
    y = (manifest["kind"] == "incidente").astype(int).values
    case_pid = cases.set_index("set_id")["pid"].to_dict()
    groups = np.array([case_pid[s] for s in manifest["set_id"]])
    sets = manifest["set_id"].values
    return feats, y, groups, sets


def conditional_or(X, y, groups, sets) -> tuple[float, float]:
    """OR por desviacion tipica del score fuera de fold, regresion logistica
    condicional con estratos = conjunto caso + controles."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from statsmodels.discrete.conditional_models import ConditionalLogit

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    score = cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    z = (score - score.mean()) / score.std()
    r = ConditionalLogit(y, z[:, None], groups=sets).fit(disp=0)
    return float(np.exp(r.params[0])), float(r.pvalues[0])


def main() -> None:
    manifest = pd.read_csv(CSAW_DIR / "manifest_csaw.csv")
    feats, y, groups, sets = compute(manifest)
    stack = lambda key, fs=feats: np.stack([f[key] for f in fs])
    Xa, Xt, Xd = stack("pi_t0"), stack("pi_t1"), stack("drift")
    n_cancer = manifest.loc[y == 1, "pid"].nunique()

    L = [f"VALIDACION EXTERNA EN CSAW-CC (plan: docs/plan_validacion_csaw.md)",
         f"{int(y.sum())} vistas de {n_cancer} canceres, {int((1 - y).sum())} controles; "
         f"ventana {2 * HALF} px; CV agrupada por paciente del caso", ""]

    auc, p = ls.permutation_auc(Xa, y, groups, sets, n=N_PERM_PRINCIPAL)
    or_, p_or = conditional_or(Xa, y, groups, sets)
    exito = auc > 0.5 and p < 0.05 and or_ > 1
    L += ["ANALISIS PRINCIPAL (anticipacion: t-1 en el sitio)",
          f"  AUC = {auc:.3f}, p (permutacion, {N_PERM_PRINCIPAL}) = {p:.4f}",
          f"  regresion logistica condicional: OR por DE = {or_:.2f}, p = {p_or:.3g}",
          f"  CRITERIO DEL PLAN (AUC > 0.5, p < 0.05, OR > 1): {'CUMPLIDO' if exito else 'NO CUMPLIDO'}", ""]

    L.append(f"SECUNDARIOS ({N_PERM_SECUNDARIOS} permutaciones)")
    a, pp = ls.permutation_auc(Xt, y, groups, sets, n=N_PERM_SECUNDARIOS)
    L.append(f"  1. control positivo (t en el sitio)      : AUC = {a:.3f}, p = {pp:.3f}")
    fs, ys, gs, ss = compute(manifest, shift=0.5)
    a, pp = ls.permutation_auc(stack("pi_t0", fs), ys, gs, ss, n=N_PERM_SECUNDARIOS)
    a2, pp2 = ls.permutation_auc(stack("pi_t1", fs), ys, gs, ss, n=N_PERM_SECUNDARIOS)
    L.append(f"  2. sitio alejado (media mama): anticipacion AUC = {a:.3f}, p = {pp:.3f}; "
             f"control positivo AUC = {a2:.3f}, p = {pp2:.3f}")
    for vw in ("MLO", "CC"):
        m = (manifest["view"] == vw).values
        a, pp = ls.permutation_auc(Xa[m], y[m], groups[m], sets[m], n=N_PERM_SECUNDARIOS)
        L.append(f"  3. solo {vw:3s}: anticipacion AUC = {a:.3f}, p = {pp:.3f}")
    half_cols = Xa.shape[1] // 2
    for nombre, cols in (("H0", slice(0, half_cols)), ("H1", slice(half_cols, None))):
        a, pp = ls.permutation_auc(Xa[:, cols], y, groups, sets, n=N_PERM_SECUNDARIOS)
        L.append(f"  4. solo {nombre}: anticipacion AUC = {a:.3f}, p = {pp:.3f}")
    marcado = manifest.groupby("set_id")["mask_t0"].transform("max").astype(bool).values
    for nombre, m in (("sitio marcado en t-1 (regla 1)", marcado), ("sitio trasladado desde t (regla 2)", ~marcado)):
        if len(np.unique(groups[m])) >= 5:
            a, pp = ls.permutation_auc(Xa[m], y[m], groups[m], sets[m], n=N_PERM_SECUNDARIOS)
            L.append(f"  5. {nombre}: anticipacion AUC = {a:.3f}, p = {pp:.3f} ({int(y[m].sum())} vistas)")
    L.append(f"  (deriva en el sitio, no prevista como secundario; exploratorio): "
             f"AUC = {ls.permutation_auc(Xd, y, groups, sets, n=0)[0]:.3f}")

    report = "\n".join(L)
    print("\n" + report)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "csaw_validacion.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\nGuardado en {RESULTS_DIR / 'csaw_validacion.txt'}")


if __name__ == "__main__":
    main()
