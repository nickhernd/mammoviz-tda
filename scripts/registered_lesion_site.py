"""Anticipacion en el sitio de la lesion con REGISTRO t -> t-1 (mejora 1).

Uso:
    python scripts/registered_lesion_site.py --dataset embed [--metodo elastico]
    python scripts/registered_lesion_site.py --dataset csaw  [--metodo elastico]

Igual que el analisis de la cohorte de lesion, salvo como se coloca la
ventana en t-1 de cada caso:

- si el radiologo marco el sitio en t-1 (solo CSAW-CC), se usa esa marca;
- si no, la marca de t se traslada a t-1 con src/registration.py
  (``--metodo``: caja | afin | elastico), en vez de con la caja de la mama.

Los controles reciben la posicion relativa (en su caja de la mama) del
sitio de su caso en t-1 y en t, como hasta ahora: el registro solo tiene
sentido entre dos exploraciones de la MISMA mujer.

El metodo por defecto (elastico) se eligio por el error geometrico en
results/registro_csaw.txt, no por ningun AUC.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.preprocessing import read_dicom_gray  # noqa: E402
from src.registration import map_point_registered  # noqa: E402
import csaw_lesion_site as cs  # noqa: E402
import embed_lesion_site as ls  # noqa: E402

RESULTS_DIR = ROOT / "results"
DATASETS = {
    "embed": (ROOT / "data" / "embed" / "longitudinal_manifest_lesion_edad.csv",
              ROOT / "data" / "embed" / "registered_site_cache.pkl"),
    "csaw": (ROOT / "data" / "csaw" / "manifest_csaw.csv",
             ROOT / "data" / "csaw" / "registered_site_cache.pkl"),
}


def normalize(manifest: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Columnas comunes: set_id, kind, pid, view, path_t0, path_t1,
    roi0 (sitio marcado en t-1 o NaN) y roi1 (sitio en t)."""
    m = manifest.copy()
    if dataset == "embed":
        m["pid"] = m["empi_anon"]
        m["roi0_r"], m["roi0_c"] = np.nan, np.nan
        m["roi1_r"], m["roi1_c"] = m["roi_r"], m["roi_c"]
        m["view"] = m["view"].str.split("_").str[-1]
    return m


def case_sites(path_t0, path_t1, roi0, roi1, metodo: str):
    """Posiciones relativas (caja de la mama) del sitio en t-1 y en t de un caso."""
    a0, a1 = read_dicom_gray(path_t0), read_dicom_gray(path_t1)
    b0, b1 = ls._full_bbox(a0), ls._full_bbox(a1)
    if roi1 is None:                                  # solo marca en t-1: se lleva a t
        roi1 = map_point_registered(a0, a1, *roi0, method=metodo)
    if roi0 is None:                                  # solo marca en t: se lleva a t-1
        roi0 = map_point_registered(a1, a0, *roi1, method=metodo)
    return cs._rel(roi0, b0), cs._rel(roi1, b1)


def compute(m: pd.DataFrame, cache_path: Path, metodo: str, verbose: bool = True):
    from joblib import Parallel, delayed

    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    cases = m[m["kind"] == "incidente"]
    roi = lambda r, w: None if pd.isna(getattr(r, f"roi{w}_r")) else (getattr(r, f"roi{w}_r"), getattr(r, f"roi{w}_c"))

    key_site = lambda r: ("site", r.path_t0, r.path_t1, metodo, roi(r, 0), roi(r, 1))
    pending = [r for r in cases.itertuples() if key_site(r) not in cache]
    for start in range(0, len(pending), 32):
        chunk = pending[start:start + 32]
        res = Parallel(n_jobs=-1)(delayed(case_sites)(
            ROOT / r.path_t0, ROOT / r.path_t1, roi(r, 0), roi(r, 1), metodo) for r in chunk)
        cache.update({key_site(r): s for r, s in zip(chunk, res)})
        cache_path.write_bytes(pickle.dumps(cache))
        if verbose:
            print(f"  registro: {min(start + 32, len(pending))}/{len(pending)} casos", flush=True)
    sites = {r.set_id: cache[key_site(r)] for r in cases.itertuples()}

    key_feat = lambda r: ("feat", r.path_t0, r.path_t1, sites[r.set_id], cs.HALF)
    rows = list(m.itertuples())
    pending = [r for r in rows if key_feat(r) not in cache]
    for start in range(0, len(pending), 64):
        chunk = pending[start:start + 64]
        res = Parallel(n_jobs=-1)(delayed(cs.site_features)(
            ROOT / r.path_t0, ROOT / r.path_t1, *sites[r.set_id], cs.HALF) for r in chunk)
        cache.update({key_feat(r): f for r, f in zip(chunk, res)})
        cache_path.write_bytes(pickle.dumps(cache))
        if verbose:
            print(f"  ventanas: {min(start + 64, len(pending))}/{len(pending)}", flush=True)

    feats = [cache[key_feat(r)] for r in rows]
    y = (m["kind"] == "incidente").astype(int).values
    case_pid = cases.set_index("set_id")["pid"].to_dict()
    groups = np.array([case_pid[s] for s in m["set_id"]])
    return feats, y, groups, m["set_id"].values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--metodo", choices=["caja", "afin", "elastico"], default="elastico")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--subconjunto", choices=["todo", "trasladado"], default="todo",
                        help="'trasladado': solo conjuntos cuyo caso NO tiene marca en t-1 (CSAW-CC).")
    args = parser.parse_args()

    manifest_path, cache_path = DATASETS[args.dataset]
    m = normalize(pd.read_csv(manifest_path), args.dataset)
    if args.subconjunto == "trasladado":
        sin_marca = m.groupby("set_id")["roi0_r"].transform(lambda s: s.isna().all())
        m = m[sin_marca].reset_index(drop=True)
    feats, y, groups, sets = compute(m, cache_path, args.metodo)

    Xa = np.stack([f["pi_t0"] for f in feats])
    Xt = np.stack([f["pi_t1"] for f in feats])
    auc, p = ls.permutation_auc(Xa, y, groups, sets, n=args.permutations)
    or_, p_or = cs.conditional_or(Xa, y, groups, sets)
    a_t, p_t = ls.permutation_auc(Xt, y, groups, sets, n=args.permutations)
    lines = [f"Sitio de la lesion con registro t -> t-1 ({args.dataset}, metodo {args.metodo}, "
             f"subconjunto {args.subconjunto})",
             f"{int(y.sum())} vistas de caso, {int((1 - y).sum())} controles; {args.permutations} permutaciones", "",
             f"  anticipacion (t-1)     : AUC = {auc:.3f}, p = {p:.3f}; OR condicional = {or_:.2f} (p = {p_or:.2g})",
             f"  control positivo (t)   : AUC = {a_t:.3f}, p = {p_t:.3f}"]
    report = "\n".join(lines)
    print("\n" + report)
    out = RESULTS_DIR / f"registro_sitio_{args.dataset}_{args.metodo}_{args.subconjunto}.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
