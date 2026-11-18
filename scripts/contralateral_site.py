"""Mama contralateral como control de cada mujer (mejora 2).

Uso:
    python scripts/contralateral_site.py --dataset embed

Para cada fila (caso o control) se toma, en el MISMO cribado t-1, la ventana
homologa de la otra mama: misma altura y misma profundidad desde la pared
toracica (``src.local_topology.mirror_relative``). La diferencia entre la
imagen de persistencia del sitio y la del sitio homologo cancela lo que es
comun a las dos mamas de una mujer (densidad, edad, equipo, procesado), y
deja la asimetria, que es lo que buscan los radiologos.

El sitio en t-1 es el del analisis con registro (scripts/registered_lesion_site.py,
metodo elastico). Requiere scripts/download_contralateral.py.
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

from src.local_topology import chest_wall_side, mirror_relative  # noqa: E402
from src.preprocessing import read_dicom_gray  # noqa: E402
from src.topology import persistence_image_vector  # noqa: E402
import csaw_lesion_site as cs  # noqa: E402
import embed_lesion_site as ls  # noqa: E402
import registered_lesion_site as rs  # noqa: E402

RESULTS_DIR = ROOT / "results"


def contra_window(path_t0: Path, path_contra: Path, rel0: tuple[float, float]) -> np.ndarray:
    """Imagen de persistencia de la ventana homologa en la mama contralateral."""
    a, b = read_dicom_gray(path_t0), read_dicom_gray(path_contra)
    rel_c = mirror_relative(rel0, chest_wall_side(a[::4, ::4]), chest_wall_side(b[::4, ::4]))
    bb = ls._full_bbox(b)
    d = ls._window_diagram(b, *cs._abs(rel_c, bb), cs.HALF)
    return persistence_image_vector(d, n_bins=ls.N_BINS)[0]


def main() -> None:
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["embed", "csaw"], required=True)
    parser.add_argument("--metodo", default="elastico")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--subconjunto", choices=["todo", "trasladado"], default="todo")
    args = parser.parse_args()

    manifest_path, cache_path = rs.DATASETS[args.dataset]
    m = rs.normalize(pd.read_csv(manifest_path.with_name(manifest_path.stem + "_contra.csv")), args.dataset)
    if args.subconjunto == "trasladado":
        sin_marca = m.groupby("set_id")["roi0_r"].transform(lambda s: s.isna().all())
        m = m[sin_marca].reset_index(drop=True)
    # solo conjuntos completos: el caso y todos sus controles con contralateral
    completo = m.groupby("set_id")["path_t0_contra"].transform(lambda s: s.notna().all())
    m = m[completo].reset_index(drop=True)

    feats, y, groups, sets = rs.compute(m, cache_path, args.metodo)
    sites = rs.compute_sites(m, cache_path, args.metodo, verbose=False)

    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    key = lambda r: ("contra", r.path_t0, r.path_t0_contra, sites[r.set_id][0], cs.HALF)
    rows = list(m.itertuples())
    pending = [r for r in rows if key(r) not in cache]
    for start in range(0, len(pending), 64):
        chunk = pending[start:start + 64]
        res = Parallel(n_jobs=-1)(delayed(contra_window)(
            ROOT / r.path_t0, ROOT / r.path_t0_contra, sites[r.set_id][0]) for r in chunk)
        cache.update({key(r): f for r, f in zip(chunk, res)})
        cache_path.write_bytes(pickle.dumps(cache))
        print(f"  contralateral: {min(start + 64, len(pending))}/{len(pending)}", flush=True)
    contra = np.stack([cache[key(r)] for r in rows])

    Xa = np.stack([f["pi_t0"] for f in feats])
    lines = [f"Mama contralateral como control ({args.dataset}, sitio con registro {args.metodo}, "
             f"subconjunto {args.subconjunto})",
             f"{int(y.sum())} vistas de caso, {int((1 - y).sum())} controles (conjuntos completos); "
             f"{args.permutations} permutaciones", ""]
    for nombre, X in [("sitio solo (referencia)", Xa), ("sitio - contralateral", Xa - contra),
                      ("sitio + contralateral (concatenado)", np.hstack([Xa, contra])),
                      ("contralateral solo (control)", contra)]:
        auc, p = ls.permutation_auc(X, y, groups, sets, n=args.permutations)
        lines.append(f"  {nombre:36s}: AUC = {auc:.3f}, p = {p:.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    out = RESULTS_DIR / f"contralateral_{args.dataset}_{args.subconjunto}.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
