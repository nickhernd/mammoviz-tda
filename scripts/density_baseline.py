"""¿Es la anticipacion solo densidad local? Linea base sin topologia.

Uso:
    python scripts/density_baseline.py --dataset csaw
    python scripts/density_baseline.py --dataset embed

Para la MISMA ventana de t-1 del analisis en el sitio (512 px), se calculan
descriptores de intensidad sin topologia:

- densidad absoluta: media, desviacion tipica y percentiles 10/50/90 de la
  ventana en bruto (la topologia trabaja con la ventana normalizada, asi que
  no ve el nivel absoluto);
- histograma de la ventana normalizada (la misma que ve la topologia):
  distribucion de intensidades sin estructura espacial.

Se compara su AUC con el de la topologia y con la combinacion. Si la
densidad sola iguala a la topologia, la señal es "densidad local"; si la
topologia aporta sobre ella, hay estructura espacial que la densidad no capta.
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

from src.local_topology import window_around  # noqa: E402
from src.preprocessing import apply_clahe, read_dicom_gray, resize, to_grayscale_float  # noqa: E402
import csaw_lesion_site as cs  # noqa: E402
import embed_lesion_site as ls  # noqa: E402

RESULTS_DIR = ROOT / "results"


def window_stats(path: Path, rel: tuple[float, float], half: int = 256) -> np.ndarray:
    a = read_dicom_gray(path)
    w = window_around(a, *cs._abs(rel, ls._full_bbox(a)), half).astype(np.float64)
    absolute = [w.mean(), w.std(), *np.percentile(w, [10, 50, 90])]
    norm = resize(apply_clahe(to_grayscale_float(w)), (ls.WINDOW_SIZE, ls.WINDOW_SIZE))
    hist = np.histogram(norm, bins=10, range=(0, 1), density=True)[0]
    return np.concatenate([absolute, hist])


def site_rels(dataset: str, m: pd.DataFrame) -> dict:
    """Posicion relativa del sitio en t-1 de cada conjunto, la misma que usa
    el analisis en el sitio de cada dataset."""
    if dataset == "csaw":
        cache = pickle.loads(cs.CACHE_PATH.read_bytes())
        return {r.set_id: cache[("rel", r.path_t0, r.path_t1)][0]
                for r in m[m["kind"] == "incidente"].itertuples()}
    out = {}
    for r in m[m["kind"] == "incidente"].itertuples():
        b1 = ls._full_bbox(read_dicom_gray(ROOT / r.path_t1))
        out[r.set_id] = cs._rel((r.roi_r, r.roi_c), b1)   # la caja conserva la posicion relativa
    return out


def main() -> None:
    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["csaw", "embed"], required=True)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()

    if args.dataset == "csaw":
        m = pd.read_csv(ROOT / "data" / "csaw" / "manifest_csaw.csv")
        feats, y, groups, sets = cs.compute(m, verbose=False)
    else:
        m = pd.read_csv(ROOT / "data" / "embed" / "longitudinal_manifest_lesion_edad.csv")
        ls.HALF = 256
        feats, y, groups, sets = ls.compute_lesion_cohort_features(m, verbose=False)
    cache_file = ROOT / "data" / args.dataset / "density_window_stats.npy"
    if cache_file.exists() and len(np.load(cache_file)) == len(m):
        stats_ = np.load(cache_file)
    else:
        rels = site_rels(args.dataset, m)
        stats_ = np.stack(Parallel(n_jobs=-1)(delayed(window_stats)(ROOT / r.path_t0, rels[r.set_id])
                                              for r in m.itertuples()))
        np.save(cache_file, stats_)
    topo = np.stack([f["pi_t0"] for f in feats])

    lines = [f"Densidad local frente a topologia, ventana del sitio en t-1 ({args.dataset})",
             f"{int(y.sum())} vistas de caso, {int((1 - y).sum())} controles; {args.permutations} permutaciones", ""]
    for nombre, X in [("densidad absoluta (media, sd, p10/50/90)", stats_[:, :5]),
                      ("histograma de la ventana normalizada", stats_[:, 5:]),
                      ("densidad + histograma (sin topologia)", stats_),
                      ("topologia (imagen de persistencia)", topo),
                      ("topologia + densidad + histograma", np.hstack([topo, stats_]))]:
        auc, p = ls.permutation_auc(X, y, groups, sets, n=args.permutations)
        lines.append(f"  {nombre:42s}: AUC = {auc:.3f}, p = {p:.3f}")

    # ¿aporta la topologia sobre la densidad? scores fuera de fold de cada uno y
    # regresion logistica condicional con los dos a la vez
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from statsmodels.discrete.conditional_models import ConditionalLogit

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    oof = lambda X: cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    z = lambda v: (v - v.mean()) / v.std()
    s_topo, s_dens = z(oof(topo)), z(oof(stats_))
    r = ConditionalLogit(y, np.c_[s_topo, s_dens], groups=sets).fit(disp=0)
    lines += ["", "Regresion logistica condicional con los dos scores (OR por DE):",
              f"  topologia ajustada por densidad: OR = {np.exp(r.params[0]):.2f}, p = {r.pvalues[0]:.3g}",
              f"  densidad ajustada por topologia: OR = {np.exp(r.params[1]):.2f}, p = {r.pvalues[1]:.3g}",
              f"  correlacion entre los dos scores: {np.corrcoef(s_topo, s_dens)[0, 1]:.2f}"]
    report = "\n".join(lines)
    print("\n" + report)
    (RESULTS_DIR / f"densidad_vs_topologia_{args.dataset}.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
