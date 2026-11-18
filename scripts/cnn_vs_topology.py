"""Topologia frente a una CNN en la misma ventana del sitio (adenda 5 del plan).

Uso:
    python scripts/cnn_vs_topology.py --dataset csaw
    python scripts/cnn_vs_topology.py --dataset embed

Para cada fila se extrae la MISMA ventana que usa el analisis topologico
(512 px en el sitio, normalizada y con CLAHE), se reduce a 224x224 y se pasa
por DenseNet121 preentrenada en ImageNet y congelada (1024 valores tras el
pooling global). Se compara con la imagen de persistencia usando el mismo
clasificador y protocolo, y se prueba si la topologia aporta sobre la CNN con
una regresion logistica condicional con los dos scores a la vez.
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
SIZE = 224


def windows_for_row(path_t0: Path, path_t1: Path, rel0, rel1) -> np.ndarray:
    """Ventanas normalizadas (2, 224, 224) de t-1 y t."""
    out = []
    for path, rel in ((path_t0, rel0), (path_t1, rel1)):
        a = read_dicom_gray(path)
        w = window_around(a, *cs._abs(rel, ls._full_bbox(a)), 256)
        out.append(resize(apply_clahe(to_grayscale_float(w)), (SIZE, SIZE)).astype(np.float32))
    return np.stack(out)


def site_rels(dataset: str, m: pd.DataFrame) -> dict:
    """(posicion relativa en t-1, en t) del sitio de cada conjunto, la misma
    que usa el analisis topologico de cada cohorte."""
    if dataset == "csaw":
        cache = pickle.loads(cs.CACHE_PATH.read_bytes())
        return {r.set_id: cache[("rel", r.path_t0, r.path_t1)] for r in m[m["kind"] == "incidente"].itertuples()}
    out = {}
    for r in m[m["kind"] == "incidente"].itertuples():
        rel = cs._rel((r.roi_r, r.roi_c), ls._full_bbox(read_dicom_gray(ROOT / r.path_t1)))
        out[r.set_id] = (rel, rel)          # EMBED: la caja conserva la posicion relativa
    return out


def cnn_embeddings(windows: np.ndarray, batch: int = 32) -> np.ndarray:
    """Embeddings de DenseNet121 (ImageNet, congelada) de ventanas (N, 224, 224) en [0, 1]."""
    import torch
    from torchvision.models import DenseNet121_Weights, densenet121

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch):
            x = torch.from_numpy(windows[start:start + batch]).unsqueeze(1).repeat(1, 3, 1, 1)
            f = model.features((x - mean) / std)
            f = torch.nn.functional.adaptive_avg_pool2d(torch.relu(f), 1).flatten(1)
            out.append(f.numpy())
    return np.vstack(out)


def main() -> None:
    from joblib import Parallel, delayed
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from statsmodels.discrete.conditional_models import ConditionalLogit

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

    cache_file = ROOT / "data" / args.dataset / "cnn_embeddings.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        emb_t0, emb_t1 = z["t0"], z["t1"]
    else:
        rels = site_rels(args.dataset, m)
        wins = np.stack(Parallel(n_jobs=-1)(delayed(windows_for_row)(
            ROOT / r.path_t0, ROOT / r.path_t1, *rels[r.set_id]) for r in m.itertuples()))
        print(f"  ventanas extraidas: {wins.shape}", flush=True)
        emb_t0, emb_t1 = cnn_embeddings(wins[:, 0]), cnn_embeddings(wins[:, 1])
        np.savez_compressed(cache_file, t0=emb_t0, t1=emb_t1)

    topo_t0 = np.stack([f["pi_t0"] for f in feats])
    topo_t1 = np.stack([f["pi_t1"] for f in feats])
    lines = [f"Topologia frente a CNN en la misma ventana ({args.dataset}; adenda 5, exploratorio en CSAW)",
             f"{int(y.sum())} vistas de caso, {int((1 - y).sum())} controles; {args.permutations} permutaciones; "
             f"CNN = DenseNet121 ImageNet congelada (1024)", ""]
    for nombre, T, C in (("anticipacion (t-1)", topo_t0, emb_t0), ("control positivo (t)", topo_t1, emb_t1)):
        lines.append(nombre)
        for etiqueta, X in (("topologia", T), ("CNN", C), ("topologia + CNN", np.hstack([T, C]))):
            auc, p = ls.permutation_auc(X, y, groups, sets, n=args.permutations)
            lines.append(f"  {etiqueta:16s}: AUC = {auc:.3f}, p = {p:.3f}")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
        oof = lambda X: cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
        zs = lambda v: (v - v.mean()) / v.std()
        s_t, s_c = zs(oof(T)), zs(oof(C))
        r = ConditionalLogit(y, np.c_[s_t, s_c], groups=sets).fit(disp=0)
        lines += [f"  logit condicional: topologia ajustada por CNN OR = {np.exp(r.params[0]):.2f} "
                  f"(p = {r.pvalues[0]:.3g}); CNN ajustada por topologia OR = {np.exp(r.params[1]):.2f} "
                  f"(p = {r.pvalues[1]:.3g}); correlacion de scores {np.corrcoef(s_t, s_c)[0, 1]:.2f}", ""]
    report = "\n".join(lines)
    print("\n" + report)
    (RESULTS_DIR / f"cnn_vs_topologia_{args.dataset}.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
