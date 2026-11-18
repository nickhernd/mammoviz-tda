"""¿Que cambia en la topologia del sitio un cribado antes? (interpretacion)

Uso:
    python scripts/interpret_anticipation.py

Para cada conjunto emparejado (caso + controles), diferencia entre la imagen
de persistencia de la ventana del sitio en t-1 del caso y la media de la de
sus controles. Se resume celda a celda (H0 y H1, rejilla 6x6 de nacimiento
x persistencia, ambos normalizados a [0, 1]) con la diferencia media
estandarizada y un test de Wilcoxon con correccion FDR.

Se hace por separado en CSAW-CC (cohorte del analisis principal prefijado) y
en EMBED (cohorte de lesion emparejada por edad) y se compara el patron:
si los dos mapas se parecen, el cambio topologico es reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import csaw_lesion_site as cs  # noqa: E402
import embed_lesion_site as ls  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figuras"
N = ls.N_BINS


def paired_differences(X: np.ndarray, y: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Una fila por conjunto: caso menos la media de sus controles."""
    out = []
    for s in pd.unique(sets):
        idx = np.where(sets == s)[0]
        case, ctl = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(case) and len(ctl):
            out.append(X[case[0]] - X[ctl].mean(axis=0))
    return np.array(out)


def cell_stats(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diferencia media estandarizada (d de Cohen pareada) y p-valor FDR por celda."""
    from scipy import stats
    from statsmodels.stats.multitest import fdrcorrection

    sd = D.std(axis=0, ddof=1)
    d = np.divide(D.mean(axis=0), sd, out=np.zeros(D.shape[1]), where=sd > 0)
    p = np.ones(D.shape[1])
    for k in range(D.shape[1]):
        if np.any(D[:, k] != 0):
            p[k] = stats.wilcoxon(D[:, k]).pvalue
    return d, fdrcorrection(p)[1]


def load(dataset: str):
    if dataset == "csaw":
        m = pd.read_csv(ROOT / "data" / "csaw" / "manifest_csaw.csv")
        feats, y, _, sets = cs.compute(m, verbose=False)
    else:
        m = pd.read_csv(ROOT / "data" / "embed" / "longitudinal_manifest_lesion_edad.csv")
        ls.HALF = 256
        feats, y, _, sets = ls.compute_lesion_cohort_features(m, verbose=False)
    return np.stack([f["pi_t0"] for f in feats]), y, sets


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = {}
    for ds in ("csaw", "embed"):
        X, y, sets = load(ds)
        D = paired_differences(X, y, sets)
        res[ds] = (D, *cell_stats(D))

    lines = ["Interpretacion de la anticipacion: caso menos media de sus controles en la ventana de t-1",
             "celdas: (H, nacimiento, persistencia), ambos ejes normalizados a [0, 1]", ""]
    for ds, (D, d, q) in res.items():
        top = np.argsort(-np.abs(d))[:6]
        lines.append(f"{ds}: {len(D)} conjuntos; celdas con FDR < 0.05: {int(np.sum(q < 0.05))} de {len(d)}")
        for k in top:
            dim, cell = divmod(k, N * N)
            b, p = divmod(cell, N)
            lines.append(f"   H{dim} nacimiento {b / (N - 1):.1f} persistencia {p / (N - 1):.1f}: "
                         f"d = {d[k]:+.2f}, q = {q[k]:.3g}")
        lines.append("")
    r = np.corrcoef(res["csaw"][1], res["embed"][1])[0, 1]
    signo = np.mean(np.sign(res["csaw"][1]) == np.sign(res["embed"][1]))
    lines.append(f"Concordancia del patron EMBED-CSAW: correlacion de los mapas d = {r:.2f}; "
                 f"mismo signo en {signo:.0%} de las celdas")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.2), constrained_layout=True)
    lim = max(np.abs(res[k][1]).max() for k in res)
    for i, ds in enumerate(("csaw", "embed")):
        _, d, q = res[ds]
        for dim in (0, 1):
            ax = axes[i, dim]
            im = ax.imshow(d[dim * N * N:(dim + 1) * N * N].reshape(N, N), cmap="RdBu_r", vmin=-lim, vmax=lim,
                           origin="lower", extent=[0, 1, 0, 1], aspect="auto")
            sig = q[dim * N * N:(dim + 1) * N * N].reshape(N, N) < 0.05
            for bi, pj in zip(*np.nonzero(sig)):
                ax.text((pj + 0.5) / N, (bi + 0.5) / N, "*", ha="center", va="center", fontsize=14)
            ax.set_title(f"{ds.upper()} · H{dim}")
            ax.set_xlabel("persistencia"); ax.set_ylabel("nacimiento (intensidad)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="d pareada (caso − controles)")
    fig.suptitle("Cambio topológico en el sitio, un cribado antes (* = FDR < 0.05)")
    fig.savefig(FIG_DIR / "interpretacion_anticipacion.png", dpi=150)

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "interpretacion_anticipacion.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
