"""Casos de estudio: que estructuras topologicas ve el modelo en el sitio.

Uso:
    python scripts/case_studies.py

En CSAW-CC se toma, entre los casos cuyo sitio en t-1 esta marcado con un
PUNTO (el radiologo no veia nada), el de mayor puntuacion de anticipacion
fuera de particion, y su primer control. Para cada uno se dibuja la ventana
del sitio en t-1 y se marcan los pixeles donde NACEN los rasgos que separan
casos de controles (results/interpretacion_anticipacion.txt):

- ciclos H1 de intensidad y persistencia medias (0.2-0.7): los casos tienen
  menos;
- componentes H0 que nacen en el extremo brillante (>= 0.8) y duran poco
  (< 0.3): los casos tienen mas.

Las posiciones salen de ``gudhi.CubicalComplex.cofaces_of_persistence_pairs``,
que asocia cada par de persistencia a la celda (pixel) donde nace.
"""
from __future__ import annotations

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
import download_csaw as dc  # noqa: E402
import embed_lesion_site as ls  # noqa: E402


def window(path: Path, rel) -> np.ndarray:
    a = read_dicom_gray(path)
    w = window_around(a, *cs._abs(rel, ls._full_bbox(a)), cs.HALF)
    return resize(apply_clahe(to_grayscale_float(w)), (ls.WINDOW_SIZE, ls.WINDOW_SIZE))


def birth_pixels(img: np.ndarray) -> dict[str, np.ndarray]:
    """Pixeles (fila, columna) donde nacen los dos tipos de rasgo."""
    import gudhi

    cc = gudhi.CubicalComplex(top_dimensional_cells=img)
    cc.persistence(homology_coeff_field=2, min_persistence=0.0)
    regular, _ = cc.cofaces_of_persistence_pairs()
    flat = img.ravel(order="F")          # gudhi indexa las celdas en orden de Fortran
    out = {}
    for dim, key, cond in ((1, "ciclos_medios", lambda b, p: (0.2 <= b <= 0.7) and (0.2 <= p <= 0.7)),
                           (0, "componentes_brillantes", lambda b, p: b >= 0.8 and p < 0.3)):
        pts = []
        if dim < len(regular):
            for birth_cell, death_cell in regular[dim]:
                b, d = flat[birth_cell], flat[death_cell]
                if cond(b, d - b):
                    pts.append(np.unravel_index(birth_cell, img.shape, order="F"))
        out[key] = np.array(pts).reshape(-1, 2)
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    m = pd.read_csv(ROOT / "data" / "csaw" / "manifest_csaw.csv")
    feats, y, groups, sets = cs.compute(m, verbose=False)
    Xa = np.stack([f["pi_t0"] for f in feats])
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    score = cross_val_predict(rf, Xa, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    m = m.assign(score=score)

    def is_dot(r):
        if not r.mask_t0:
            return False
        mk = np.array(Image.open(dc.CSAW_DIR / "masks" / r.file_t0.replace(".dcm", "_mask.png")))
        return np.count_nonzero(mk) <= 25

    casos = m[(m["kind"] == "incidente")]
    casos = casos[[is_dot(r) for r in casos.itertuples()]].sort_values("score", ascending=False)
    import pickle
    rel_cache = pickle.loads(cs.CACHE_PATH.read_bytes())

    filas = []
    for caso in casos.head(2).itertuples():
        control = m[(m["set_id"] == caso.set_id) & (m["kind"] == "control")].iloc[0]
        rel0, rel1 = rel_cache[("rel", caso.path_t0, caso.path_t1)]
        filas.append((f"caso (punto) · puntuación {caso.score:.2f}", caso, rel0, rel1))
        filas.append((f"su control · puntuación {control.score:.2f}", control, rel0, rel1))

    fig, axes = plt.subplots(len(filas), 3, figsize=(10.5, 3.5 * len(filas)))
    resumen = []
    for i, (titulo, r, rel0, rel1) in enumerate(filas):
        w0, w1 = window(ROOT / r.path_t0, rel0), window(ROOT / r.path_t1, rel1)
        bp = birth_pixels(w0)
        axes[i, 0].imshow(w0, cmap="gray"); axes[i, 0].set_title(f"{titulo}\nt-1 (cribado anterior)", fontsize=9)
        axes[i, 1].imshow(w0, cmap="gray")
        for key, color, lab in (("ciclos_medios", "#2F8FD8", "ciclos H1 medios"),
                                ("componentes_brillantes", "#F28E2B", "H0 brillantes efímeras")):
            p = bp[key]
            if len(p):
                axes[i, 1].scatter(p[:, 1], p[:, 0], s=9, c=color, label=f"{lab} ({len(p)})", linewidths=0)
        axes[i, 1].legend(fontsize=7, loc="lower right", framealpha=0.8)
        axes[i, 1].set_title("dónde nacen los rasgos", fontsize=9)
        axes[i, 2].imshow(w1, cmap="gray"); axes[i, 2].set_title("t (diagnóstico)", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")
        resumen.append(f"{titulo}: ciclos H1 medios {len(bp['ciclos_medios'])}, "
                       f"H0 brillantes efimeras {len(bp['componentes_brillantes'])}")
    fig.tight_layout()
    out = ROOT / "tfg" / "figuras" / "casos_estudio.png"
    fig.savefig(out, dpi=170)
    texto = "\n".join(["Casos de estudio (CSAW-CC, sitio marcado con punto, mayor puntuacion de anticipacion)", *resumen])
    print(texto)
    (ROOT / "results" / "casos_estudio.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
