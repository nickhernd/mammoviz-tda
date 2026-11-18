"""¿Señala la topologia del cribado anterior donde aparecera el cancer?
(adenda 2 de docs/plan_validacion_csaw.md; EXPLORATORIO)

Uso:
    python scripts/csaw_localization.py

Se recorre toda la mama de la imagen con una rejilla de ventanas (paso de
media ventana). Un Random Forest, entrenado por validacion cruzada agrupada
por paciente, puntua cada ventana; en cada mama de test se mira en que
percentil queda la ventana del sitio marcado frente al resto de la rejilla
(azar: 0.5). Se hace en t-1 (la pregunta) y en t (control positivo).
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
from src.topology import persistence_image_vector  # noqa: E402
import embed_lesion_site as ls  # noqa: E402

CSAW_DIR = ROOT / "data" / "csaw"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = CSAW_DIR / "localization_cache.pkl"
HALF = 256
STRIDE = 256
PIXEL_MM = 0.07
EXCLUDE_MM = 36.0          # negativas de entrenamiento: a mas de esto del sitio
NEAR_MM = 18.0             # en la evaluacion se ignoran las ventanas que ya contienen el sitio
N_CONTROL_WINDOWS = 10
SEED = 42


def _pi(arr, r, c):
    return persistence_image_vector(ls._window_diagram(arr, r, c, HALF), n_bins=ls.N_BINS)[0]


def grid_centres(arr: np.ndarray) -> list[tuple[int, int]]:
    """Centros de la rejilla dentro de la mama (pixel central por encima del
    umbral de fondo, sobre la imagen reducida)."""
    r0, c0, r1, c1 = ls._full_bbox(arr)
    small = arr[::4, ::4]
    thr = small.min() + 0.05 * (small.max() - small.min())
    out = []
    for r in range(r0 + HALF // 2, r1, STRIDE):
        for c in range(c0 + HALF // 2, c1, STRIDE):
            if small[min(r // 4, small.shape[0] - 1), min(c // 4, small.shape[1] - 1)] > thr:
                out.append((r, c))
    return out


def image_windows(path: Path, site: tuple[float, float] | None, n_random: int | None = None, seed: int = 0) -> dict:
    """Imagenes de persistencia de la rejilla (todas, o ``n_random`` al azar)
    y, si se da ``site``, de la ventana centrada en el sitio."""
    arr = read_dicom_gray(path)
    centres = grid_centres(arr)
    if n_random is not None and len(centres) > n_random:
        idx = np.random.default_rng(seed).choice(len(centres), n_random, replace=False)
        centres = [centres[i] for i in sorted(idx)]
    out = {"centres": np.array(centres, dtype=float).reshape(-1, 2),
           "pi_grid": np.stack([_pi(arr, r, c) for r, c in centres]) if centres else np.empty((0, 72))}
    if site is not None:
        out["pi_site"] = _pi(arr, *site)
    return out


def image_windows_rel(path: Path, rel: tuple[float, float]) -> dict:
    """Rejilla completa y ventana en un PSEUDO-sitio dado por su posicion
    relativa en la caja de la mama (adenda 3: el sitio del caso llevado a su
    control)."""
    import csaw_lesion_site as cs

    arr = read_dicom_gray(path)
    site = cs._abs(rel, ls._full_bbox(arr))
    out = image_windows(path, site)
    out["site"] = np.array(site)
    return out


def load_all(manifest: pd.DataFrame, which: str) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Ventanas de las mamas de caso con sitio marcado en ``which`` (t0 | t1)
    y de las mamas de sus controles en la misma exploracion."""
    from joblib import Parallel, delayed

    sys.path.insert(0, str(ROOT / "scripts"))
    import download_csaw as dc

    col_mask, col_file, col_path = f"mask_{which}", f"file_{which}", f"path_{which}"
    cases = manifest[(manifest["kind"] == "incidente") & manifest[col_mask]].reset_index(drop=True)
    sites = [dc.mask_centroid(CSAW_DIR / "masks" / f.replace(".dcm", "_mask.png")) for f in cases[col_file]]
    controls = manifest[(manifest["kind"] == "control") & manifest["set_id"].isin(cases["set_id"])].reset_index(drop=True)

    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    jobs = [("case", r[col_path], s, None) for r, s in zip(cases.to_dict("records"), sites)] + \
           [("ctl", p, None, N_CONTROL_WINDOWS) for p in controls[col_path]]
    key = lambda j: (j[0], j[1], j[2], j[3], HALF, STRIDE)
    pending = [j for j in jobs if key(j) not in cache]
    for start in range(0, len(pending), 32):
        chunk = pending[start:start + 32]
        res = Parallel(n_jobs=-1)(delayed(image_windows)(ROOT / j[1], j[2], j[3], SEED) for j in chunk)
        cache.update({key(j): x for j, x in zip(chunk, res)})
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        print(f"  {which}: {min(start + 32, len(pending))}/{len(pending)} imagenes", flush=True)
    case_w = [cache[key(j)] | {"site": np.array(j[2])} for j in jobs if j[0] == "case"]
    ctl_w = [cache[key(j)] for j in jobs if j[0] == "ctl"]

    # adenda 3: rejilla completa + pseudo-sitio en el primer control de cada conjunto
    import csaw_lesion_site as cs
    rel_cache = pickle.loads(cs.CACHE_PATH.read_bytes())
    widx = 0 if which == "t0" else 1
    first = controls.groupby("set_id").head(1).set_index("set_id")
    pjobs = []
    for r in cases.itertuples():
        if r.set_id in first.index:
            rel = rel_cache[("rel", r.path_t0, r.path_t1)][widx]
            pjobs.append((r.set_id, first.loc[r.set_id, col_path], tuple(rel)))
    pkey = lambda j: ("pseudo", j[1], j[2], HALF, STRIDE)
    pending = [j for j in pjobs if pkey(j) not in cache]
    for start in range(0, len(pending), 32):
        chunk = pending[start:start + 32]
        res = Parallel(n_jobs=-1)(delayed(image_windows_rel)(ROOT / j[1], j[2]) for j in chunk)
        cache.update({pkey(j): x for j, x in zip(chunk, res)})
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        print(f"  {which} pseudo-sitio: {min(start + 32, len(pending))}/{len(pending)} imagenes", flush=True)
    pseudo = {j[0]: cache[pkey(j)] for j in pjobs}
    return case_w, ctl_w, cases.assign(ctl_index=None), controls, pseudo


def _percentile(rf, w) -> float | None:
    """Percentil de la ventana del sitio frente a las ventanas de la rejilla
    de la misma mama que no contienen el sitio."""
    d = np.hypot(*(w["centres"] - w["site"]).T) * PIXEL_MM
    others = w["pi_grid"][d > NEAR_MM]
    if len(others) < 5:
        return None
    s_site = rf.predict_proba(w["pi_site"][None])[0, 1]
    s_other = rf.predict_proba(others)[:, 1]
    return float(np.mean(s_other < s_site) + 0.5 * np.mean(s_other == s_site))


def evaluate(case_w, ctl_w, cases: pd.DataFrame, controls: pd.DataFrame, pseudo: dict | None = None) -> pd.DataFrame:
    """Percentil del sitio en cada mama de test (CV agrupada por paciente)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    groups = cases["pid"].values
    ctl_by_set = controls.groupby("set_id").indices
    rows = []
    for train, test in GroupKFold(n_splits=5).split(np.zeros(len(cases)), groups=groups):
        X, y = [], []
        for i in train:
            w = case_w[i]
            X.append(w["pi_site"][None]); y.append([1])
            d = np.hypot(*(w["centres"] - w["site"]).T) * PIXEL_MM
            far = w["pi_grid"][d > EXCLUDE_MM]
            X.append(far); y.append(np.zeros(len(far)))
            for j in ctl_by_set.get(cases.loc[i, "set_id"], []):
                X.append(ctl_w[j]["pi_grid"]); y.append(np.zeros(len(ctl_w[j]["pi_grid"])))
        X, y = np.vstack(X), np.concatenate(y)
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced_subsample",
                                    min_samples_leaf=3, n_jobs=-1).fit(X, y)
        for i in test:
            p_case = _percentile(rf, case_w[i])
            if p_case is None:
                continue
            sid = cases.loc[i, "set_id"]
            p_ctl = _percentile(rf, pseudo[sid]) if pseudo and sid in pseudo else None
            rows.append({"set_id": sid, "pid": cases.loc[i, "pid"], "percentil": p_case,
                         "percentil_control": np.nan if p_ctl is None else p_ctl})
    return pd.DataFrame(rows)


def summarize_vs_control(res: pd.DataFrame, label: str) -> list[str]:
    """Adenda 3: sitio del caso frente a pseudo-sitio de su control (pareado)."""
    from scipy import stats

    r = res.dropna(subset=["percentil_control"])
    if len(r) < 6:
        return [f"  {label}: pareados insuficientes ({len(r)})"]
    w = stats.wilcoxon(r["percentil"] - r["percentil_control"], alternative="greater")
    return [f"  {label}: {len(r)} pares | sitio del caso {r['percentil'].median():.3f} frente a pseudo-sitio "
            f"del control {r['percentil_control'].median():.3f} | Wilcoxon pareado p = {w.pvalue:.3g} | "
            f"10 % superior: caso {np.mean(r['percentil'] >= 0.9):.0%}, control {np.mean(r['percentil_control'] >= 0.9):.0%}"]


def summarize(res: pd.DataFrame, label: str) -> list[str]:
    from scipy import stats

    p = res["percentil"].values
    w = stats.wilcoxon(p - 0.5, alternative="greater") if len(p) > 5 else None
    return [f"  {label}: {len(p)} mamas | percentil mediano del sitio = {np.median(p):.3f} (azar 0.5), "
            f"Wilcoxon p = {w.pvalue:.3g} | sitio en el 10 % superior: {np.mean(p >= 0.9):.0%} (azar 10 %)"]


def main() -> None:
    from PIL import Image

    manifest = pd.read_csv(CSAW_DIR / "manifest_csaw.csv")
    lines = ["LOCALIZACION EN CSAW-CC (adenda 2 del plan; EXPLORATORIO)",
             f"rejilla de ventanas de {2 * HALF} px con paso {STRIDE} px; RF por CV agrupada por paciente", ""]
    for which, label in (("t1", "control positivo, t (diagnostico)"), ("t0", "pregunta, t-1 (cribado anterior)")):
        case_w, ctl_w, cases, controls, pseudo = load_all(manifest, which)
        res = evaluate(case_w, ctl_w, cases, controls, pseudo)
        lines += summarize(res, label)
        lines += summarize_vs_control(res, f"   ADENDA 3, {which}")
        if which == "t0":
            npx = {r.set_id: int(np.count_nonzero(np.array(Image.open(
                CSAW_DIR / "masks" / r.file_t0.replace(".dcm", "_mask.png")))))
                for r in cases.itertuples()}
            res["tipo"] = res["set_id"].map(lambda k: "punto (nada visible)" if npx[k] <= 25 else "contorno (signos visibles)")
            for t, g in res.groupby("tipo"):
                lines += summarize(g, f"   t-1, {t}")
                lines += summarize_vs_control(g, f"   ADENDA 3, t-1, {t}")
        res.to_csv(CSAW_DIR / f"localizacion_{which}.csv", index=False)
    report = "\n".join(lines)
    print("\n" + report)
    (RESULTS_DIR / "csaw_localizacion.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
