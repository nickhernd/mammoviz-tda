"""Transferencia del modelo de anticipacion entre cohortes (adenda 4 del plan).

Uso:
    python scripts/transfer_model.py

Entrena un Random Forest con TODA una cohorte y lo aplica sin ningun ajuste
a la otra: EMBED -> CSAW-CC (hipotesis H4) y CSAW-CC -> EMBED (secundario),
para la anticipacion (t-1) y el control positivo (t). El p-valor se obtiene
permutando, dentro de cada conjunto caso + controles de la cohorte de
prueba, que miembro es el caso; el modelo entrenado no cambia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import csaw_lesion_site as cs  # noqa: E402
import embed_lesion_site as ls  # noqa: E402

N_PERM = 1000


def load(dataset: str):
    if dataset == "csaw":
        m = pd.read_csv(ROOT / "data" / "csaw" / "manifest_csaw.csv")
        feats, y, _, sets = cs.compute(m, verbose=False)
    else:
        m = pd.read_csv(ROOT / "data" / "embed" / "longitudinal_manifest_lesion_edad.csv")
        ls.HALF = 256
        feats, y, _, sets = ls.compute_lesion_cohort_features(m, verbose=False)
    return {k: np.stack([f[k] for f in feats]) for k in ("pi_t0", "pi_t1")}, y, np.asarray(sets)


def permutation_p(score: np.ndarray, y: np.ndarray, sets: np.ndarray, n: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, score)
    idx_by_set = [np.where(sets == s)[0] for s in np.unique(sets)]
    null = np.empty(n)
    for k in range(n):
        yp = np.zeros_like(y)
        for idx in idx_by_set:
            yp[rng.choice(idx)] = 1
        null[k] = roc_auc_score(yp, score)
    return obs, (1 + np.sum(null >= obs)) / (1 + n)


def main() -> None:
    data = {d: load(d) for d in ("embed", "csaw")}
    lines = ["TRANSFERENCIA DEL MODELO ENTRE COHORTES (adenda 4 del plan)",
             f"Random Forest entrenado con toda la cohorte de origen, aplicado sin ajuste; {N_PERM} permutaciones", ""]
    for src, dst in (("embed", "csaw"), ("csaw", "embed")):
        Xs, ys, _ = data[src]
        Xd, yd, sd = data[dst]
        for clave, nombre in (("pi_t0", "anticipacion (t-1)"), ("pi_t1", "control positivo (t)")):
            rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
            rf.fit(Xs[clave], ys)
            auc, p = permutation_p(rf.predict_proba(Xd[clave])[:, 1], yd, sd, N_PERM)
            tag = "  <- H4" if (src, clave) == ("embed", "pi_t0") else ""
            lines.append(f"  {src.upper():5s} -> {dst.upper():5s} | {nombre:22s}: AUC = {auc:.3f}, p = {p:.4f}{tag}")
    report = "\n".join(lines)
    print(report)
    (ROOT / "results" / "transferencia_modelo.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
