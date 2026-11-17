"""¿Es ciego el descriptor global por RESOLUCION o por DILUCION? (issue #95)

Uso:
    python scripts/embed_global_resolution.py [--manifest longitudinal_manifest_cribado.csv]

En la cohorte de cribado, la imagen persistente de la mamografia entera
reducida a 224x224 no separa casos de controles ni en t, con el cancer ya
presente (control positivo ~0.53), mientras que una ventana de ~3 cm en la
lesion si lo hace (~0.67). Dos explicaciones posibles:

  - resolucion: a 224x224 una lesion de 1 cm son 3-4 pixeles;
  - dilucion: aunque se vea, es un rasgo entre miles de la mama entera.

Aqui se repite el control positivo global (imagen t sola) con la mama
recortada a su caja y reducida a 224, 512 y 1024 px. Si el AUC sube con la
resolucion, el problema era la resolucion; si se queda en el azar, es la
dilucion y hace falta un descriptor local.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_topology import breast_bbox  # noqa: E402
from src.preprocessing import apply_clahe, read_dicom_gray, resize, to_grayscale_float  # noqa: E402
from src.topology import persistence_diagram, persistence_image_vector  # noqa: E402

EMBED_DIR = ROOT / "data" / "embed"
RESULTS_DIR = ROOT / "results"
CACHE_PATH = EMBED_DIR / "global_resolution_cache.pkl"
SIZES = (224, 512, 1024)
N_BINS = 6
VERSION = 1


def image_features(path: Path, sizes=SIZES) -> dict[int, np.ndarray]:
    """Imagen de persistencia de la mama recortada a su caja, a varios tamaños."""
    arr = read_dicom_gray(path)
    r0, c0, r1, c1 = breast_bbox(arr[::4, ::4])
    crop = to_grayscale_float(arr[r0 * 4:r1 * 4, c0 * 4:c1 * 4])
    out = {}
    for s in sizes:
        out[s] = persistence_image_vector(persistence_diagram(apply_clahe(resize(crop, (s, s)))), n_bins=N_BINS)[0]
    return out


def main() -> None:
    import argparse

    from joblib import Parallel, delayed

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="longitudinal_manifest_cribado.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(EMBED_DIR / args.manifest)
    y = (manifest["kind"] == "incidente").astype(int).values
    cache = pickle.loads(CACHE_PATH.read_bytes()) if CACHE_PATH.exists() else {}
    keys = [(p, SIZES, N_BINS, VERSION) for p in manifest["path_t1"]]
    pending = [k for k in dict.fromkeys(keys) if k not in cache]
    for start in range(0, len(pending), 32):
        chunk = pending[start:start + 32]
        cache.update(zip(chunk, Parallel(n_jobs=-1)(delayed(image_features)(ROOT / k[0]) for k in chunk)))
        CACHE_PATH.write_bytes(pickle.dumps(cache))
        print(f"  {min(start + 32, len(pending))}/{len(pending)} imagenes", flush=True)

    lines = [f"Control positivo GLOBAL por resolucion ({args.manifest}, {len(manifest)} pacientes; "
             f"imagen t sola, mama recortada a su caja)", ""]
    for s in SIZES:
        X = np.stack([cache[k][s] for k in keys])
        sc = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced",
                                                    n_jobs=-1), X, y, cv=5, scoring="roc_auc")
        lines.append(f"  {s:4d}x{s:<4d}: AUC (5-fold CV) = {sc.mean():.3f} +/- {sc.std():.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    stem = Path(args.manifest).stem.replace("longitudinal_manifest_", "")
    out = RESULTS_DIR / f"embed_global_resolution_{stem}.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
