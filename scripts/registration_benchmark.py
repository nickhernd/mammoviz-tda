"""Error geometrico del registro t -> t-1 (calibracion, sin etiquetas).

Uso:
    python scripts/registration_benchmark.py

En CSAW-CC, las vistas de caso con marca del radiologo en t-1 Y en t
señalan el mismo sitio anatomico en dos exploraciones. Se traslada el
centroide de la marca de t a t-1 con cada metodo de src/registration.py y se
mide la distancia (en mm) al centroide de la marca de t-1.

No usa controles, etiquetas ni ningun AUC: solo posiciones. Por eso no
toca el resultado de la validacion (docs/plan_validacion_csaw.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.preprocessing import read_dicom_gray  # noqa: E402
from src.registration import map_point_registered  # noqa: E402
import download_csaw as dc  # noqa: E402

CSAW_DIR = ROOT / "data" / "csaw"
PIXEL_MM = 0.07
METHODS = ("caja", "afin", "elastico")


def one(row) -> dict:
    a0, a1 = read_dicom_gray(ROOT / row.path_t0), read_dicom_gray(ROOT / row.path_t1)
    m0 = dc.mask_centroid(CSAW_DIR / "masks" / row.file_t0.replace(".dcm", "_mask.png"))
    m1 = dc.mask_centroid(CSAW_DIR / "masks" / row.file_t1.replace(".dcm", "_mask.png"))
    out = {"set_id": row.set_id, "view": row.view}
    for meth in METHODS:
        try:
            r, c = map_point_registered(a1, a0, *m1, method=meth)
            out[meth] = float(np.hypot(r - m0[0], c - m0[1])) * PIXEL_MM
        except Exception as e:  # noqa: BLE001
            out[meth] = np.nan
            out[f"{meth}_error"] = repr(e)[:80]
    return out


def main() -> None:
    from joblib import Parallel, delayed

    m = pd.read_csv(CSAW_DIR / "manifest_csaw.csv")
    both = m[(m["kind"] == "incidente") & m["mask_t0"] & m["mask_t1"]]
    res = pd.DataFrame(Parallel(n_jobs=-1)(delayed(one)(r) for r in both.itertuples()))
    lines = [f"Error del registro t -> t-1 en CSAW-CC ({len(res)} vistas de caso con marca en t-1 y en t)",
             "distancia entre la marca de t trasladada y la marca de t-1, en mm", ""]
    for meth in METHODS:
        e = res[meth].dropna()
        lines.append(f"  {meth:9s}: mediana {e.median():5.1f} mm | P75 {e.quantile(.75):5.1f} | "
                     f"<= 18 mm (dentro de media ventana) {np.mean(e <= 18):.0%} | n = {len(e)}")
    report = "\n".join(lines)
    print(report)
    (ROOT / "results" / "registro_csaw.txt").write_text(report + "\n", encoding="utf-8")
    res.to_csv(CSAW_DIR / "registro_errores.csv", index=False)


if __name__ == "__main__":
    main()
