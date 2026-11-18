"""Cohorte de validacion externa en CSAW-CC (docs/plan_validacion_csaw.md).

Uso:
    python scripts/download_csaw.py [--dry-run]

Construye la cohorte EXACTAMENTE como la fija el plan, descarga solo las
imagenes y mascaras necesarias a data/csaw/images/ y escribe
data/csaw/manifest_csaw.csv.

Credenciales: ~/.aws/credentials, perfil [csaw-cc] (nunca en el repo).
Endpoint: https://s3.ki.se, bucket 2021-204.
Cita y condiciones: ver el README de CSAW-CC (los curadores deben ser
invitados como coautores en publicaciones, criterios ICMJE).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSAW_DIR = ROOT / "data" / "csaw"
BUCKET = "2021-204"
IMG_PREFIX = "2021-204-1-1/data/"
MASK_PREFIX = IMG_PREFIX + "anon_annotations_nonhidden/"
ENDPOINT = "https://s3.ki.se"
METADATA = "CSAW_CaseControl_WITHOUTHIDDEN_for_anon_dataset_210902.csv"


def load_metadata(path: Path) -> pd.DataFrame:
    """Una fila por imagen, con el numero de orden de la exploracion y la
    lateralidad/vista normalizadas (L/R, MLO/CC)."""
    d = pd.read_csv(path)
    d["seq"] = d["anon_filename"].str.extract(r"_(\d+)\.dcm$")[0].astype(int)
    d["side"] = d["imagelaterality"].str[0].str.upper()
    d["view"] = d["viewposition"].str.upper()
    d["case_side"] = d["x_cancer_laterality"].astype(str).str[0].str.upper()
    return d


def exam_table(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por exploracion (paciente, orden)."""
    return d.groupby(["anon_patientid", "seq"]).agg(
        case=("x_case", "first"), timing=("rad_timing", "first"), recall=("rad_recall", "first"),
        age=("x_age", "first"), case_side=("case_side", "first"),
    ).reset_index()


def build_cohort(d: pd.DataFrame, masks: set[str], n_controls: int = 3, seed: int = 42) -> pd.DataFrame:
    """Casos, sitio y controles segun el plan.

    ``masks``: nombres de fichero de imagen (``*.dcm``) que tienen mascara.
    Devuelve una fila por miembro de cada conjunto (caso + controles) con
    los nombres de fichero de t-1 y t, y para los casos que mascara existe
    en cada exploracion (``mask_t0``/``mask_t1``)."""
    rng = np.random.default_rng(seed)
    ex = exam_table(d)
    files = d.set_index(["anon_patientid", "seq", "side", "view"])["anon_filename"]

    rows = []
    cases = ex[(ex["case"] == 1) & (ex["timing"] == 1)].groupby("anon_patientid")["seq"].max()
    for pid, s in cases.items():
        prev = ex[(ex["anon_patientid"] == pid) & (ex["seq"] == s - 1)]
        if prev.empty or prev["recall"].iloc[0] != 0:
            continue
        side = prev["case_side"].iloc[0]
        if side not in ("L", "R"):
            continue
        for view in ("MLO", "CC"):
            f0, f1 = files.get((pid, s - 1, side, view)), files.get((pid, s, side, view))
            if not isinstance(f0, str) or not isinstance(f1, str):
                continue
            m0, m1 = f0 in masks, f1 in masks
            if not (m0 or m1):
                continue          # regla 3 del plan: sin mascara en ninguna, se excluye
            rows.append({"pid": pid, "seq0": s - 1, "side": side, "view": view, "age": prev["age"].iloc[0],
                         "file_t0": f0, "file_t1": f1, "mask_t0": m0, "mask_t1": m1})
    cases_df = pd.DataFrame(rows)

    ctl = ex[ex["case"] == 0]
    pairs = ctl.merge(ctl, on="anon_patientid", suffixes=("0", "1"))
    pairs = pairs[(pairs["seq1"] == pairs["seq0"] + 1) & (pairs["recall0"] == 0) & (pairs["recall1"] == 0)]

    used: set = set()
    out = []
    for c in cases_df.itertuples():
        set_id = f"{c.pid}_{c.side}_{c.view}"
        cand = pairs[(pairs["age0"] == c.age) & ~pairs["anon_patientid"].isin(used)].copy()
        has_view = np.array([isinstance(files.get((p, s0, c.side, c.view)), str)
                             and isinstance(files.get((p, s0 + 1, c.side, c.view)), str)
                             for p, s0 in zip(cand["anon_patientid"], cand["seq0"])], dtype=bool)
        cand = cand.loc[has_view]
        cand = cand.assign(dist=(cand["seq0"] - c.seq0).abs(), r=rng.random(len(cand)))
        chosen = []
        for k in cand.sort_values(["dist", "r"]).itertuples():
            if k.anon_patientid in used:
                continue
            used.add(k.anon_patientid)
            chosen.append(k)
            if len(chosen) == n_controls:
                break
        if not chosen:
            continue
        out.append({"set_id": set_id, "kind": "incidente", "pid": c.pid, "side": c.side, "view": c.view,
                    "age": c.age, "seq0": c.seq0, "file_t0": c.file_t0, "file_t1": c.file_t1,
                    "mask_t0": c.mask_t0, "mask_t1": c.mask_t1})
        for k in chosen:
            out.append({"set_id": set_id, "kind": "control", "pid": k.anon_patientid, "side": c.side,
                        "view": c.view, "age": k.age0, "seq0": k.seq0,
                        "file_t0": files[(k.anon_patientid, k.seq0, c.side, c.view)],
                        "file_t1": files[(k.anon_patientid, k.seq0 + 1, c.side, c.view)],
                        "mask_t0": False, "mask_t1": False})
    return pd.DataFrame(out)


def mask_centroid(path: Path) -> tuple[float, float]:
    """Centroide (fila, columna) de los pixeles marcados de una mascara."""
    from PIL import Image

    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean())


def main() -> None:
    import boto3

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-controls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="csaw-cc")
    parser.add_argument("--dry-run", action="store_true", help="Solo construye la cohorte y muestra el tamaño.")
    args = parser.parse_args()

    d = load_metadata(CSAW_DIR / METADATA)
    listing = pd.read_csv(CSAW_DIR / "bucket_listing.csv")
    masks = set(listing.loc[listing["key"].str.endswith("_mask.png"), "key"]
                .str.split("/").str[-1].str.replace("_mask.png", ".dcm", regex=False))
    cohort = build_cohort(d, masks, n_controls=args.n_controls, seed=args.seed)
    inc = cohort[cohort["kind"] == "incidente"]
    print(f"Cohorte CSAW-CC: {inc['pid'].nunique()} canceres, {len(inc)} vistas "
          f"({inc['mask_t0'].sum()} con el sitio marcado en t-1), "
          f"{(cohort['kind'] == 'control').sum()} controles.", flush=True)
    if args.dry_run:
        return

    s3 = boto3.Session(profile_name=args.profile).client("s3", endpoint_url=ENDPOINT, region_name="us-east-1")
    img_dir, mask_dir = CSAW_DIR / "images", CSAW_DIR / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    def _get(key: str, local: Path) -> None:
        if not local.exists():
            tmp = local.with_suffix(local.suffix + ".part")
            s3.download_file(BUCKET, key, str(tmp))
            tmp.replace(local)

    roi = {}
    for c in inc.itertuples():
        for col, flag in (("file_t0", c.mask_t0), ("file_t1", c.mask_t1)):
            if flag:
                fn = getattr(c, col).replace(".dcm", "_mask.png")
                _get(MASK_PREFIX + fn, mask_dir / fn)
                roi[(c.set_id, col)] = mask_centroid(mask_dir / fn)
    cohort["roi0_r"] = [roi.get((r.set_id, "file_t0"), (np.nan, np.nan))[0] if r.kind == "incidente" else np.nan
                        for r in cohort.itertuples()]
    cohort["roi0_c"] = [roi.get((r.set_id, "file_t0"), (np.nan, np.nan))[1] if r.kind == "incidente" else np.nan
                        for r in cohort.itertuples()]
    cohort["roi1_r"] = [roi.get((r.set_id, "file_t1"), (np.nan, np.nan))[0] if r.kind == "incidente" else np.nan
                        for r in cohort.itertuples()]
    cohort["roi1_c"] = [roi.get((r.set_id, "file_t1"), (np.nan, np.nan))[1] if r.kind == "incidente" else np.nan
                        for r in cohort.itertuples()]
    cohort["path_t0"] = [f"data/csaw/images/{f}" for f in cohort["file_t0"]]
    cohort["path_t1"] = [f"data/csaw/images/{f}" for f in cohort["file_t1"]]

    for i, r in enumerate(cohort.itertuples(), 1):
        _get(IMG_PREFIX + r.file_t0, ROOT / r.path_t0)
        _get(IMG_PREFIX + r.file_t1, ROOT / r.path_t1)
        print(f"  [{i}/{len(cohort)}] {r.kind} {r.pid} ({r.side}_{r.view}): OK", flush=True)

    out = CSAW_DIR / "manifest_csaw.csv"
    cohort.to_csv(out, index=False)
    print(f"Manifiesto guardado en {out}")


if __name__ == "__main__":
    sys.exit(main())
