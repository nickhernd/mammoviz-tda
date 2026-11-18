"""Descarga la mamografia t-1 de la mama CONTRALATERAL (misma vista, misma
exploracion) de cada fila de una cohorte de lesion (mejora 2).

Uso:
    python scripts/download_contralateral.py --dataset embed
    python scripts/download_contralateral.py --dataset csaw

Añade la columna ``path_t0_contra`` al manifiesto (fichero nuevo
``*_contra.csv``; el original no se toca). Filas sin imagen contralateral
quedan con la columna vacia.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _embed(args) -> None:
    import boto3
    import download_embed_longitudinal as dl

    src = dl.EMBED_DIR / "longitudinal_manifest_lesion_edad.csv"
    m = pd.read_csv(src)
    meta = pd.read_csv(dl.EMBED_DIR / "metadata_reduced.csv",
                       usecols=["acc_anon", "FinalImageType", "ImageLateralityFinal", "ViewPosition",
                                "spot_mag", "anon_dicom_path", "num_ROI", "ROI_coords"])
    images = dl._ffdm_images(meta)
    s3 = boto3.Session(profile_name="emory-embed", region_name="us-west-2").client("s3")
    paths = []
    for i, r in enumerate(m.itertuples(), 1):
        side, view = r.view.split("_")
        other = f"{'R' if side == 'L' else 'L'}_{view}"
        key = images.get((r.acc_t0, other))
        if key is None:
            paths.append(None)
            continue
        rel = f"data/embed/images/lesion/{r.empi_anon}_{r.acc_t0}_{other}.dcm"
        local = ROOT / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            s3.download_file("embed-dataset-open", key[0], str(local))
        paths.append(rel)
        print(f"  [{i}/{len(m)}] {r.kind} {r.empi_anon} ({other}): OK", flush=True)
    m["path_t0_contra"] = paths
    out = src.with_name(src.stem + "_contra.csv")
    m.to_csv(out, index=False)
    print(f"Manifiesto guardado en {out} ({m['path_t0_contra'].notna().sum()}/{len(m)} con contralateral)")


def _csaw(args) -> None:
    import boto3
    import download_csaw as dc

    src = dc.CSAW_DIR / "manifest_csaw.csv"
    m = pd.read_csv(src)
    listing = set(pd.read_csv(dc.CSAW_DIR / "bucket_listing.csv")["key"])
    s3 = boto3.Session(profile_name="csaw-cc").client("s3", endpoint_url=dc.ENDPOINT, region_name="us-east-1")
    paths = []
    for i, r in enumerate(m.itertuples(), 1):
        other = "R" if r.side == "L" else "L"
        fn = r.file_t0.replace(f"_{r.side}_{r.view}_", f"_{other}_{r.view}_")
        if dc.IMG_PREFIX + fn not in listing:
            paths.append(None)
            continue
        rel = f"data/csaw/images/{fn}"
        local = ROOT / rel
        if not local.exists():
            tmp = local.with_suffix(".dcm.part")
            s3.download_file(dc.BUCKET, dc.IMG_PREFIX + fn, str(tmp))
            tmp.replace(local)
        paths.append(rel)
        print(f"  [{i}/{len(m)}] {r.kind} {r.pid} ({other}_{r.view}): OK", flush=True)
    m["path_t0_contra"] = paths
    out = src.with_name(src.stem + "_contra.csv")
    m.to_csv(out, index=False)
    print(f"Manifiesto guardado en {out} ({m['path_t0_contra'].notna().sum()}/{len(m)} con contralateral)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["embed", "csaw"], required=True)
    args = parser.parse_args()
    (_embed if args.dataset == "embed" else _csaw)(args)


if __name__ == "__main__":
    main()
