"""Selecciona y descarga un piloto longitudinal REAL de EMBED (issue #92,
#94, O7/O8): para cada paciente, dos exploraciones de la misma vista/lado
separadas en el tiempo -- la primera "normal" (t-1) y, para los casos
incidentes, una posterior donde aparece sospecha/malignidad (t).

Requiere credenciales AWS en el perfil 'emory-embed' (~/.aws/credentials)
y las tablas de metadatos ya descargadas en data/embed/ (ver
docs/acceso_datasets_longitudinales.pdf para como conseguir el acceso).

Uso:
    python scripts/download_embed_longitudinal.py --n-incident 20 --n-control 20

Metodologia de seleccion (ver tfg/capitulos/05-metodologia.tex, seccion
"Metodologia longitudinal"):
  - INCIDENTE: la exploracion mas antigua del paciente tiene asses en
    {N, B} (Negativo/Benigno, BI-RADS 1-2) y una exploracion POSTERIOR
    tiene asses en {S, M, K} (Sospechoso/Altamente sugestivo/Malignidad
    confirmada, BI-RADS 4-6). Se usan esas dos exploraciones como (t-1, t).
  - CONTROL: el paciente tiene 2+ exploraciones y TODAS tienen asses en
    {N, B}. Se usan las dos primeras en el tiempo como (t-1, t).
  - Para cada par de exploraciones, se busca una vista (lateralidad +
    posicion, p.ej. "R MLO") PRESENTE EN AMBAS, de tipo FinalImageType=2D
    (mamografia digital real, no el "C-view" sintetizado de tomosintesis),
    para que la comparacion t-1 vs t sea sobre la misma proyeccion
    anatomica. Si un paciente no tiene ninguna vista en comun, se descarta.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EMBED_DIR = ROOT / "data" / "embed"
IMAGES_DIR = EMBED_DIR / "images"

NORMAL = {"N", "B"}
MALIGN = {"S", "M", "K"}
ASSES_RANK = {"N": 0, "B": 1, "P": 2, "A": 2, "S": 3, "M": 4, "K": 5, "X": -1}


def _load_per_study_assessment(clinical_path: Path) -> pd.DataFrame:
    """Una fila por (paciente, estudio): fecha y assessment mas grave de ese
    estudio (un estudio puede tener varias filas, una por hallazgo)."""
    df = pd.read_csv(clinical_path, usecols=["empi_anon", "acc_anon", "study_date_anon", "asses"])
    df = df.dropna(subset=["study_date_anon"])
    df["study_date_anon"] = pd.to_datetime(df["study_date_anon"], errors="coerce")
    df = df.dropna(subset=["study_date_anon"])
    df["asses_rank"] = df["asses"].map(ASSES_RANK).fillna(-1)
    per_study = (
        df.sort_values("asses_rank")
        .groupby(["empi_anon", "acc_anon"])
        .agg(study_date=("study_date_anon", "first"), asses=("asses", "last"))
        .reset_index()
    )
    return per_study


def select_pairs(per_study: pd.DataFrame, n_incident: int, n_control: int, seed: int = 42):
    """Devuelve una lista de dicts {empi_anon, kind, acc_t0, acc_t1} con los
    accession numbers de la exploracion t-1 (normal) y t (la posterior)."""
    incident_pairs, control_pairs = [], []
    for pid, g in per_study.groupby("empi_anon"):
        g = g.sort_values("study_date")
        if len(g) < 2:
            continue
        first = g.iloc[0]
        if first["asses"] not in NORMAL:
            continue
        rest = g.iloc[1:]
        malign_rows = rest[rest["asses"].isin(MALIGN)]
        if len(malign_rows):
            incident_pairs.append({
                "empi_anon": pid, "kind": "incidente",
                "acc_t0": first["acc_anon"], "acc_t1": malign_rows.iloc[0]["acc_anon"],
            })
        elif (g["asses"].isin(NORMAL)).all():
            control_pairs.append({
                "empi_anon": pid, "kind": "control",
                "acc_t0": first["acc_anon"], "acc_t1": g.iloc[1]["acc_anon"],
            })

    rng_incident = pd.Series(incident_pairs).sample(
        n=min(n_incident, len(incident_pairs)), random_state=seed
    )
    rng_control = pd.Series(control_pairs).sample(
        n=min(n_control, len(control_pairs)), random_state=seed
    )
    return list(rng_incident) + list(rng_control)


def _matching_view(metadata: pd.DataFrame, acc_t0, acc_t1) -> dict | None:
    """Busca una vista (lateralidad+posicion) de tipo 2D presente en AMBOS
    estudios; devuelve las dos rutas DICOM o None si no hay ninguna comun."""
    imgs_t0 = metadata[(metadata["acc_anon"] == acc_t0) & (metadata["FinalImageType"] == "2D")]
    imgs_t1 = metadata[(metadata["acc_anon"] == acc_t1) & (metadata["FinalImageType"] == "2D")]
    if imgs_t0.empty or imgs_t1.empty:
        return None
    imgs_t0 = imgs_t0.assign(view=imgs_t0["ImageLateralityFinal"] + "_" + imgs_t0["ViewPosition"])
    imgs_t1 = imgs_t1.assign(view=imgs_t1["ImageLateralityFinal"] + "_" + imgs_t1["ViewPosition"])
    common = set(imgs_t0["view"]) & set(imgs_t1["view"])
    if not common:
        return None
    # Preferimos MLO (vista mas usada en el resto del proyecto) si esta disponible.
    view = next((v for v in common if v.endswith("_MLO")), sorted(common)[0])
    row_t0 = imgs_t0[imgs_t0["view"] == view].iloc[0]
    row_t1 = imgs_t1[imgs_t1["view"] == view].iloc[0]
    return {"view": view, "path_t0": row_t0["anon_dicom_path"], "path_t1": row_t1["anon_dicom_path"]}


def select_interval_matched_controls(
    manifest: pd.DataFrame, per_study: pd.DataFrame, metadata: pd.DataFrame,
    caliper_days: int = 90, seed: int = 42,
) -> pd.DataFrame:
    """Diseño caso-control emparejado por intervalo (corrige el sesgo del
    diseño original).

    En el diseño original, los controles usaban sus dos PRIMERAS
    exploraciones (consecutivas, mediana ~420 dias) y los incidentes la
    primera normal y la primera maligna (mediana ~1100 dias). Como la deriva
    topologica crece con el tiempo en ambos grupos, el intervalo por si solo
    distinguia las clases (AUC ~0.75 sin mirar ninguna imagen).

    Aqui, para CADA incidente del manifiesto se busca en el pool completo de
    controles (pacientes con 2+ exploraciones, todas Normal/Benigno) un par
    (primera exploracion, exploracion posterior) cuyo intervalo este a
    ``caliper_days`` del intervalo del incidente, con una vista 2D comun a
    ambas exploraciones (preferentemente la misma vista que el incidente).
    Cada paciente control se usa como mucho una vez.

    Devuelve un DataFrame de controles con las columnas del manifiesto mas
    ``s3_t0``/``s3_t1`` (claves a descargar) y ``matched_to`` (el incidente
    al que empareja).
    """
    rng = np.random.default_rng(seed)
    dates = per_study.set_index(["empi_anon", "acc_anon"])["study_date"]
    incidents = manifest[manifest["kind"] == "incidente"].copy()
    incidents["interval"] = [
        (dates[(r.empi_anon, r.acc_t1)] - dates[(r.empi_anon, r.acc_t0)]).days
        for r in incidents.itertuples()
    ]

    pool_rows = []
    for pid, grp in per_study.sort_values("study_date").groupby("empi_anon"):
        if len(grp) < 2 or not grp["asses"].isin(NORMAL).all():
            continue
        first = grp.iloc[0]
        for later in grp.iloc[1:].itertuples():
            pool_rows.append((pid, first["acc_anon"], later.acc_anon,
                              (later.study_date - first["study_date"]).days))
    pool = pd.DataFrame(pool_rows, columns=["empi_anon", "acc_t0", "acc_t1", "interval"])

    views: dict = {}
    meta_2d = metadata[metadata["FinalImageType"] == "2D"]
    for r in meta_2d.itertuples():
        views.setdefault(r.acc_anon, {}).setdefault(
            f"{r.ImageLateralityFinal}_{r.ViewPosition}", r.anon_dicom_path)

    used: set = set()
    out = []
    for inc in incidents.sample(frac=1, random_state=seed).itertuples():
        cand = pool[((pool["interval"] - inc.interval).abs() <= caliper_days)
                    & ~pool["empi_anon"].isin(used)]
        cand = cand.iloc[rng.permutation(len(cand))]
        for c in cand.itertuples():
            v0, v1 = views.get(c.acc_t0, {}), views.get(c.acc_t1, {})
            common = set(v0) & set(v1)
            if not common:
                continue
            if inc.view in common:
                view = inc.view
            else:
                view = next((v for v in sorted(common) if v.endswith("_MLO")), sorted(common)[0])
            used.add(c.empi_anon)
            folder = f"data/embed/images/control_emparejado_{c.empi_anon}"
            out.append({
                "empi_anon": c.empi_anon, "kind": "control", "view": view,
                "acc_t0": c.acc_t0, "acc_t1": c.acc_t1,
                "path_t0": f"{folder}/t0_normal.dcm", "path_t1": f"{folder}/t1_posterior.dcm",
                "s3_t0": v0[view], "s3_t1": v1[view], "matched_to": inc.empi_anon,
            })
            break
    return pd.DataFrame(out)


def _run_match_interval(args) -> None:
    import boto3

    manifest = pd.read_csv(EMBED_DIR / "longitudinal_pilot_manifest.csv")
    if "acc_t0" not in manifest.columns:
        raise SystemExit("El manifiesto no tiene acc_t0/acc_t1; vuelve a generarlo con la version actual del script.")
    per_study = _load_per_study_assessment(EMBED_DIR / "clinical.csv")
    metadata = pd.read_csv(
        EMBED_DIR / "metadata_reduced.csv",
        usecols=["acc_anon", "FinalImageType", "ImageLateralityFinal", "ViewPosition", "anon_dicom_path"],
    )
    controls = select_interval_matched_controls(manifest, per_study, metadata,
                                                caliper_days=args.caliper_days, seed=args.seed)
    n_inc = (manifest["kind"] == "incidente").sum()
    print(f"Controles emparejados 1:1 por intervalo: {len(controls)}/{n_inc} incidentes "
          f"(caliper {args.caliper_days} dias).", flush=True)

    s3 = boto3.Session(profile_name=args.profile, region_name="us-west-2").client("s3")
    for i, row in controls.reset_index(drop=True).iterrows():
        for key, rel in [(row["s3_t0"], row["path_t0"]), (row["s3_t1"], row["path_t1"])]:
            local = ROOT / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                s3.download_file("embed-dataset-open", key, str(local))
        print(f"  [{i + 1}/{len(controls)}] control {row['empi_anon']} ({row['view']}): OK", flush=True)

    incidents = manifest[manifest["kind"] == "incidente"]
    matched_ids = set(controls["matched_to"])
    incidents = incidents[incidents["empi_anon"].isin(matched_ids)]
    matched = pd.concat([incidents, controls.drop(columns=["s3_t0", "s3_t1"])], ignore_index=True)
    out = EMBED_DIR / "longitudinal_pilot_manifest_matched.csv"
    matched.to_csv(out, index=False)
    print(f"Manifiesto emparejado ({len(incidents)} incidentes + {len(controls)} controles) "
          f"guardado en {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-incident", type=int, default=20)
    parser.add_argument("--n-control", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="emory-embed")
    parser.add_argument("--match-interval", action="store_true",
                        help="Diseño caso-control emparejado: para cada incidente del manifiesto "
                             "existente busca en el pool completo de controles uno con el mismo "
                             "intervalo t-1 -> t (+/- --caliper-days), y lo descarga. Corrige el "
                             "sesgo de intervalo (ver select_interval_matched_controls). Escribe "
                             "data/embed/longitudinal_pilot_manifest_matched.csv.")
    parser.add_argument("--caliper-days", type=int, default=90)
    args = parser.parse_args()

    if args.match_interval:
        _run_match_interval(args)
        return

    print("Cargando metadatos clinicos (fechas y BI-RADS por estudio)...")
    per_study = _load_per_study_assessment(EMBED_DIR / "clinical.csv")

    print(f"Seleccionando hasta {args.n_incident} incidentes + {args.n_control} control...")
    pairs = select_pairs(per_study, args.n_incident, args.n_control, seed=args.seed)
    print(f"  {len(pairs)} pacientes candidatos (antes de comprobar vista comun).")

    print("Cargando metadatos de imagen (rutas DICOM)...")
    metadata = pd.read_csv(
        EMBED_DIR / "metadata_reduced.csv",
        usecols=["empi_anon", "acc_anon", "FinalImageType", "ImageLateralityFinal",
                  "ViewPosition", "anon_dicom_path"],
    )

    import boto3

    session = boto3.Session(profile_name=args.profile, region_name="us-west-2")
    s3 = session.client("s3")
    bucket = "embed-dataset-open"

    manifest = []
    for pair in pairs:
        match = _matching_view(metadata, pair["acc_t0"], pair["acc_t1"])
        if match is None:
            continue
        patient_dir = IMAGES_DIR / f"{pair['kind']}_{pair['empi_anon']}"
        patient_dir.mkdir(parents=True, exist_ok=True)
        local_t0 = patient_dir / "t0_normal.dcm"
        local_t1 = patient_dir / "t1_posterior.dcm"
        for key, local in [(match["path_t0"], local_t0), (match["path_t1"], local_t1)]:
            if not local.exists():
                s3.download_file(bucket, key, str(local))
        manifest.append({
            "empi_anon": pair["empi_anon"], "kind": pair["kind"], "view": match["view"],
            "acc_t0": pair["acc_t0"], "acc_t1": pair["acc_t1"],
            "path_t0": str(local_t0.relative_to(ROOT)), "path_t1": str(local_t1.relative_to(ROOT)),
        })
        print(f"  [{len(manifest)}/{len(pairs)}] {pair['kind']} {pair['empi_anon']} ({match['view']}): OK")

    manifest_df = pd.DataFrame(manifest)
    manifest_path = EMBED_DIR / "longitudinal_pilot_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nDescargados {len(manifest)} pares (t-1, t) "
          f"({(manifest_df['kind'] == 'incidente').sum()} incidentes, "
          f"{(manifest_df['kind'] == 'control').sum()} control).")
    print(f"Manifiesto guardado en {manifest_path}")


if __name__ == "__main__":
    main()
