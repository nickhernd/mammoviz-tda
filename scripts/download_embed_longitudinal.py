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


MALIGNANT_PATH = {0, 1}  # path_severity: 0 = cancer invasivo, 1 = no invasivo (CDIS)


def _exam_table(clinical_path: Path) -> pd.DataFrame:
    """Una fila por exploracion: fecha, si es de cribado (``desc``), BI-RADS
    presentes, si tiene patologia maligna asociada y el lado del hallazgo
    maligno."""
    c = pd.read_csv(clinical_path, usecols=["empi_anon", "acc_anon", "study_date_anon", "asses",
                                            "desc", "path_severity", "side"])
    c["date"] = pd.to_datetime(c["study_date_anon"], errors="coerce")
    c = c.dropna(subset=["date"])
    c["screen"] = c["desc"].str.contains("screen", case=False, na=False)
    c["malig"] = c["path_severity"].isin(MALIGNANT_PATH)
    malig_side = c[c["malig"]].groupby(["empi_anon", "acc_anon"])["side"].first()
    ex = c.groupby(["empi_anon", "acc_anon"]).agg(
        date=("date", "first"), screen=("screen", "first"),
        asses=("asses", lambda s: set(s.dropna())), malig=("malig", "any"),
    ).reset_index()
    ex["malig_side"] = [malig_side.get((p, a)) for p, a in zip(ex["empi_anon"], ex["acc_anon"])]
    ex["normal"] = ex["asses"].apply(lambda s: bool(s) and s <= NORMAL)
    return ex


def _ffdm_views(metadata: pd.DataFrame) -> dict:
    """acc_anon -> {vista: clave S3} con solo mamografias 2D de campo completo
    (sin compresion focal ni magnificacion: spot_mag != 1)."""
    m = metadata[(metadata["FinalImageType"] == "2D") & (metadata["spot_mag"].fillna(0) != 1)]
    views: dict = {}
    for r in m.itertuples():
        views.setdefault(r.acc_anon, {}).setdefault(
            f"{r.ImageLateralityFinal}_{r.ViewPosition}", r.anon_dicom_path)
    return views


def _pick_view(common: set, side) -> str | None:
    """Vista del lado del cancer si se conoce, preferentemente MLO."""
    if side in ("L", "R"):
        common = {v for v in common if v.startswith(side + "_")}
    if not common:
        return None
    return next((v for v in sorted(common) if v.endswith("_MLO")), sorted(common)[0])


def build_screening_cohort(
    exams: pd.DataFrame, views: dict, n_incident: int | None = None,
    min_gap_days: int = 180, interval_caliper: int = 90, date_caliper: int = 365, seed: int = 42,
) -> pd.DataFrame:
    """Cohorte longitudinal de CRIBADO, caso-control emparejado.

    Corrige los problemas del diseño basado en BI-RADS 4/5:
      - el caso es un cancer CONFIRMADO por anatomia patologica
        (path_severity 0/1), no una sospecha (la mayoria de biopsias de
        BI-RADS 4 son benignas);
      - t y t-1 son SIEMPRE exploraciones de cribado de campo completo
        (spot_mag != 1) en ambos grupos: antes, la t de los incidentes era un
        estudio diagnostico (193/194), en 136/194 una vista de compresion
        focal/magnificacion, y la "deriva" media en parte el tipo de vista;
      - se usa la mama del lado del cancer.

    Caso: t = ultimo cribado en los ``min_gap_days`` previos (inclusive) al
    diagnostico; t-1 = ultimo cribado Normal/Benigno al menos
    ``min_gap_days`` antes de t. Control: paciente sin patologia maligna y
    con todas sus exploraciones Normal/Benigno; su par (t-1, t) son dos
    cribados con el intervalo del caso (+/- ``interval_caliper``), t-1 en la
    misma epoca (+/- ``date_caliper`` dias) y la misma vista.
    """
    rng = np.random.default_rng(seed)
    gap = pd.Timedelta(days=min_gap_days)
    cases = []
    for pid, g in exams.groupby("empi_anon"):
        if not g["malig"].any():
            continue
        dx_row = g[g["malig"]].sort_values("date").iloc[0]
        screens = g[g["screen"]].sort_values("date")
        t = screens[(screens["date"] <= dx_row["date"]) & (screens["date"] >= dx_row["date"] - gap)]
        if t.empty:
            continue
        t = t.iloc[-1]
        prior = screens[(screens["date"] <= t["date"] - gap) & screens["normal"]]
        if prior.empty:
            continue
        t0 = prior.iloc[-1]
        common = set(views.get(t0["acc_anon"], {})) & set(views.get(t["acc_anon"], {}))
        view = _pick_view(common, dx_row["malig_side"])
        if view is None:
            continue
        cases.append({"empi_anon": pid, "acc_t0": t0["acc_anon"], "acc_t1": t["acc_anon"],
                      "date_t0": t0["date"], "interval": (t["date"] - t0["date"]).days, "view": view})
    cases = pd.DataFrame(cases)
    if n_incident is not None and len(cases) > n_incident:
        cases = cases.sample(n=n_incident, random_state=seed)

    healthy = exams.groupby("empi_anon").filter(lambda g: not g["malig"].any() and g["normal"].all())
    pool = []
    for pid, g in healthy[healthy["screen"]].sort_values("date").groupby("empi_anon"):
        rows = list(g.itertuples())
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                pool.append((pid, a.acc_anon, b.acc_anon, a.date, (b.date - a.date).days))
    pool = pd.DataFrame(pool, columns=["empi_anon", "acc_t0", "acc_t1", "date_t0", "interval"])

    used: set = set()
    out = []
    for case in cases.itertuples():
        cand = pool[((pool["interval"] - case.interval).abs() <= interval_caliper)
                    & ((pool["date_t0"] - case.date_t0).dt.days.abs() <= date_caliper)
                    & ~pool["empi_anon"].isin(used)]
        for c in cand.iloc[rng.permutation(len(cand))].itertuples():
            common = set(views.get(c.acc_t0, {})) & set(views.get(c.acc_t1, {}))
            if case.view not in common:
                continue
            used.add(c.empi_anon)
            for kind, row in [("incidente", case), ("control", c)]:
                folder = f"data/embed/images/cribado_{kind}_{row.empi_anon}"
                out.append({
                    "empi_anon": row.empi_anon, "kind": kind, "view": case.view,
                    "acc_t0": row.acc_t0, "acc_t1": row.acc_t1,
                    "path_t0": f"{folder}/t0.dcm", "path_t1": f"{folder}/t1.dcm",
                    "s3_t0": views[row.acc_t0][case.view], "s3_t1": views[row.acc_t1][case.view],
                    "matched_to": None if kind == "incidente" else case.empi_anon,
                })
            break
    return pd.DataFrame(out)


def _ffdm_images(metadata: pd.DataFrame) -> dict:
    """(acc_anon, vista) -> (clave S3, centro de la primera ROI o None) de la
    PRIMERA imagen 2D de campo completo de esa vista: la misma que se
    descarga, para que la ROI corresponda a la imagen usada."""
    m = metadata[(metadata["FinalImageType"] == "2D") & (metadata["spot_mag"].fillna(0) != 1)]
    out: dict = {}
    for r in m.itertuples():
        key = (r.acc_anon, f"{r.ImageLateralityFinal}_{r.ViewPosition}")
        if key in out:
            continue
        roi = None
        if r.num_ROI > 0 and isinstance(r.ROI_coords, str):
            import ast
            ymin, xmin, ymax, xmax = ast.literal_eval(r.ROI_coords)[0]
            roi = ((ymin + ymax) / 2, (xmin + xmax) / 2)
        out[key] = (r.anon_dicom_path, roi)
    return out


def build_lesion_cohort(
    exams: pd.DataFrame, images: dict, n_controls: int = 3,
    min_gap_days: int = 180, interval_caliper: int = 90, date_caliper: int = 365, seed: int = 42,
    ages: dict | None = None, age_caliper: float | None = None,
) -> pd.DataFrame:
    """Cohorte para el analisis en el sitio de la lesion: misma definicion de
    caso que ``build_screening_cohort``, pero una fila por (caso, VISTA) para
    cada vista del lado del cancer con ROI anotada en t (MLO y CC), y hasta
    ``n_controls`` controles emparejados por (caso, vista), con los mismos
    criterios de emparejamiento. Cada control se usa una sola vez.

    Con ``ages`` ({(empi_anon, acc_anon): edad}) y ``age_caliper``, el
    control tambien debe tener en su t-1 la edad del caso en su t-1
    (+/- ``age_caliper`` años); casos sin edad conocida se descartan.

    Columnas extra: ``set_id`` (caso + vista, el grupo de la validacion
    cruzada) y ``roi_r``/``roi_c`` (centro de la lesion en la imagen t del
    caso; vacio en los controles). Las rutas locales incluyen el acc_anon,
    asi que una misma paciente en dos cohortes con exploraciones distintas
    nunca reutiliza una imagen equivocada."""
    use_age = ages is not None and age_caliper is not None
    rng = np.random.default_rng(seed)
    gap = pd.Timedelta(days=min_gap_days)
    views_of: dict = {}
    for acc, view in images:
        views_of.setdefault(acc, set()).add(view)

    cases = []
    for pid, g in exams.groupby("empi_anon"):
        if not g["malig"].any():
            continue
        dx_row = g[g["malig"]].sort_values("date").iloc[0]
        screens = g[g["screen"]].sort_values("date")
        t = screens[(screens["date"] <= dx_row["date"]) & (screens["date"] >= dx_row["date"] - gap)]
        if t.empty:
            continue
        t = t.iloc[-1]
        prior = screens[(screens["date"] <= t["date"] - gap) & screens["normal"]]
        if prior.empty:
            continue
        t0 = prior.iloc[-1]
        common = views_of.get(t0["acc_anon"], set()) & views_of.get(t["acc_anon"], set())
        side = dx_row["malig_side"]
        for view in sorted(common):
            if side in ("L", "R") and not view.startswith(side + "_"):
                continue
            roi = images[(t["acc_anon"], view)][1]
            if roi is None:
                continue
            cases.append({"empi_anon": pid, "acc_t0": t0["acc_anon"], "acc_t1": t["acc_anon"],
                          "date_t0": t0["date"], "interval": (t["date"] - t0["date"]).days,
                          "view": view, "roi": roi})

    healthy = exams.groupby("empi_anon").filter(lambda g: not g["malig"].any() and g["normal"].all())
    pool = []
    for pid, g in healthy[healthy["screen"]].sort_values("date").groupby("empi_anon"):
        rows = list(g.itertuples())
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                pool.append((pid, a.acc_anon, b.acc_anon, a.date, (b.date - a.date).days))
    pool = pd.DataFrame(pool, columns=["empi_anon", "acc_t0", "acc_t1", "date_t0", "interval"])

    used: set = set()
    out = []
    for case in cases:
        set_id = f"{case['empi_anon']}_{case['view']}"
        case_age = ages.get((case["empi_anon"], case["acc_t0"])) if use_age else None
        if use_age and (case_age is None or pd.isna(case_age)):
            continue
        cand = pool[((pool["interval"] - case["interval"]).abs() <= interval_caliper)
                    & ((pool["date_t0"] - case["date_t0"]).dt.days.abs() <= date_caliper)
                    & ~pool["empi_anon"].isin(used)]
        chosen = []
        for c in cand.iloc[rng.permutation(len(cand))].itertuples():
            if c.empi_anon in used or (c.acc_t0, case["view"]) not in images \
                    or (c.acc_t1, case["view"]) not in images:
                continue
            if use_age:
                c_age = ages.get((c.empi_anon, c.acc_t0))
                if c_age is None or pd.isna(c_age) or abs(c_age - case_age) > age_caliper:
                    continue
            used.add(c.empi_anon)
            chosen.append(c)
            if len(chosen) == n_controls:
                break
        if not chosen:
            continue
        members = [("incidente", case["empi_anon"], case["acc_t0"], case["acc_t1"])] + \
                  [("control", c.empi_anon, c.acc_t0, c.acc_t1) for c in chosen]
        for kind, pid, acc0, acc1 in members:
            folder = "data/embed/images/lesion"
            out.append({
                "empi_anon": pid, "kind": kind, "view": case["view"], "set_id": set_id,
                "acc_t0": acc0, "acc_t1": acc1,
                "path_t0": f"{folder}/{pid}_{acc0}_{case['view']}.dcm",
                "path_t1": f"{folder}/{pid}_{acc1}_{case['view']}.dcm",
                "s3_t0": images[(acc0, case["view"])][0], "s3_t1": images[(acc1, case["view"])][0],
                "roi_r": case["roi"][0] if kind == "incidente" else None,
                "roi_c": case["roi"][1] if kind == "incidente" else None,
            })
    return pd.DataFrame(out)


def _run_lesion_cohort(args) -> None:
    import boto3

    exams = _exam_table(EMBED_DIR / "clinical.csv")
    metadata = pd.read_csv(
        EMBED_DIR / "metadata_reduced.csv",
        usecols=["acc_anon", "FinalImageType", "ImageLateralityFinal", "ViewPosition",
                 "spot_mag", "anon_dicom_path", "num_ROI", "ROI_coords"],
    )
    ages = None
    if args.age_caliper is not None:
        clin = pd.read_csv(EMBED_DIR / "clinical.csv", usecols=["empi_anon", "acc_anon", "age_at_study"])
        ages = clin.dropna().groupby(["empi_anon", "acc_anon"])["age_at_study"].first().to_dict()
    cohort = build_lesion_cohort(exams, _ffdm_images(metadata), n_controls=args.n_controls, seed=args.seed,
                                 ages=ages, age_caliper=args.age_caliper)
    inc = cohort[cohort["kind"] == "incidente"]
    print(f"Cohorte de lesion: {inc['empi_anon'].nunique()} canceres, {len(inc)} vistas con ROI, "
          f"{(cohort['kind'] == 'control').sum()} controles.", flush=True)

    s3 = boto3.Session(profile_name=args.profile, region_name="us-west-2").client("s3")
    for i, row in cohort.reset_index(drop=True).iterrows():
        for key, rel in [(row["s3_t0"], row["path_t0"]), (row["s3_t1"], row["path_t1"])]:
            local = ROOT / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                s3.download_file("embed-dataset-open", key, str(local))
        print(f"  [{i + 1}/{len(cohort)}] {row['kind']} {row['empi_anon']} ({row['view']}): OK", flush=True)

    name = "longitudinal_manifest_lesion" + ("_edad" if args.age_caliper is not None else "")
    out = EMBED_DIR / f"{name}.csv"
    cohort.drop(columns=["s3_t0", "s3_t1"]).to_csv(out, index=False)
    print(f"Manifiesto guardado en {out}")


def _run_screening_cohort(args) -> None:
    import boto3

    exams = _exam_table(EMBED_DIR / "clinical.csv")
    metadata = pd.read_csv(
        EMBED_DIR / "metadata_reduced.csv",
        usecols=["acc_anon", "FinalImageType", "ImageLateralityFinal", "ViewPosition",
                 "spot_mag", "anon_dicom_path"],
    )
    cohort = build_screening_cohort(exams, _ffdm_views(metadata), n_incident=args.n_incident, seed=args.seed)
    n_inc = (cohort["kind"] == "incidente").sum()
    print(f"Cohorte de cribado: {n_inc} canceres confirmados detectados en cribado, "
          f"cada uno con su control emparejado.", flush=True)

    s3 = boto3.Session(profile_name=args.profile, region_name="us-west-2").client("s3")
    for i, row in cohort.reset_index(drop=True).iterrows():
        for key, rel in [(row["s3_t0"], row["path_t0"]), (row["s3_t1"], row["path_t1"])]:
            local = ROOT / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                s3.download_file("embed-dataset-open", key, str(local))
        print(f"  [{i + 1}/{len(cohort)}] {row['kind']} {row['empi_anon']} ({row['view']}): OK", flush=True)

    out = EMBED_DIR / "longitudinal_manifest_cribado.csv"
    cohort.drop(columns=["s3_t0", "s3_t1"]).to_csv(out, index=False)
    print(f"Manifiesto guardado en {out}")


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
    parser.add_argument("--cribado", action="store_true",
                        help="DISEÑO RECOMENDADO. Cohorte de cribado: canceres confirmados por "
                             "patologia detectados en cribado (t = ese cribado, t-1 = cribado "
                             "negativo previo) frente a controles con dos cribados negativos "
                             "emparejados por intervalo, epoca y vista. Solo imagenes de campo "
                             "completo. --n-incident limita el numero de casos. Escribe "
                             "data/embed/longitudinal_manifest_cribado.csv.")
    parser.add_argument("--lesion", action="store_true",
                        help="Cohorte para el analisis en el sitio de la lesion: los mismos canceres "
                             "que --cribado, una fila por cada vista del lado del cancer con ROI "
                             "anotada en t, y --n-controls controles emparejados por vista. Escribe "
                             "data/embed/longitudinal_manifest_lesion.csv.")
    parser.add_argument("--n-controls", type=int, default=3)
    parser.add_argument("--age-caliper", type=float, default=None,
                        help="Con --lesion: empareja tambien por edad en t-1 (+/- años). Escribe "
                             "data/embed/longitudinal_manifest_lesion_edad.csv.")
    args = parser.parse_args()

    if args.lesion:
        _run_lesion_cohort(args)
        return
    if args.cribado:
        _run_screening_cohort(args)
        return
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
