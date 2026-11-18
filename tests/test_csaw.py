"""Tests de la cohorte CSAW-CC (scripts/download_csaw.py) con una tabla de
metadatos construida a mano: comprueban las reglas del plan
docs/plan_validacion_csaw.md, no son datos del estudio."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_csaw as dc  # noqa: E402


def _meta(rows):
    """rows: (pid, seq, case, timing, recall, age, cancer_side)."""
    out = []
    for pid, seq, case, timing, recall, age, cside in rows:
        for lat in ("Left", "Right"):
            for view in ("MLO", "CC"):
                out.append({"anon_patientid": pid, "x_age": age, "x_case": case,
                            "x_cancer_laterality": cside, "rad_timing": timing, "rad_recall": recall,
                            "imagelaterality": lat, "viewposition": view,
                            "anon_filename": f"{pid:05d}_20990909_{lat[0]}_{view}_{seq}.dcm"})
    df = pd.DataFrame(out)
    df["seq"] = df["anon_filename"].str.extract(r"_(\d+)\.dcm$")[0].astype(int)
    df["side"] = df["imagelaterality"].str[0]
    df["view"] = df["viewposition"]
    df["case_side"] = df["x_cancer_laterality"].astype(str).str[0].str.upper()
    return df


def test_csaw_cohort_follows_the_plan():
    d = _meta([
        # caso: cribado normal (seq 1) -> cancer izquierdo detectado en cribado (seq 2)
        (1, 1, 1, 3, 0, 2, "Left"), (1, 2, 1, 1, 0, 2, "Left"),
        # caso descartado: su cribado anterior fue una rellamada
        (2, 1, 1, 3, 1, 2, "Right"), (2, 2, 1, 1, 0, 2, "Right"),
        # controles: misma banda de edad (2) con dos cribados normales consecutivos
        (3, 1, 0, None, 0, 2, None), (3, 2, 0, None, 0, 2, None),
        (4, 3, 0, None, 0, 2, None), (4, 4, 0, None, 0, 2, None),
        # control descartado: otra banda de edad
        (5, 1, 0, None, 0, 1, None), (5, 2, 0, None, 0, 1, None),
    ])
    masks = {"00001_20990909_L_MLO_1.dcm", "00001_20990909_L_CC_2.dcm"}   # sitio en t-1 (MLO) y en t (CC)
    cohort = dc.build_cohort(d, masks, n_controls=1, seed=0)   # 1 por vista: hay 2 controles validos

    cases = cohort[cohort["kind"] == "incidente"]
    assert set(cases["pid"]) == {1}                       # el 2 se excluye por la rellamada
    assert sorted(cases["view"]) == ["CC", "MLO"] and (cases["side"] == "L").all()
    mlo = cases[cases["view"] == "MLO"].iloc[0]
    assert mlo["mask_t0"] and not mlo["mask_t1"]
    controls = cohort[cohort["kind"] == "control"]
    assert set(controls["pid"]) <= {3, 4} and 5 not in set(controls["pid"])
    # control con el mismo numero de orden de t-1 que el caso, si lo hay
    first_ctl = controls[controls["set_id"] == mlo["set_id"]].iloc[0]
    assert first_ctl["seq0"] == 1


def test_csaw_cohort_excludes_views_without_any_mask():
    d = _meta([(1, 1, 1, 3, 0, 2, "Left"), (1, 2, 1, 1, 0, 2, "Left"),
               (3, 1, 0, None, 0, 2, None), (3, 2, 0, None, 0, 2, None)])
    assert dc.build_cohort(d, set(), n_controls=3, seed=0).empty
