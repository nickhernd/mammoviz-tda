"""Tests de la seleccion longitudinal de EMBED (scripts/download_embed_longitudinal.py)
con tablas minimas construidas a mano: no necesitan credenciales, red ni
imagenes descargadas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_embed_longitudinal as dl  # noqa: E402


def _per_study(rows):
    df = pd.DataFrame(rows, columns=["empi_anon", "acc_anon", "study_date", "asses"])
    df["study_date"] = pd.to_datetime(df["study_date"])
    return df


def test_select_pairs_classifies_incident_and_control():
    per_study = _per_study([
        # incidente: normal -> benigno -> sospechoso
        (1, 10, "2015-01-01", "N"), (1, 11, "2016-01-01", "B"), (1, 12, "2018-01-01", "S"),
        # control: todo normal/benigno
        (2, 20, "2015-01-01", "N"), (2, 21, "2016-01-01", "N"), (2, 22, "2017-01-01", "B"),
        # excluido: la primera exploracion ya es sospechosa
        (3, 30, "2015-01-01", "S"), (3, 31, "2016-01-01", "N"),
        # excluido: una sola exploracion
        (4, 40, "2015-01-01", "N"),
    ])
    pairs = {p["empi_anon"]: p for p in dl.select_pairs(per_study, 10, 10, seed=0)}
    assert set(pairs) == {1, 2}
    assert pairs[1]["kind"] == "incidente" and (pairs[1]["acc_t0"], pairs[1]["acc_t1"]) == (10, 12)
    assert pairs[2]["kind"] == "control" and (pairs[2]["acc_t0"], pairs[2]["acc_t1"]) == (20, 21)


def _meta(accs, view=("L", "MLO")):
    return pd.DataFrame([
        {"acc_anon": acc, "FinalImageType": "2D", "ImageLateralityFinal": view[0],
         "ViewPosition": view[1], "anon_dicom_path": f"images/{acc}.dcm"}
        for acc in accs
    ])


def _incident_manifest():
    return pd.DataFrame([{
        "empi_anon": 1, "kind": "incidente", "view": "L_MLO", "acc_t0": 10, "acc_t1": 12,
        "path_t0": "data/embed/images/incidente_1/t0_normal.dcm",
        "path_t1": "data/embed/images/incidente_1/t1_posterior.dcm",
    }])


def test_interval_matched_control_has_the_incident_interval():
    # Incidente a 3 años. Dos controles candidatos: uno solo con seguimiento a
    # 1 año (no vale), otro con exploraciones a 1, 2 y 3 años (vale: par a 3 años).
    per_study = _per_study([
        (1, 10, "2015-01-01", "N"), (1, 12, "2018-01-01", "S"),
        (2, 20, "2015-01-01", "N"), (2, 21, "2016-01-01", "N"),
        (3, 30, "2014-06-01", "N"), (3, 31, "2015-06-01", "N"),
        (3, 32, "2016-06-01", "B"), (3, 33, "2017-06-01", "N"),
    ])
    controls = dl.select_interval_matched_controls(
        _incident_manifest(), per_study, _meta([20, 21, 30, 31, 32, 33]), caliper_days=90, seed=0)

    assert len(controls) == 1
    c = controls.iloc[0]
    assert c["empi_anon"] == 3 and (c["acc_t0"], c["acc_t1"]) == (30, 33)
    assert c["matched_to"] == 1 and c["view"] == "L_MLO"
    assert c["s3_t0"] == "images/30.dcm" and c["s3_t1"] == "images/33.dcm"


def test_interval_matching_excludes_patients_with_any_suspicious_exam():
    per_study = _per_study([
        (1, 10, "2015-01-01", "N"), (1, 12, "2018-01-01", "S"),
        (2, 20, "2015-01-01", "N"), (2, 21, "2016-06-01", "S"), (2, 22, "2018-01-01", "N"),
    ])
    controls = dl.select_interval_matched_controls(
        _incident_manifest(), per_study, _meta([20, 22]), caliper_days=90, seed=0)
    assert controls.empty


def test_interval_matching_requires_a_common_view():
    per_study = _per_study([
        (1, 10, "2015-01-01", "N"), (1, 12, "2018-01-01", "S"),
        (2, 20, "2015-01-01", "N"), (2, 21, "2018-01-01", "N"),
    ])
    metadata = pd.concat([_meta([20], ("L", "MLO")), _meta([21], ("R", "CC"))])
    controls = dl.select_interval_matched_controls(
        _incident_manifest(), per_study, metadata, caliper_days=90, seed=0)
    assert controls.empty


# ---------------------------------------------------------------------------
# Cohorte de cribado (build_screening_cohort): el diseño recomendado
# ---------------------------------------------------------------------------

def _exams(rows):
    """rows: (empi, acc, fecha, screen, asses(set), malig, malig_side)."""
    df = pd.DataFrame(rows, columns=["empi_anon", "acc_anon", "date", "screen", "asses", "malig", "malig_side"])
    df["date"] = pd.to_datetime(df["date"])
    df["normal"] = df["asses"].apply(lambda s: bool(s) and s <= dl.NORMAL)
    return df


def _views(accs_views):
    return {acc: {v: f"images/{acc}_{v}.dcm" for v in vs} for acc, vs in accs_views.items()}


def test_screening_cohort_uses_screens_cancer_side_and_matched_control():
    exams = _exams([
        # caso: cribado negativo 2016 -> cribado 2017 (rellamada) -> diagnostico 2017 con cancer izquierdo
        (1, 10, "2016-01-01", True, {"N"}, False, None),
        (1, 11, "2017-01-10", True, {"A"}, False, None),
        (1, 12, "2017-02-01", False, {"M"}, True, "L"),
        # control valido: dos cribados negativos a 1 año, misma epoca
        (2, 20, "2016-03-01", True, {"N"}, False, None),
        (2, 21, "2017-03-05", True, {"B"}, False, None),
        # no puede ser control: tuvo una rellamada
        (3, 30, "2016-02-01", True, {"N"}, False, None),
        (3, 31, "2017-02-01", True, {"A"}, False, None),
    ])
    views = _views({10: ["L_MLO", "R_MLO"], 11: ["L_MLO", "R_MLO"], 12: ["L_MLO"],
                    20: ["L_MLO", "R_MLO"], 21: ["L_MLO", "R_MLO"],
                    30: ["L_MLO"], 31: ["L_MLO"]})

    cohort = dl.build_screening_cohort(exams, views, seed=0)

    case = cohort[cohort["kind"] == "incidente"].iloc[0]
    ctrl = cohort[cohort["kind"] == "control"].iloc[0]
    assert (case["acc_t0"], case["acc_t1"]) == (10, 11)   # t = el CRIBADO, no el diagnostico
    assert case["view"] == "L_MLO"                         # lado del cancer
    assert ctrl["empi_anon"] == 2 and (ctrl["acc_t0"], ctrl["acc_t1"]) == (20, 21)
    assert ctrl["view"] == "L_MLO" and ctrl["matched_to"] == 1


def test_screening_cohort_skips_cancers_without_a_preceding_screen():
    exams = _exams([
        # sintomatica: va directa a diagnostico, sin cribado en los 180 dias previos
        (1, 10, "2015-01-01", True, {"N"}, False, None),
        (1, 12, "2017-02-01", False, {"M"}, True, "R"),
        (2, 20, "2015-01-01", True, {"N"}, False, None),
        (2, 21, "2017-02-01", True, {"N"}, False, None),
    ])
    views = _views({10: ["R_MLO"], 12: ["R_MLO"], 20: ["R_MLO"], 21: ["R_MLO"]})
    assert dl.build_screening_cohort(exams, views, seed=0).empty


def test_ffdm_views_excludes_spot_and_magnification():
    meta = pd.DataFrame([
        {"acc_anon": 1, "FinalImageType": "2D", "ImageLateralityFinal": "L", "ViewPosition": "MLO",
         "spot_mag": 1.0, "anon_dicom_path": "spot.dcm"},
        {"acc_anon": 1, "FinalImageType": "2D", "ImageLateralityFinal": "L", "ViewPosition": "MLO",
         "spot_mag": float("nan"), "anon_dicom_path": "ffdm.dcm"},
        {"acc_anon": 1, "FinalImageType": "cview", "ImageLateralityFinal": "L", "ViewPosition": "CC",
         "spot_mag": 0.0, "anon_dicom_path": "cview.dcm"},
    ])
    assert dl._ffdm_views(meta) == {1: {"L_MLO": "ffdm.dcm"}}
