"""Tests de src/data.py: mapeo de etiquetas y particion por paciente.

El foco principal es la garantia central del modulo: que split_by_patient
NUNCA deje un mismo paciente repartido entre dos particiones (fuga de datos).
"""
import pandas as pd
import pytest

from src.data import load_case_descriptions, load_manifest, split_by_patient, to_binary_label


# ---------------------------------------------------------------------------
# to_binary_label
# ---------------------------------------------------------------------------

def test_to_binary_label_maps_known_categories():
    pathology = pd.Series(["BENIGN", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK"])
    labels = to_binary_label(pathology)
    assert labels.tolist() == [0, 1, 0]


def test_to_binary_label_is_case_and_whitespace_insensitive():
    pathology = pd.Series([" benign ", "Malignant", "benign_without_callback"])
    labels = to_binary_label(pathology)
    assert labels.tolist() == [0, 1, 0]


def test_to_binary_label_unknown_category_is_nan():
    pathology = pd.Series(["UNKNOWN"])
    labels = to_binary_label(pathology)
    assert labels.isna().all()


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------

def test_load_manifest_extracts_patient_id(tmp_path):
    xlsx = tmp_path / "manifest.xlsx"
    df = pd.DataFrame(
        {
            " Patient ID ": [
                "Mass-Training_P_01009_RIGHT_CC_1",
                "Calc-Test_P_00512_LEFT_MLO_1",
            ],
            "Series Description": ["full mammogram images", "ROI mask images"],
        }
    )
    df.to_excel(xlsx, index=False)

    manifest = load_manifest(xlsx)

    assert "patient" in manifest.columns
    assert manifest["patient"].tolist() == ["P_01009", "P_00512"]
    # Los nombres de columna deben quedar sin espacios sobrantes.
    assert "Patient ID" in manifest.columns


# ---------------------------------------------------------------------------
# load_case_descriptions
# ---------------------------------------------------------------------------

def test_load_case_descriptions_raises_when_no_csv_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_case_descriptions(tmp_path)


def test_load_case_descriptions_concatenates_and_labels(tmp_path):
    mass_csv = tmp_path / "mass_case_description_train_set.csv"
    calc_csv = tmp_path / "calc_case_description_train_set.csv"
    pd.DataFrame(
        {"patient id": ["P_00001"], "pathology": ["MALIGNANT"]}
    ).to_csv(mass_csv, index=False)
    pd.DataFrame(
        {"patient id": ["P_00002"], "pathology": ["BENIGN"]}
    ).to_csv(calc_csv, index=False)

    df = load_case_descriptions(tmp_path)

    assert len(df) == 2
    # Los espacios en los nombres de columna se normalizan a guion bajo.
    assert "patient_id" in df.columns
    # sorted(glob(...)) ordena "calc_..." antes que "mass_..." alfabeticamente.
    assert df["patient_id"].tolist() == ["P_00002", "P_00001"]
    assert df["label"].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# split_by_patient
# ---------------------------------------------------------------------------

def _make_patient_df(n_patients: int = 40, views_per_patient: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_patients):
        patient_id = f"P_{i:05d}"
        label = i % 2  # alterna benigno/maligno
        for v in range(views_per_patient):
            rows.append({"patient_id": patient_id, "label": label, "view": v})
    return pd.DataFrame(rows)


def test_split_by_patient_no_patient_appears_in_two_partitions():
    df = _make_patient_df()
    partitions = split_by_patient(df, test_size=0.2, val_size=0.2, seed=0)

    train_patients = set(partitions["train"]["patient_id"])
    val_patients = set(partitions["val"]["patient_id"])
    test_patients = set(partitions["test"]["patient_id"])

    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def test_split_by_patient_covers_all_rows_without_duplication():
    df = _make_patient_df()
    partitions = split_by_patient(df, test_size=0.2, val_size=0.2, seed=0)

    total_rows = sum(len(p) for p in partitions.values())
    assert total_rows == len(df)


def test_split_by_patient_is_reproducible_with_same_seed():
    df = _make_patient_df()
    a = split_by_patient(df, seed=123)
    b = split_by_patient(df, seed=123)

    assert a["train"]["patient_id"].tolist() == b["train"]["patient_id"].tolist()
    assert a["test"]["patient_id"].tolist() == b["test"]["patient_id"].tolist()


def test_split_by_patient_keeps_both_classes_in_each_partition():
    df = _make_patient_df(n_patients=60)
    partitions = split_by_patient(df, test_size=0.2, val_size=0.2, seed=0)

    for name, part in partitions.items():
        assert set(part["label"].unique()) == {0, 1}, f"particion {name} sin ambas clases"
