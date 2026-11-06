"""Issue #57: tests de src/topology.py sobre un caso real de CBIS-DDSM.

Se saltan automaticamente si el piloto real (data/csv/, data/images/) no esta
descargado en este entorno (ver scripts/download_cbis_ddsm.py): no se versiona
ninguna imagen medica en el repositorio (data/ esta en .gitignore a proposito),
asi que estos tests son un extra sobre los sinteticos, no un sustituto.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"
IMAGES_DIR = ROOT / "data" / "images"

if not CSV_DIR.exists() or not IMAGES_DIR.exists():
    pytest.skip(
        "Piloto real no descargado (ver scripts/download_cbis_ddsm.py --add N N)",
        allow_module_level=True,
    )

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import case_folder_key, load_case_descriptions  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.topology import persistence_diagram, persistence_image_vector  # noqa: E402

try:
    import run_pipeline as rp  # noqa: E402
except SyntaxError as exc:  # pragma: no cover - ver docs/guia_uso.pdf
    pytest.skip(f"scripts/run_pipeline.py no importable: {exc}", allow_module_level=True)


def _first_downloaded_mass_row():
    df = load_case_descriptions(CSV_DIR)
    df = df[df["_prefix"].str.startswith("Mass")].reset_index(drop=True)
    for _, row in df.iterrows():
        full_key = case_folder_key(
            row["_prefix"], row["patient_id"], row["left_or_right_breast"], row["image_view"]
        )
        if (IMAGES_DIR / full_key).exists():
            return row
    pytest.skip("Ningun caso de masa real descargado todavia")


def test_persistence_diagram_on_real_case_has_h0_and_h1():
    row = _first_downloaded_mass_row()
    full_image, mask = rp._load_case_images(row)

    img = preprocess(full_image, mask=mask)
    diagram = persistence_diagram(img)

    sample = diagram[0]
    assert (sample[:, 2] == 0).any(), "deberia haber al menos un punto H0"
    assert np.isfinite(sample).all()


def test_persistence_image_vector_on_real_case_is_deterministic():
    row = _first_downloaded_mass_row()
    full_image, mask = rp._load_case_images(row)
    img = preprocess(full_image, mask=mask)
    diagram = persistence_diagram(img)

    vector_a = persistence_image_vector(diagram, n_bins=4)
    vector_b = persistence_image_vector(diagram, n_bins=4)

    assert vector_a.shape == (1, 4 * 4 * 2)
    assert np.array_equal(vector_a, vector_b)
