"""Tests de src/preprocessing.py: normalizacion, ROI, CLAHE y reescalado."""
import numpy as np
import pytest

from src.preprocessing import (
    apply_clahe,
    crop_roi,
    preprocess,
    resize,
    to_grayscale_float,
)


# ---------------------------------------------------------------------------
# to_grayscale_float
# ---------------------------------------------------------------------------

def test_to_grayscale_float_converts_rgb_to_single_channel():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # solo canal rojo encendido

    out = to_grayscale_float(rgb)

    assert out.ndim == 2
    assert out.shape == (10, 10)


def test_to_grayscale_float_normalizes_to_unit_range():
    image = np.array([[0, 50], [100, 200]], dtype=np.uint8)

    out = to_grayscale_float(image)

    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_to_grayscale_float_constant_image_does_not_divide_by_zero():
    image = np.full((5, 5), 128, dtype=np.uint8)

    out = to_grayscale_float(image)

    assert np.all(np.isfinite(out))
    assert np.array_equal(out, np.full((5, 5), 128, dtype=np.float32))


# ---------------------------------------------------------------------------
# crop_roi
# ---------------------------------------------------------------------------

def test_crop_roi_crops_to_mask_bounding_box_with_margin():
    image = np.arange(400).reshape(20, 20).astype(np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 1  # lesion cuadrada en el centro

    cropped = crop_roi(image, mask, margin=2)

    # pixeles de mascara en [8, 11] (ambos inclusive) + margen 2 -> [6, 13)
    assert cropped.shape == (7, 7)
    assert np.array_equal(cropped, image[6:13, 6:13])


def test_crop_roi_empty_mask_returns_original_image():
    image = np.ones((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)

    cropped = crop_roi(image, mask)

    assert np.array_equal(cropped, image)


def test_crop_roi_clips_to_image_bounds():
    image = np.arange(100).reshape(10, 10).astype(np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1  # lesion en la esquina: el margen se saldria de la imagen

    cropped = crop_roi(image, mask, margin=5)

    assert cropped.shape[0] <= 10 and cropped.shape[1] <= 10


# ---------------------------------------------------------------------------
# resize / apply_clahe (requieren OpenCV)
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")


def test_resize_changes_shape_to_requested_size():
    image = np.random.rand(50, 80).astype(np.float32)

    out = resize(image, size=(224, 224))

    assert out.shape == (224, 224)


def test_apply_clahe_keeps_output_in_unit_range():
    image = np.random.rand(32, 32).astype(np.float32)

    out = apply_clahe(image)

    assert out.shape == (32, 32)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# preprocess (pipeline completo)
# ---------------------------------------------------------------------------

def test_preprocess_pipeline_returns_requested_size():
    image = (np.random.rand(100, 120, 3) * 255).astype(np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[30:60, 40:80] = 1

    out = preprocess(image, mask, size=(64, 64))

    assert out.shape == (64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_preprocess_without_mask_skips_crop():
    image = (np.random.rand(50, 50) * 255).astype(np.uint8)

    out = preprocess(image, mask=None, size=(32, 32), use_clahe=False)

    assert out.shape == (32, 32)
