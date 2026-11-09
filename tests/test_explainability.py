"""Tests de src/explainability.py con datos e imagenes sinteticas pequenas
(no requieren el piloto real descargado ni la CNN completa de DenseNet121,
para que corran rapido en CI). Issues #59, #60, #61, #64.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import explainability as xai

torch = pytest.importorskip("torch", reason="torch no instalado en este entorno")


def test_iou_and_dice_identical_masks_are_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert xai.iou(mask, mask) == pytest.approx(1.0)
    assert xai.dice(mask, mask) == pytest.approx(1.0)


def test_iou_and_dice_disjoint_masks_are_zero():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert xai.iou(a, b) == 0.0
    assert xai.dice(a, b) == 0.0


def test_iou_and_dice_empty_masks_are_zero_not_nan():
    empty = np.zeros((5, 5), dtype=bool)
    assert xai.iou(empty, empty) == 0.0
    assert xai.dice(empty, empty) == 0.0


def test_binarize_heatmap_keeps_requested_fraction():
    heatmap = np.linspace(0, 1, 100).reshape(10, 10)
    mask = xai.binarize_heatmap(heatmap, percentile=80.0)
    # top 20% de 100 valores = 20 pixeles (puede variar 1-2 por el metodo de interpolacion del percentil)
    assert 15 <= mask.sum() <= 25


def test_dominant_topological_region_finds_a_localized_dip():
    # Filtracion por sublevel (igual que src.topology): un "hueco" oscuro
    # (valor bajo) sobre un fondo mas claro nace pronto en el barrido de
    # umbral y no se fusiona con el resto hasta que el umbral sube bastante,
    # dando una persistencia grande localizada en su centro. No se fija una
    # coordenada exacta esperada (una imagen sintetica demasiado simple
    # puede crear topologia incidental, p.ej. un borde que forma un ciclo
    # H1 el mismo); se comprueba solo que la region encontrada es localizada
    # y no trivial, que es lo que de verdad usa scripts/explainability_benchmark.py.
    rng = np.random.default_rng(0)
    image = rng.normal(loc=0.6, scale=0.02, size=(32, 32)).clip(0, 1)
    yy, xx = np.mgrid[0:32, 0:32]
    blob = np.exp(-(((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 4.0 ** 2)))
    image = (image - 0.45 * blob).clip(0, 1)

    region, info = xai.dominant_topological_region(image)

    assert region.shape == image.shape
    assert region.dtype == bool
    assert info["dim"] in (0, 1)
    assert info["persistence"] > 0.0
    # una region "dominante" util deberia ser localizada, no toda la imagen
    assert 0 < region.sum() < image.size


def test_dominant_topological_region_handles_flat_image():
    flat_image = np.full((8, 8), 0.5)
    region, info = xai.dominant_topological_region(flat_image)
    assert region.shape == flat_image.shape
    assert info["persistence"] == pytest.approx(0.0)


@pytest.fixture
def tiny_cnn_and_image():
    """Una CNN minima con el mismo atributo `features` que usa compute_gradcam
    (model.features), para no tener que descargar/ejecutar DenseNet121 en
    cada test."""
    torch = pytest.importorskip("torch")
    from torch import nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))

        def forward(self, x):
            return self.classifier(self.features(x))

    model = TinyCNN().eval()
    device = torch.device("cpu")
    image = np.random.default_rng(1).normal(size=(3, 16, 16)).astype(np.float32)
    return model, device, image


def test_compute_gradcam_returns_normalized_map_same_size_as_input(tiny_cnn_and_image):
    model, device, image = tiny_cnn_and_image
    cam = xai.compute_gradcam(model, device, image)
    assert cam.shape == image.shape[1:]
    assert cam.min() >= 0.0 - 1e-6
    assert cam.max() <= 1.0 + 1e-6


def test_compute_shap_region_returns_binary_mask_same_size(tiny_cnn_and_image):
    pytest.importorskip("shap", reason="shap no instalado en este entorno")
    model, device, image = tiny_cnn_and_image
    mask = xai.compute_shap_region(model, device, image, percentile=80.0)
    assert mask.shape == image.shape[1:]
    assert mask.dtype == bool


def test_compute_lime_region_returns_binary_mask_same_size(tiny_cnn_and_image):
    pytest.importorskip("lime", reason="lime no instalado en este entorno")
    pytest.importorskip("skimage", reason="scikit-image no instalado en este entorno")
    model, device, image = tiny_cnn_and_image
    mask = xai.compute_lime_region(model, device, image, num_samples=20)
    assert mask.shape == image.shape[1:]
    assert mask.dtype == bool
