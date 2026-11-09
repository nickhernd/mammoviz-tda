"""Metodos de interpretabilidad para comparar frente a la region topologica
dominante (issues #59-#66, linea secundaria del TFG: "Propuesta A").

Cuatro metodos producen una REGION BINARIA sobre la misma imagen, para que
puedan compararse entre si y contra la mascara de ROI real (ground truth
clinico) con las mismas metricas (IoU/Dice):

  1. Grad-CAM (Selvaraju et al. 2017) sobre la ultima capa convolucional de
     la CNN de referencia (DenseNet121, src.models.build_cnn).
  2. Region topologica dominante: el par de persistencia (H0 o H1) con mayor
     persistencia, localizado en pixeles via gudhi.cofaces_of_persistence_pairs
     (issue #60).
  3. SHAP (GradientExplainer) sobre la misma CNN.
  4. LIME (lime_image, segmentacion por superpixeles) sobre la misma CNN.

Todas las funciones de este modulo son deliberadamente independientes de
cómo se entrena o carga el modelo/los datos: scripts/explainability_benchmark.py
es el que las orquesta sobre el piloto real.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Metricas de acuerdo entre regiones binarias (issue #61)
# ---------------------------------------------------------------------------

def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Interseccion sobre union de dos mascaras binarias (0.0 si ambas vacias)."""
    a, b = np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Coeficiente de Dice (2x mas sensible al solape que IoU para regiones pequenas)."""
    a, b = np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def binarize_heatmap(heatmap: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    """Convierte un mapa de saliencia continuo en una region binaria: el
    `100-percentile` % de pixeles mas salientes. Percentile=80 -> top 20%.
    """
    threshold = np.percentile(heatmap, percentile)
    return heatmap >= threshold


# ---------------------------------------------------------------------------
# 1. Grad-CAM (issue #59)
# ---------------------------------------------------------------------------

def compute_gradcam(model, device, image_chw: np.ndarray, target_class: int | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al. 2017) sobre `model.features` (la salida de
    la ultima capa convolucional de DenseNet121, forma (1024, 7, 7) para una
    entrada de 224x224). Devuelve un mapa de saliencia 2D normalizado a
    [0, 1], del mismo tamano que la imagen de entrada.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    x = torch.from_numpy(image_chw).float().unsqueeze(0).to(device)
    # La base (model.features) esta congelada (requires_grad=False en sus
    # parametros, src.models.build_cnn); sin esto, autograd no construye
    # grafo dentro de features en absoluto (ni la entrada ni los parametros
    # lo requieren) y el backward hook de abajo nunca se dispara.
    x.requires_grad_(True)

    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(_module, _inp, out):
        activations["value"] = out
        # Hook a nivel de TENSOR (no de modulo): DenseNet121.forward aplica
        # F.relu(features, inplace=True) justo despues de esta capa, y un
        # register_full_backward_hook de modulo choca con esa modificacion
        # in-place ("Output 0 of BackwardHookFunction is a view..."). Un
        # hook sobre el propio tensor de salida evita el problema.
        out.register_hook(lambda grad: gradients.setdefault("value", grad))

    target_layer = model.features
    h_fwd = target_layer.register_forward_hook(fwd_hook)
    try:
        logits = model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        acts = activations["value"][0]   # (C, h, w)
        grads = gradients["value"][0]    # (C, h, w)
        weights = grads.mean(dim=(1, 2))  # (C,) -- promedio espacial del gradiente por canal
        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return cam.detach().cpu().numpy()
    finally:
        h_fwd.remove()


# ---------------------------------------------------------------------------
# 2. Region topologica dominante (issue #60)
# ---------------------------------------------------------------------------

def dominant_topological_region(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Localiza en pixeles el par de persistencia mas significativo (H0 o H1)
    y construye una mascara binaria comparable con un heatmap de Grad-CAM.

    Metodologia (issue #60, formalizada en tfg/capitulos/05-metodologia.tex):
      1. Mismo complejo cubico por sublevel sets que
         src.topology._persistence_diagram_gudhi (altura = intensidad).
      2. ``cofaces_of_persistence_pairs()`` de GUDHI asocia cada par de
         persistencia (nacimiento, muerte) a dos celulas top-dimensionales
         concretas, es decir, dos pixeles.
      3. Se elige el par con mayor persistencia (|muerte - nacimiento|)
         entre H0 y H1 ("el par de persistencia dominante").
      4. Se umbraliza la imagen en el punto medio de ese par y se toma la
         componente conexa del conjunto de sublevel que contiene el pixel
         de nacimiento: esa componente es la region "responsable" de esa
         caracteristica topologica.

    Nota de alcance (honesta, no un resultado exacto): para H1 esto es una
    aproximacion razonable --- la region "real" asociada a un ciclo no es,
    en general, un simple conjunto de sublevel --- pero es la misma
    simplificacion que usan practicamente todos los metodos de localizacion
    basados en homologia persistente cuando no se dispone de ciclos
    representativos explicitos (gudhi no los calcula para complejos
    cubicos). Se documenta aqui para que no se lea como una reconstruccion
    exacta del ciclo.
    """
    import gudhi
    from scipy import ndimage

    img = np.asarray(image, dtype=np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=img)
    cc.persistence(homology_coeff_field=2, min_persistence=0.0)
    regular_pairs, _essential = cc.cofaces_of_persistence_pairs()

    flat = img.reshape(-1)
    best = None  # (persistence, dim, birth_idx, death_idx)
    for dim, pairs in enumerate(regular_pairs):
        if pairs.size == 0:
            continue
        for birth_idx, death_idx in pairs:
            persistence = float(flat[death_idx] - flat[birth_idx])
            if best is None or persistence > best[0]:
                best = (persistence, dim, int(birth_idx), int(death_idx))

    if best is None:
        return np.zeros_like(img, dtype=bool), {"dim": None, "persistence": 0.0}

    persistence, dim, birth_idx, death_idx = best
    birth_val, death_val = float(flat[birth_idx]), float(flat[death_idx])
    threshold = (birth_val + death_val) / 2.0

    sublevel = img <= threshold
    labels, _n = ndimage.label(sublevel)
    birth_coord = np.unravel_index(birth_idx, img.shape)
    target_label = labels[birth_coord]
    region = (labels == target_label) if target_label != 0 else sublevel

    info = {
        "dim": dim,
        "persistence": persistence,
        "birth_coord": birth_coord,
        "death_coord": np.unravel_index(death_idx, img.shape),
        "threshold": threshold,
    }
    return region, info


# ---------------------------------------------------------------------------
# 3. SHAP (issue #64)
# ---------------------------------------------------------------------------

def compute_shap_region(model, device, image_chw: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    """SHAP (GradientExplainer) sobre la misma CNN. El "background" (la
    referencia respecto a la que se mide la contribucion de cada pixel) es
    la imagen en negro, la convencion mas simple y habitual para imagenes
    medicas en escala de grises donde "ausencia de senal" es un fondo
    oscuro razonable.
    """
    import shap
    import torch

    model.eval()
    x = torch.from_numpy(image_chw).float().unsqueeze(0).to(device)
    background = torch.zeros_like(x)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(x)  # (1, C, H, W, n_classes)

    with torch.no_grad():
        target_class = int(model(x).argmax(dim=1).item())

    attribution = np.asarray(shap_values)[0, :, :, :, target_class]  # (C, H, W)
    saliency = np.abs(attribution).sum(axis=0)  # agregamos canales -> (H, W)
    return binarize_heatmap(saliency, percentile=percentile)


# ---------------------------------------------------------------------------
# 4. LIME (issue #64)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_lime_region(model, device, image_chw: np.ndarray, num_samples: int = 200, seed: int = 42) -> np.ndarray:
    """LIME (lime_image, segmentacion por superpixeles de scikit-image) sobre
    la misma CNN. ``image_chw`` se recibe ya normalizado con las
    estadisticas de ImageNet (formato de scripts/train_cnn.py:_to_cnn_input);
    se desnormaliza para mostrarselo a LIME como una imagen RGB [0,255], que
    es lo que espera su API.
    """
    import torch
    import torch.nn.functional as F
    from lime import lime_image

    model.eval()

    def classifier_fn(images_rgb_uint8: np.ndarray) -> np.ndarray:
        batch = []
        for im in images_rgb_uint8:
            norm = (im.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            batch.append(norm.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(batch)).float().to(device)
        with torch.no_grad():
            return F.softmax(model(x), dim=1).cpu().numpy()

    display = image_chw.transpose(1, 2, 0) * _IMAGENET_STD + _IMAGENET_MEAN
    display_uint8 = np.clip(display * 255.0, 0, 255).astype(np.uint8)

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    explanation = explainer.explain_instance(
        display_uint8, classifier_fn, top_labels=2, hide_color=0,
        num_samples=num_samples, random_seed=seed,
    )
    target_class = int(np.argmax(classifier_fn(display_uint8[None])[0]))
    _, mask = explanation.get_image_and_mask(
        target_class, positive_only=True, num_features=8, hide_rest=False
    )
    return mask.astype(bool)
