"""Benchmark de interpretabilidad: TDA vs Grad-CAM vs SHAP vs LIME (issues
#59-#66, linea secundaria del TFG -- "Propuesta A", ver issue #91).

Uso:
    python scripts/explainability_benchmark.py
    python scripts/explainability_benchmark.py --max-cases 15 --lime-samples 100

Que hace, en orden:
  1. Carga el piloto real de CBIS-DDSM (igual que run_pipeline.py/train_cnn.py)
     y entrena la CNN de referencia sobre el split de train.
  2. Para cada caso usable del piloto (train+test; esto es un estudio
     descriptivo de interpretabilidad, no una evaluacion predictiva, asi que
     no hace falta reservar un test set, issue #62), calcula 4 regiones
     binarias sobre la MISMA imagen preprocesada (recortada a la ROI +-16px,
     igual que el resto del pipeline):
       - Grad-CAM (src.explainability.compute_gradcam + binarize_heatmap)
       - Region topologica dominante (src.explainability.dominant_topological_region)
       - SHAP GradientExplainer (src.explainability.compute_shap_region)
       - LIME (src.explainability.compute_lime_region)
     y mide el tiempo que tarda cada metodo (issue #66).
  3. Calcula el acuerdo por pares (IoU y Dice) entre los 4 metodos, la
     metrica central de la "Propuesta A" (issue #61/#65), y tambien el IoU
     de cada metodo contra la mascara de ROI real (issue #61: referencia
     clinica), con una advertencia metodologica honesta: como la imagen ya
     esta recortada muy cerca de la ROI (+-16px), ese IoU-vs-mascara esta
     sesgado al alza por construccion y es menos informativo que el acuerdo
     entre metodos; se reporta igualmente por completitud.
  4. Desglosa el acuerdo por `mass_margins` (benigno circunscrito vs.
     maligno espiculado, etc., issue #62).
  5. Guarda una figura con 2 casos ilustrativos (mas acuerdo / menos acuerdo
     entre TDA y Grad-CAM) y una tabla de resultados en
     results/explainability_benchmark.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import load_case_descriptions, split_by_patient  # noqa: E402
from src.evaluate import bootstrap_ci, paired_test  # noqa: E402
from src.explainability import (  # noqa: E402
    binarize_heatmap,
    compute_gradcam,
    compute_lime_region,
    compute_shap_region,
    dominant_topological_region,
    iou,
    dice,
)
from src.preprocessing import preprocess  # noqa: E402
import run_pipeline as rp  # noqa: E402
import train_cnn as tc  # noqa: E402

RESULTS_DIR = ROOT / "results"
METHODS = ["gradcam", "tda", "shap", "lime"]
PAIRS = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]


def _collect_cases(df):
    """Reutiliza rp._load_case_images pero conserva la fila del CSV (para el
    desglose por mass_margins) y la imagen preprocesada."""
    cases = []
    for _, row in df.iterrows():
        loaded = rp._load_case_images(row)
        if loaded is None:
            continue
        full_image, mask = loaded
        img = preprocess(full_image, mask=mask)
        # use_clahe=False: la mascara ROI es binaria (0/255), no una imagen
        # real; aplicarle CLAHE no tiene sentido (es un realce de contraste
        # pensado para intensidades de tejido, no para una mascara binaria).
        mask_pp = preprocess(mask, mask=mask, use_clahe=False) if mask is not None else None
        cases.append({"row": row, "img": img, "mask": mask_pp})
    return cases


def _methods_for_case(model, device, img, roi_mask, timings):
    """Calcula las 4 regiones binarias (+ tiempos) para una imagen."""
    cnn_input = tc._to_cnn_input(img)
    regions = {}

    t0 = time.perf_counter()
    cam = compute_gradcam(model, device, cnn_input)
    regions["gradcam"] = binarize_heatmap(cam, percentile=80.0)
    timings["gradcam"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    region, _info = dominant_topological_region(img)
    regions["tda"] = region
    timings["tda"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    regions["shap"] = compute_shap_region(model, device, cnn_input, percentile=80.0)
    timings["shap"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    regions["lime"] = compute_lime_region(model, device, cnn_input, num_samples=LIME_SAMPLES)
    timings["lime"].append(time.perf_counter() - t0)

    return regions


LIME_SAMPLES = 150


def main() -> None:
    global LIME_SAMPLES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-cases", type=int, default=None, help="Limitar el numero de casos (para pruebas rapidas)")
    parser.add_argument("--lime-samples", type=int, default=150, help="num_samples de LIME (issue #66: mas = mas lento y mas estable)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    LIME_SAMPLES = args.lime_samples

    print("Cargando metadatos reales de CBIS-DDSM (data/csv/)...")
    df = load_case_descriptions(ROOT / "data" / "csv")
    df = df[df["_prefix"].str.startswith("Mass")].reset_index(drop=True)
    partitions = split_by_patient(df)

    print("Entrenando la CNN de referencia sobre el split de train...")
    X_img_train, y_img_train = tc.build_image_dataset(partitions["train"])
    model, device = tc.train_cnn(X_img_train, y_img_train, epochs=args.epochs, lr=1e-4, batch_size=8)

    print("Cargando todos los casos usables del piloto (train+test, estudio descriptivo, issue #62)...")
    all_df = pd_concat_train_test(partitions)
    cases = _collect_cases(all_df)
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"  {len(cases)} casos usables.")

    timings = {m: [] for m in METHODS}
    pairwise = {p: [] for p in PAIRS}
    vs_mask = {m: [] for m in METHODS}
    per_case = []

    for i, case in enumerate(cases):
        row, img, roi_mask = case["row"], case["img"], case["mask"]
        print(f"  [{i + 1}/{len(cases)}] {row['patient_id']} ({row.get('mass_margins', '?')})...")
        regions = _methods_for_case(model, device, img, roi_mask, timings)

        for a, b in PAIRS:
            pairwise[(a, b)].append(iou(regions[a], regions[b]))
        # NaN (no float("nan")) si no hay mascara de ROI real para este caso
        # (variante rara de CBIS-DDSM, ver run_pipeline.py): asi vs_mask[m]
        # queda SIEMPRE alineado indice a indice con per_case, en vez de
        # quedarse mas corto y desalinear el CSV de _save_raw_csv.
        roi_bool = roi_mask > roi_mask.mean() if roi_mask is not None else None
        for m in METHODS:
            vs_mask[m].append(iou(regions[m], roi_bool) if roi_bool is not None else float("nan"))

        per_case.append({"row": row, "img": img, "mask": roi_mask, "regions": regions})

    _report(per_case, pairwise, vs_mask, timings)


def pd_concat_train_test(partitions):
    import pandas as pd

    return pd.concat([partitions["train"], partitions["test"]], ignore_index=True)


def _breakdown(lines, title, per_case, values, field):
    lines.append("")
    lines.append(title)
    groups: dict[str, list[float]] = {}
    for case, v in zip(per_case, values):
        key = str(case["row"].get(field, "NaN"))
        groups.setdefault(key, []).append(v)
    for key, vals in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        vals = np.asarray(vals)
        lines.append(f"  {key:25s}: {vals.mean():.3f} +/- {vals.std():.3f}  (n={len(vals)})")


def _report(per_case, pairwise, vs_mask, timings):
    RESULTS_DIR.mkdir(exist_ok=True)
    lines = []
    lines.append(f"Casos evaluados: {len(per_case)}")
    lines.append("")
    lines.append("=== Acuerdo por pares entre metodos (IoU media, IC 95% bootstrap) ===")
    lines.append("(issue #79: bootstrap de 2000 remuestreos en vez de asumir normalidad,")
    lines.append(" poco fiable con el tamano de piloto actual)")
    for (a, b), values in pairwise.items():
        mean, lo, hi = bootstrap_ci(np.asarray(values))
        lines.append(f"  {a:8s} vs {b:8s}: {mean:.3f}  IC95%=[{lo:.3f}, {hi:.3f}]  (n={len(values)})")

    lines.append("")
    lines.append("=== Test pareado: TDA/Grad-CAM vs. SHAP/LIME (issue #79) ===")
    lines.append("(Wilcoxon signed-rank pareado por caso: ¿el acuerdo que involucra a TDA es")
    lines.append(" significativamente distinto del acuerdo entre dos metodos basados en la")
    lines.append(" misma CNN (SHAP vs LIME)? H0: misma distribucion de IoU por caso.)")
    test = paired_test(pairwise[("gradcam", "tda")], pairwise[("shap", "lime")])
    significativo = "SI (p<0.05)" if test["p_value"] < 0.05 else "NO (p>=0.05)"
    lines.append(f"  estadistico={test['statistic']:.2f}  p-valor={test['p_value']:.4f}  "
                 f"diferencia mediana={test['median_diff']:+.3f}  significativo: {significativo}")

    lines.append("")
    lines.append("=== IoU de cada metodo contra la mascara de ROI real (IC 95% bootstrap) ===")
    lines.append("(nota metodologica: la imagen ya esta recortada a +-16px de la ROI, asi")
    lines.append(" que este IoU esta sesgado al alza por construccion; es menos informativo")
    lines.append(" que el acuerdo entre metodos de arriba, se reporta por completitud)")
    for m in METHODS:
        values = np.asarray(vs_mask[m])
        values = values[~np.isnan(values)]
        if len(values):
            mean, lo, hi = bootstrap_ci(values)
            lines.append(f"  {m:8s}: {mean:.3f}  IC95%=[{lo:.3f}, {hi:.3f}]  (n={len(values)})")

    _breakdown(lines, "=== Desglose de acuerdo TDA vs Grad-CAM por mass_margins ===",
               per_case, pairwise[("gradcam", "tda")], "mass_margins")
    _breakdown(lines, "=== Desglose de acuerdo TDA vs Grad-CAM por breast_density (BI-RADS, issue #90) ===",
               per_case, pairwise[("gradcam", "tda")], "breast_density")

    lines.append("")
    lines.append("=== Coste computacional por metodo (segundos/caso, media +/- desv.) ===")
    for m in METHODS:
        values = np.asarray(timings[m])
        lines.append(f"  {m:8s}: {values.mean():.3f} +/- {values.std():.3f} s")

    report = "\n".join(lines)
    print("\n" + report)
    out_path = RESULTS_DIR / "explainability_benchmark.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nGuardado en {out_path}")

    _save_raw_csv(per_case, pairwise, vs_mask)
    _plot_illustrative_cases(per_case, pairwise[("gradcam", "tda")])


def _save_raw_csv(per_case, pairwise, vs_mask):
    """Datos crudos por caso (issue #79/#90): permite recalcular cualquier
    estadistico (otro IC, otro test, otro desglose) sin re-ejecutar el
    benchmark completo, que es caro por el coste de SHAP (~23 s/caso)."""
    import csv

    out_path = RESULTS_DIR / "explainability_benchmark_cases.csv"
    fieldnames = ["patient_id", "mass_margins", "breast_density"] + \
        [f"{a}_vs_{b}" for a, b in pairwise] + [f"{m}_vs_roi" for m in METHODS]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, case in enumerate(per_case):
            row = {
                "patient_id": case["row"]["patient_id"],
                "mass_margins": case["row"].get("mass_margins", ""),
                "breast_density": case["row"].get("breast_density", ""),
            }
            for (a, b), values in pairwise.items():
                row[f"{a}_vs_{b}"] = values[i]
            for m in METHODS:
                row[f"{m}_vs_roi"] = vs_mask[m][i] if i < len(vs_mask[m]) else ""
            writer.writerow(row)
    print(f"Datos crudos por caso guardados en {out_path}")


def _plot_illustrative_cases(per_case, gradcam_vs_tda):
    import matplotlib.pyplot as plt

    gradcam_vs_tda = np.asarray(gradcam_vs_tda)
    if len(gradcam_vs_tda) < 2:
        return
    best_idx = int(np.argmax(gradcam_vs_tda))
    worst_idx = int(np.argmin(gradcam_vs_tda))

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for row_idx, (idx, label) in enumerate([(best_idx, "Mas acuerdo"), (worst_idx, "Menos acuerdo")]):
        case = per_case[idx]
        img, regions = case["img"], case["regions"]
        patient = case["row"]["patient_id"]

        axes[row_idx, 0].imshow(img, cmap="gray")
        axes[row_idx, 0].set_title(f"{label}\n{patient}")
        for col_idx, method in enumerate(METHODS, start=1):
            axes[row_idx, col_idx].imshow(img, cmap="gray")
            axes[row_idx, col_idx].imshow(regions[method], cmap="Reds", alpha=0.45)
            axes[row_idx, col_idx].set_title(method)
        for ax in axes[row_idx]:
            ax.axis("off")

    plt.tight_layout(h_pad=3.0)
    out_path = RESULTS_DIR / "explainability_cases.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figura de casos ilustrativos guardada en {out_path}")


if __name__ == "__main__":
    main()
