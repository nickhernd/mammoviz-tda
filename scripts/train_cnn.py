"""Entrena y evalua la CNN de referencia (DenseNet121, PyTorch) sobre los
casos reales ya descargados, como segunda via de comparacion frente al
enfoque topologico de scripts/run_pipeline.py (objetivo O5 del TFG).

Uso:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --epochs 15 --lr 1e-4

Reutiliza la carga de datos de scripts/run_pipeline.py (mismos CSV reales,
misma particion por paciente, misma deteccion de mascara ROI real), pero en
vez de extraer descriptores topologicos alimenta la imagen preprocesada
directamente a la CNN.

Con el tamano de piloto actual (unos pocos cientos de casos), entrenar
DenseNet121 incluso con la base congelada es un regimen de muy pocos datos
para deep learning: se espera que estos resultados sean tan o mas ruidosos
que los del enfoque topologico (ver issues #77, #89), no una demostracion de
que la CNN "gana". El objetivo aqui es tener el segundo brazo de comparacion
funcionando de extremo a extremo con datos reales, no un resultado definitivo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import load_case_descriptions, split_by_patient  # noqa: E402
from src.evaluate import clinical_metrics, plot_confusion  # noqa: E402
from src.models import build_cnn  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from run_pipeline import _load_case_images  # noqa: E402

RESULTS_DIR = ROOT / "results"

# Normalizacion estandar de ImageNet: DenseNet121 se preentreno con estas
# medias/desviaciones por canal RGB, y espera entradas en ese mismo rango.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_cnn_input(img_gray: np.ndarray) -> np.ndarray:
    """Convierte una imagen 2D [0,1] preprocesada a un tensor (3, 224, 224)
    apto para DenseNet121: replica el canal (mamografia = escala de grises)
    a 3 canales y normaliza con las estadisticas de ImageNet.
    """
    rgb = np.repeat(img_gray[:, :, None], 3, axis=2)  # (H, W, 3)
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return rgb.transpose(2, 0, 1)  # (3, H, W), formato esperado por PyTorch


def build_image_dataset(df) -> tuple[np.ndarray, np.ndarray]:
    """Carga y preprocesa las imagenes reales disponibles (sin pasar por TDA)."""
    images, labels = [], []
    skipped = 0
    for _, row in df.iterrows():
        loaded = _load_case_images(row)
        if loaded is None:
            skipped += 1
            continue
        full_image, mask = loaded
        img = preprocess(full_image, mask=mask)
        images.append(_to_cnn_input(img))
        labels.append(row["label"])
    if skipped:
        print(f"  ({skipped} casos del CSV omitidos: imagenes no descargadas todavia)")
    return np.asarray(images, dtype=np.float32), np.asarray(labels)


def train_cnn(X_train, y_train, epochs: int, lr: float, batch_size: int, seed: int = 42):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Semilla fija: sin esto, la inicializacion de la cabeza nueva (Linear +
    # BatchNorm) y el orden de barajado del DataLoader cambian en cada
    # ejecucion, y el AUC final puede variar mas por esa aleatoriedad que por
    # cualquier cambio real al pipeline (se observo swing de 0.67 a 0.76 en
    # dos ejecuciones identicas antes de fijar la semilla).
    torch.manual_seed(seed)

    model = build_cnn()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # class_weight="balanced" equivalente para CrossEntropyLoss: pesa mas la
    # clase minoritaria, igual que build_random_forest hace con sklearn.
    counts = np.bincount(y_train)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    # Generator explicito: el generador implicito de DataLoader(shuffle=True)
    # no siempre queda ligado de forma reproducible a torch.manual_seed() solo
    # (se observo variacion de loss entre ejecuciones identicas incluso con la
    # semilla puesta). Con un generator propio, el orden de barajado es
    # bit-a-bit reproducible.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True, generator=shuffle_generator,
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        print(f"  epoch {epoch}/{epochs}  loss={total_loss / len(X_train):.4f}")

    return model, device


def evaluate_cnn(model, device, X_test, y_test):
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)
    return y_pred, probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("Cargando metadatos reales de CBIS-DDSM (data/csv/)...")
    df = load_case_descriptions(ROOT / "data" / "csv")
    df = df[df["_prefix"].str.startswith("Mass")].reset_index(drop=True)

    print("Particionando por paciente (misma particion que run_pipeline.py, seed=42)...")
    partitions = split_by_patient(df)

    print("Cargando y preprocesando imagenes reales (sin TDA, para la CNN)...")
    print("  [train]")
    X_train, y_train = build_image_dataset(partitions["train"])
    print("  [test]")
    X_test, y_test = build_image_dataset(partitions["test"])

    print(f"\nCasos usables: {len(X_train)} train, {len(X_test)} test.")
    if len(X_train) == 0 or len(X_test) == 0:
        print("No hay suficientes casos con imagenes descargadas todavia.")
        return

    print(f"Entrenando DenseNet121 (base congelada, cabeza nueva, {args.epochs} epocas, CPU/GPU segun disponible)...")
    model, device = train_cnn(X_train, y_train, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)

    y_pred, y_score = evaluate_cnn(model, device, X_test, y_test)
    metrics = clinical_metrics(y_test, y_pred, y_score=y_score)

    print("\nMetricas clinicas de la CNN sobre el conjunto de test (datos REALES):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    ax = plot_confusion(y_test, y_pred)
    ax.figure.savefig(RESULTS_DIR / "confusion_matrix_cnn.png", dpi=150, bbox_inches="tight")

    with open(RESULTS_DIR / "metrics_cnn.txt", "w") as f:
        f.write(f"modelo: DenseNet121 (PyTorch, base congelada)\n")
        f.write(f"epochs={args.epochs} lr={args.lr} batch_size={args.batch_size}\n")
        f.write(f"n_train={len(X_train)} n_test={len(X_test)}\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")

    print(f"\nResultados guardados en {RESULTS_DIR}/ (confusion_matrix_cnn.png, metrics_cnn.txt)")


if __name__ == "__main__":
    main()
