"""Fusiona features topologicas (TDA) y embeddings de la CNN, y compara las
tres vias (TDA solo, CNN sola, fusion) sobre los mismos casos reales y la
misma particion por paciente. Cierra el objetivo O5 del TFG.

Uso:
    python scripts/fuse_and_evaluate.py

Reutiliza scripts/run_pipeline.py (features topologicas) y scripts/train_cnn.py
(datos de imagen + arquitectura de la CNN). La CNN se entrena primero de forma
identica a train_cnn.py; despues se le quita la ultima capa lineal para usarla
como extractor de embeddings de 1024 dimensiones (salida de DenseNet121 tras
el pooling global), que es lo que se concatena con el vector topologico via
src.models.fuse_features.

Nota: con el tamano de piloto actual, se espera que NINGUNA de las tres vias
domine claramente a las otras (ver hallazgos en el issue #77 y en el propio
historial de ejecuciones); el valor de este script es dejar el experimento de
fusion funcionando de extremo a extremo con datos reales, no un veredicto.
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
from src.evaluate import clinical_metrics  # noqa: E402
from src.models import build_random_forest, build_svm, fuse_features  # noqa: E402
from run_pipeline import build_dataset as build_topo_dataset, tune_classifier  # noqa: E402
from train_cnn import build_image_dataset, train_cnn  # noqa: E402


def extract_cnn_embeddings(model, device, X_images: np.ndarray) -> np.ndarray:
    """Embeddings de 1024-d de DenseNet121 (salida tras pooling global, antes
    de la ultima capa lineal de clasificacion), reutilizando la cabeza ya
    entrenada (BatchNorm + Dropout) pero sustituyendo el Linear final por la
    identidad.
    """
    import torch
    from torch import nn

    original_head = model.classifier
    model.classifier = nn.Sequential(*list(original_head.children())[:-1])  # quita el Linear final
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(X_images).to(device)).cpu().numpy()
    model.classifier = original_head  # restauramos por si el modelo se reutiliza
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--classifier", choices=["svm", "rf"], default="rf",
                         help="Clasificador para las 3 vias (TDA/CNN/fusion). Sin --tune, "
                              "se usan los valores por defecto de src/models.py.")
    parser.add_argument("--tune", action="store_true",
                         help="Ajustar hiperparametros por GridSearchCV PARA CADA una de las "
                              "3 vias por separado (issue #77): el numero optimo de "
                              "caracteristicas/regularizacion no es el mismo para TDA (32-d), "
                              "CNN (1024-d) y fusion (1056-d).")
    args = parser.parse_args()

    print("Cargando metadatos reales y particionando por paciente (seed=42, igual que los otros scripts)...")
    df = load_case_descriptions(ROOT / "data" / "csv")
    df = df[df["_prefix"].str.startswith("Mass")].reset_index(drop=True)
    partitions = split_by_patient(df)

    print("\n[Features topologicas]")
    print("  [train]")
    X_topo_train, y_train, _ = build_topo_dataset(partitions["train"], n_bins=args.n_bins)
    print("  [test]")
    X_topo_test, y_test, _ = build_topo_dataset(partitions["test"], n_bins=args.n_bins)

    print("\n[Imagenes preprocesadas para la CNN]")
    print("  [train]")
    X_img_train, y_train_img = build_image_dataset(partitions["train"])
    print("  [test]")
    X_img_test, y_test_img = build_image_dataset(partitions["test"])

    # build_dataset y build_image_dataset iteran el mismo DataFrame en el
    # mismo orden y saltan exactamente los mismos casos (misma condicion de
    # carga), asi que las etiquetas deben coincidir posicionalmente. Lo
    # verificamos explicitamente para no fusionar filas desalineadas.
    assert np.array_equal(y_train, y_train_img), "Desalineacion train: revisa el orden de iteracion"
    assert np.array_equal(y_test, y_test_img), "Desalineacion test: revisa el orden de iteracion"

    print(f"\nCasos usables: {len(X_topo_train)} train, {len(X_topo_test)} test.")
    if len(X_topo_train) == 0 or len(X_topo_test) == 0:
        print("No hay suficientes casos con imagenes descargadas todavia.")
        return

    print(f"\nEntrenando CNN ({args.epochs} epocas) para extraer embeddings...")
    model, device = train_cnn(X_img_train, y_train, epochs=args.epochs, lr=1e-4, batch_size=8)

    print("Extrayendo embeddings de la CNN (1024-d, tras pooling global)...")
    emb_train = extract_cnn_embeddings(model, device, X_img_train)
    emb_test = extract_cnn_embeddings(model, device, X_img_test)

    X_fused_train = fuse_features(X_topo_train, emb_train)
    X_fused_test = fuse_features(X_topo_test, emb_test)
    print(f"Dimension fusionada: {X_fused_train.shape[1]} (topo={X_topo_train.shape[1]} + cnn={emb_train.shape[1]})")

    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    label = f"{args.classifier}{' + GridSearchCV' if args.tune else ''}"
    print(f"\n=== Comparacion TDA vs CNN-embeddings vs fusion ({label}) ===")
    print("(incluye AUC por CV de 5-fold sobre train: el test es pequeno y ruidoso, ver issue #77)")
    for name, Xtr, Xte in [
        ("TDA sola", X_topo_train, X_topo_test),
        ("CNN embeddings solos", emb_train, emb_test),
        ("Fusion TDA+CNN", X_fused_train, X_fused_test),
    ]:
        if args.tune:
            clf = tune_classifier(args.classifier, Xtr, y_train)
        else:
            clf = build_svm() if args.classifier == "svm" else build_random_forest()
            clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_score = clf.predict_proba(Xte)[:, 1]
        metrics = clinical_metrics(y_test, y_pred, y_score=y_score)
        cv_auc = cross_val_score(clone(clf), Xtr, y_train, cv=5, scoring="roc_auc")
        print(f"\n[{name}] (n_features={Xtr.shape[1]})")
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}")
        print(f"  auc_cv (5-fold, train): {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")


if __name__ == "__main__":
    main()
