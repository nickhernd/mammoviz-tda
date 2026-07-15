"""Clasificadores para las caracteristicas topologicas y de deep learning.

Cuarta etapa del pipeline. Reune los tres enfoques que compara el TFG:
  1) TOPOLOGICO: un clasificador clasico (SVM o Random Forest) que aprende
     directamente sobre los vectores de caracteristicas topologicas.
  2) DEEP LEARNING: una CNN de referencia (DenseNet121 con transfer learning)
     que aprende sobre las imagenes.
  3) FUSION: combinar ambas fuentes de informacion.

Nota de diseno: las dependencias pesadas de deep learning (TensorFlow) se
importan DENTRO de la funcion que las necesita, para que importar este modulo
sea rapido y no obligue a tener TF instalado si solo se usa la via topologica.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm() -> Pipeline:
    """SVM con kernel RBF sobre caracteristicas topologicas normalizadas.

    Se devuelve un Pipeline de scikit-learn con dos pasos encadenados:
      - StandardScaler: estandariza cada caracteristica (media 0, varianza 1);
        las SVM son sensibles a la escala, asi que este paso es importante.
      - SVC(probability=True): SVM que ademas estima probabilidades, necesarias
        para calcular la curva ROC/AUC en la evaluacion.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
        ]
    )


def build_random_forest(n_estimators: int = 300, seed: int = 42) -> RandomForestClassifier:
    """Random Forest sobre caracteristicas topologicas.

    Robusto, poco sensible a la escala y con importancia de caracteristicas
    interpretable. ``class_weight="balanced"`` compensa el desbalance de clases
    (suele haber mas casos benignos que malignos), dando mas peso a la minoritaria.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,  # reproducibilidad
        n_jobs=-1,          # usa todos los nucleos
    )


def build_cnn(input_shape=(224, 224, 3), n_classes: int = 2):
    """CNN de referencia con transfer learning (DenseNet121).

    Estrategia habitual en imagen medica, donde hay pocos datos etiquetados:
      1) Partimos de DenseNet121 preentrenada en ImageNet (millones de imagenes).
      2) CONGELAMOS la base (base.trainable = False) y entrenamos solo una
         cabeza ligera nueva. Asi reaprovechamos las caracteristicas visuales
         genericas sin sobreajustar.
      3) (Fase 2, opcional) se descongelan las ultimas capas para 'fine-tuning'.

    La cabeza: pooling global -> normalizacion -> dropout (regularizacion) ->
    capa densa softmax con una salida por clase.
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import DenseNet121

    # Base preentrenada SIN la capa de clasificacion original (include_top=False).
    base = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # congelamos: no se actualizan sus pesos en la fase 1

    # Definicion funcional del modelo (entrada -> base -> cabeza -> salida).
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)          # training=False: BatchNorm en modo inferencia
    x = layers.GlobalAveragePooling2D()(x)    # resume cada mapa de activacion en un numero
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)                # apaga el 40% de neuronas -> reduce sobreajuste
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # learning rate pequeno para fine-tuning
        loss="sparse_categorical_crossentropy",     # etiquetas enteras (0/1), no one-hot
        metrics=["accuracy"],
    )
    return model


def fuse_features(topo: np.ndarray, cnn_embeddings: np.ndarray) -> np.ndarray:
    """Concatena caracteristicas topologicas y embeddings de la CNN (enfoque 3).

    'Fusion tardia' (late fusion): se extrae por separado el vector topologico y
    el vector de activaciones de la CNN, y se pegan en un unico vector que
    alimenta a un clasificador clasico. Permite medir si la topologia APORTA
    algo por encima de lo que ya captura la red.
    """
    return np.hstack([topo, cnn_embeddings])
