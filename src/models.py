"""Clasificadores para las caracteristicas topologicas y de deep learning."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm() -> Pipeline:
    """SVM con RBF sobre caracteristicas topologicas normalizadas."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
        ]
    )


def build_random_forest(n_estimators: int = 300, seed: int = 42) -> RandomForestClassifier:
    """Random Forest sobre caracteristicas topologicas."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def build_cnn(input_shape=(224, 224, 3), n_classes: int = 2):
    """CNN de referencia con transfer learning (DenseNet121).

    Se congela la base y se entrena una cabeza ligera. Descongela las ultimas
    capas para fine-tuning en una segunda fase.
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.applications import DenseNet121

    base = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fuse_features(topo: np.ndarray, cnn_embeddings: np.ndarray) -> np.ndarray:
    """Concatena caracteristicas topologicas y embeddings de la CNN."""
    return np.hstack([topo, cnn_embeddings])
