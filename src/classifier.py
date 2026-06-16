"""
CNN Classifier Module
Defines a 1D-CNN model that classifies TESS light curve signals into:
  0: Exoplanet Transit
  1: Eclipsing Binary
  2: Stellar Blend
  3: Stellar Activity
  4: Noise / Instrument

The model uses both:
  - A 1D CNN branch on the phase-folded transit profile (128 points)
  - A dense branch on the engineered feature vector

These are concatenated and fed to a final classification head.
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

LABELS = ["Exoplanet Transit", "Eclipsing Binary", "Stellar Blend", "Stellar Activity", "Noise"]
N_CLASSES = len(LABELS)


def build_model(n_features: int, sequence_len: int = 128):
    """
    Build the dual-input CNN model.

    Parameters
    ----------
    n_features : int
        Number of engineered features in the dense branch
    sequence_len : int
        Length of the 1D phase-folded transit input

    Returns
    -------
    tf.keras.Model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    # Branch 1: 1D CNN on phase-folded profile
    seq_input = keras.Input(shape=(sequence_len, 1), name="folded_transit")
    x = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same")(seq_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    cnn_out = layers.Dropout(0.3)(x)

    # Branch 2: Dense on engineered features
    feat_input = keras.Input(shape=(n_features,), name="features")
    y = layers.Dense(64, activation="relu")(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(32, activation="relu")(y)
    feat_out = layers.Dropout(0.2)(y)

    # Merge and classify
    merged = layers.Concatenate()([cnn_out, feat_out])
    z = layers.Dense(64, activation="relu")(merged)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation="relu")(z)
    output = layers.Dense(N_CLASSES, activation="softmax", name="classification")(z)

    model = keras.Model(inputs=[seq_input, feat_input], outputs=output, name="ExoplanetCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"Model built: {model.count_params():,} parameters")
    return model


def train_model(model, X_seq: np.ndarray, X_feat: np.ndarray, y: np.ndarray,
                epochs: int = 50, batch_size: int = 32,
                validation_split: float = 0.2,
                model_save_path: str = "models/exoplanet_cnn.keras"):
    """
    Train the CNN classifier on synthetic or labelled light curve data.

    Parameters
    ----------
    model : tf.keras.Model
    X_seq : np.ndarray, shape (N, 128, 1)
        Phase-folded transit profiles
    X_feat : np.ndarray, shape (N, n_features)
        Engineered feature vectors
    y : np.ndarray, shape (N,)
        Integer class labels (0-4)
    epochs : int
    batch_size : int
    validation_split : float
    model_save_path : str

    Returns
    -------
    history object
    """
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_accuracy"),
    ]

    history = model.fit(
        {"folded_transit": X_seq, "features": X_feat},
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training complete. Model saved to {model_save_path}")
    return history


def load_model(model_path: str):
    """Load a saved Keras model."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def predict(model, folded_1d: np.ndarray, feature_vector: np.ndarray) -> dict:
    """
    Run inference on a single light curve.

    Parameters
    ----------
    model : tf.keras.Model
    folded_1d : np.ndarray, shape (128,)
    feature_vector : np.ndarray, shape (n_features,)

    Returns
    -------
    dict with classification, confidence, and per-class probabilities
    """
    X_seq = folded_1d.reshape(1, -1, 1).astype(np.float32)
    X_feat = feature_vector.reshape(1, -1).astype(np.float32)

    probs = model.predict(
        {"folded_transit": X_seq, "features": X_feat}, verbose=0
    )[0]

    best_idx = int(np.argmax(probs))
    return {
        "classification": LABELS[best_idx],
        "confidence": float(probs[best_idx]),
        "probabilities": {label: float(p) for label, p in zip(LABELS, probs)},
        "class_idx": best_idx,
    }


def rule_based_classify(bls_result: dict, shape_features: dict) -> dict:
    """
    Fallback rule-based classifier (no trained model needed).
    Uses BLS SNR, depth, secondary eclipse, and shape metrics.

    Returns same format as predict().
    """
    snr = bls_result.get("snr", 0)
    depth = bls_result.get("best_depth", 0)
    depth_ratio = shape_features.get("depth_ratio", 0)
    shape_ratio = shape_features.get("shape_ratio", 1.5)
    asymmetry = shape_features.get("asymmetry", 0)
    oot_rms = shape_features.get("oot_rms", 0)

    probs = np.zeros(N_CLASSES)

    if snr < 6:
        if oot_rms > 0.005:
            probs[3] = 0.7; probs[4] = 0.3  # stellar activity or noise
        else:
            probs[4] = 0.85; probs[3] = 0.15  # noise
    elif depth_ratio > 0.3:
        probs[1] = 0.85; probs[2] = 0.15  # eclipsing binary
    elif depth > 0.05:
        probs[1] = 0.6; probs[2] = 0.4  # deep: binary or blend
    elif shape_ratio > 1.4 and asymmetry < 0.3:
        probs[0] = 0.85; probs[2] = 0.1; probs[1] = 0.05  # U-shaped: exoplanet
    elif oot_rms > 0.003:
        probs[3] = 0.75; probs[0] = 0.15; probs[4] = 0.1  # stellar activity
    else:
        probs[0] = 0.65; probs[2] = 0.2; probs[1] = 0.1; probs[4] = 0.05  # likely transit

    # Normalise
    probs /= probs.sum()
    best_idx = int(np.argmax(probs))

    return {
        "classification": LABELS[best_idx],
        "confidence": float(probs[best_idx]),
        "probabilities": {label: float(p) for label, p in zip(LABELS, probs)},
        "class_idx": best_idx,
        "method": "rule_based",
    }
