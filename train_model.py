"""
train_model.py
Generates a synthetic training dataset and trains the CNN classifier.

Usage:
    python train_model.py [--epochs 50] [--samples 2000] [--output models/exoplanet_cnn.keras]
"""

import argparse
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_training_data(n_samples: int = 2000):
    """
    Generate a balanced synthetic training dataset.
    Returns X_seq, X_feat, y arrays.
    """
    from src.ingestion import generate_synthetic_lc
    from src.preprocessing import preprocess
    from src.bls_detection import run_bls, phase_fold
    from src.feature_extraction import build_feature_vector
    from src.classifier import LABELS

    signal_types = ["transit", "binary", "stellar_activity", "blend", "noise"]
    label_map = {
        "transit": 0, "binary": 1, "blend": 2,
        "stellar_activity": 3, "noise": 4
    }

    X_seq_list, X_feat_list, y_list = [], [], []
    n_per_class = n_samples // len(signal_types)

    for sig_type in signal_types:
        logger.info(f"Generating {n_per_class} samples for class: {sig_type}")
        label = label_map[sig_type]

        for i in range(n_per_class):
            try:
                np.random.seed(i * 100 + label)
                lc = generate_synthetic_lc(sig_type, n_points=500)
                lc_clean = preprocess(lc, method="savgol")
                t = lc_clean["time"]; f = lc_clean["flux"]; fe = lc_clean["flux_err"]
                period_max = min((t[-1] - t[0]) * 0.9, 27.0)
                bls = run_bls(t, f, fe, period_min=0.3, period_max=period_max, n_periods=3000)
                folded = phase_fold(t, f, bls["best_period"], bls["best_t0"])
                feat_vec, folded_1d, _ = build_feature_vector(lc_clean, bls, folded)

                X_seq_list.append(folded_1d)
                X_feat_list.append(feat_vec)
                y_list.append(label)
            except Exception as e:
                logger.debug(f"Sample {i} failed ({sig_type}): {e}")

    X_seq = np.array(X_seq_list).reshape(-1, 128, 1).astype(np.float32)
    X_feat = np.array(X_feat_list).astype(np.float32)
    y = np.array(y_list)

    # Normalise features
    feat_mean = X_feat.mean(axis=0)
    feat_std = X_feat.std(axis=0) + 1e-10
    X_feat = (X_feat - feat_mean) / feat_std

    logger.info(f"Dataset: {len(y)} samples, {X_feat.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Save normalisation stats
    os.makedirs("models", exist_ok=True)
    np.save("models/feat_mean.npy", feat_mean)
    np.save("models/feat_std.npy", feat_std)

    return X_seq, X_feat, y


def main():
    parser = argparse.ArgumentParser(description="Train the ExoplanetCNN classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", default="models/exoplanet_cnn.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logger.info("Generating synthetic training data...")
    X_seq, X_feat, y = generate_training_data(n_samples=args.samples)

    from src.classifier import build_model, train_model
    n_features = X_feat.shape[1]
    model = build_model(n_features=n_features)
    model.summary()

    logger.info(f"Training for {args.epochs} epochs...")
    history = train_model(
        model, X_seq, X_feat, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.output
    )

    # Print final metrics
    val_acc = max(history.history.get("val_accuracy", [0]))
    logger.info(f"\nBest validation accuracy: {val_acc*100:.1f}%")
    logger.info(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
