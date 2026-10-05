"""
Baseline CNN from scratch — Bird Species Classification.

Architecture (intentionally simple for educational demonstration):
  Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  Block 4: Conv2D(256, 3×3) → BatchNorm → ReLU → GlobalAvgPool
  Head:    Dropout(0.5) → Dense(25, softmax)

Why this is a baseline:
  - No pretrained weights; learns entirely from ~30K images
  - Smaller input (128×128) for manageable CPU training time
  - Demonstrates overfitting problem that motivates transfer learning

Usage:
    python -m src.train_baseline_cnn
    python -m src.train_baseline_cnn --epochs 30 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless/server
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras

from src.data_loader import (
    IMG_SIZE_BASELINE,
    build_datasets,
    get_class_names,
    save_class_labels,
)
from src.gpu_utils import setup_gpu

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


# ──────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────
def build_baseline_cnn(num_classes: int, input_size: tuple[int, int] = IMG_SIZE_BASELINE) -> keras.Model:
    """
    Build a simple CNN baseline demonstrating core CNN components.

    Layers explained:
        Conv2D        — learns spatial feature detectors (filters)
        BatchNorm     — normalizes activations, speeds up training
        ReLU          — non-linear activation (via 'relu' in Conv2D)
        MaxPooling2D  — spatial down-sampling, translation invariance
        GlobalAvgPool — replaces Flatten; averages spatial features → compact vector
        Dropout       — randomly zeros neurons during training → reduces overfitting
        Dense(softmax)— output layer with probability distribution over classes
    """
    inp = keras.Input(shape=(input_size[0], input_size[1], 3), name="input_image")

    # Block 1 ── low-level edges and textures
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inp)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2 ── simple shapes and colour blobs
    x = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = keras.layers.BatchNormalization(name="bn2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3 ── parts (wings, beak, tail)
    x = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = keras.layers.BatchNormalization(name="bn3")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4 ── higher-level plumage patterns
    x = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = keras.layers.BatchNormalization(name="bn4")(x)
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Classification head
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BaselineCNN")
    return model


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def _plot_history(history: dict, save_dir: Path, prefix: str = "baseline_cnn") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history["accuracy"]) + 1)

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, history["accuracy"], label="Train Accuracy", linewidth=2)
    ax.plot(epochs_range, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    ax.set_title("Baseline CNN — Training vs Validation Accuracy", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}_accuracy.png", dpi=150)
    plt.close(fig)

    # Loss plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs_range, history["loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs_range, history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_title("Baseline CNN — Training vs Validation Loss", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}_loss.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved → {save_dir}/{prefix}_accuracy.png  &  {prefix}_loss.png")


# ──────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline CNN")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Maximum training epochs (early stopping will cut this short)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "baseline_cnn.keras")
    args = parser.parse_args()

    # ── Initialize GPU ────────────────────────────
    setup_gpu(verbose=True)

    # ── Build data pipeline ──────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE CNN — Data Loading")
    print("=" * 60)
    train_ds, val_ds, test_ds, class_names = build_datasets(
        img_size=IMG_SIZE_BASELINE,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)

    # ── Build model ──────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE CNN — Model Architecture")
    print("=" * 60)
    model = build_baseline_cnn(num_classes=num_classes, input_size=IMG_SIZE_BASELINE)
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    # ── Compile ──────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_path),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    # ── Train ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE CNN — Training")
    print("=" * 60)
    t0 = time.time()
    hist = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    history = hist.history

    # ── Evaluate on validation ───────────────────
    print("\n" + "=" * 60)
    print("  BASELINE CNN — Validation Results")
    print("=" * 60)
    best_val_acc = max(history["val_accuracy"])
    best_val_loss = min(history["val_loss"])
    epochs_trained = len(history["accuracy"])
    print(f"Epochs trained   : {epochs_trained}")
    print(f"Best val accuracy: {best_val_acc:.4f}  ({best_val_acc:.2%})")
    print(f"Best val loss    : {best_val_loss:.4f}")
    print(f"Training time    : {elapsed/60:.1f} min")

    # ── Quick test evaluation ────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy    : {test_acc:.4f}  ({test_acc:.2%})")
    print(f"Test loss        : {test_loss:.4f}")

    # ── Save history and metrics ─────────────────
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "BaselineCNN",
        "epochs_trained": epochs_trained,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "training_time_min": round(elapsed / 60, 2),
        "total_parameters": total_params,
        "img_size": list(IMG_SIZE_BASELINE),
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    (metrics_dir / "baseline_cnn_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"Metrics saved → {metrics_dir}/baseline_cnn_metrics.json")

    # ── Plots ────────────────────────────────────
    _plot_history(history, RESULTS_DIR / "plots", prefix="baseline_cnn")

    print(f"\nModel saved → {args.model_path}")
    print("\n✓ Baseline CNN training complete.")


if __name__ == "__main__":
    main()
