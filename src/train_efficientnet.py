"""
EfficientNetB0 Transfer Learning — Bird Species Classification.

Two-stage training strategy
───────────────────────────
Stage 1 — Head-only training (feature extraction):
    • Load EfficientNetB0 with ImageNet weights
    • Freeze ALL backbone layers
    • Add a new classification head
    • Train at lr=1e-3 for up to 20 epochs
    • Only the new head parameters are updated (~100K params vs 4M total)
    • Purpose: adapt the head to bird species without disturbing pretrained features

Stage 2 — Fine-tuning:
    • Unfreeze the last 30 layers of the backbone
    • Train at a MUCH smaller lr=1e-5 (prevents catastrophic forgetting)
    • Run for up to 40 more epochs with early stopping
    • Purpose: fine-tune high-level features for Indian bird plumage patterns

Why EfficientNetB0?
    • Trained on 1.28M ImageNet images → rich low/mid/high-level visual features
    • Efficient architecture: compound scaling of depth, width, resolution
    • ~4M parameters: fast enough for CPU training
    • Top-1 ImageNet accuracy: 77.1% → strong feature extractor

Why NOT train from scratch?
    • ~30K bird images is insufficient to train a deep CNN (4M params) without
      severe overfitting
    • Transfer learning reuses features learned from millions of images

Usage:
    python -m src.train_efficientnet                 # full run
    python -m src.train_efficientnet --stage1-only   # stage 1 only
    python -m src.train_efficientnet --skip-stage1   # fine-tune a saved stage-1 model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0

from src.data_loader import IMG_SIZE, build_datasets

RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


# ──────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────
def build_efficientnet_stage1(num_classes: int, input_size: tuple[int, int] = IMG_SIZE) -> keras.Model:
    """
    EfficientNetB0 with frozen backbone + trainable classification head.

    Classification head:
        GlobalAveragePooling2D  — pool spatial features → fixed-size vector
        BatchNormalization      — normalize pooled features
        Dropout(0.3)            — light regularization
        Dense(256, relu)        — intermediate representation
        Dropout(0.3)            — further regularization
        Dense(num_classes, softmax) — probability over each bird species

    EfficientNetB0 preprocess_input is applied inside the model so that
    the same [0, 1] images can be fed directly.
    """
    # Note: EfficientNetB0 expects pixel values in [0, 255] with its own
    # preprocess_input, OR we can pass [0,1] scaled images directly since
    # the Keras implementation rescales internally. We pass [0,1] and
    # include a Rescaling layer to undo the /255 our data loader applies.
    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size[0], input_size[1], 3),
    )
    backbone.trainable = False  # freeze all backbone layers

    inp = keras.Input(shape=(input_size[0], input_size[1], 3), name="input_image")

    # EfficientNetB0's preprocess_input maps [0,255] → internal. Since our
    # loader provides [0,1], rescale back to [0,255] first.
    x = keras.layers.Rescaling(255.0, name="rescale_to_255")(inp)

    x = backbone(x, training=False)   # training=False keeps BN layers in inference mode

    # Classification head
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dropout(0.3, name="drop1")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense256")(x)
    x = keras.layers.Dropout(0.3, name="drop2")(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="EfficientNetB0_Stage1")
    return model, backbone


def prepare_for_finetuning(
    model: keras.Model,
    backbone: keras.Model,
    unfreeze_last_n: int = 30,
    lr: float = 1e-5,
) -> keras.Model:
    """
    Unfreeze the last `unfreeze_last_n` backbone layers for fine-tuning.

    Fine-tuning with a VERY small learning rate prevents catastrophic
    forgetting — the risk of erasing valuable pretrained features.
    """
    backbone.trainable = True

    # Keep early backbone layers frozen (they learn generic features: edges,
    # textures, colours that are useful for any image task)
    total_layers = len(backbone.layers)
    freeze_until = total_layers - unfreeze_last_n
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    for layer in backbone.layers[freeze_until:]:
        layer.trainable = True

    # Keep BatchNormalization layers frozen during fine-tuning to preserve ImageNet normalization stats
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization) or "batchnormalization" in layer.__class__.__name__.lower():
            layer.trainable = False

    trainable_count = sum(1 for l in backbone.layers if l.trainable)
    print(f"Backbone layers: {total_layers} total, {trainable_count} unfrozen for fine-tuning")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def _plot_history(
    hist1: dict | None,
    hist2: dict | None,
    save_dir: Path,
) -> None:
    """Plot combined Stage1 + Stage2 training curves."""
    save_dir.mkdir(parents=True, exist_ok=True)

    def _cat(key: str) -> list:
        h1 = list(hist1[key]) if (hist1 and key in hist1) else []
        h2 = list(hist2[key]) if (hist2 and key in hist2) else []
        return h1 + h2

    acc    = _cat("accuracy")
    val_acc = _cat("val_accuracy")
    loss   = _cat("loss")
    val_loss = _cat("val_loss")
    epochs_range = range(1, len(acc) + 1)
    stage1_end = len(hist1["accuracy"]) if hist1 else 0

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_range, acc, label="Train Accuracy", linewidth=2)
    ax.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
    if stage1_end > 0 and hist2:
        ax.axvline(x=stage1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Fine-tuning starts")
    ax.set_title("EfficientNetB0 — Training vs Validation Accuracy", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_dir / "efficientnet_accuracy.png", dpi=150)
    plt.close(fig)

    # Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_range, loss, label="Train Loss", linewidth=2)
    ax.plot(epochs_range, val_loss, label="Validation Loss", linewidth=2)
    if stage1_end > 0 and hist2:
        ax.axvline(x=stage1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Fine-tuning starts")
    ax.set_title("EfficientNetB0 — Training vs Validation Loss", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_dir / "efficientnet_loss.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved → {save_dir}/efficientnet_accuracy.png  &  efficientnet_loss.png")


# ──────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 for bird classification")
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfreeze-layers", type=int, default=30,
                        help="Number of backbone layers to unfreeze in Stage 2")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Stop after Stage 1 (head-only training)")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip Stage 1; load stage-1 checkpoint and go straight to Stage 2")
    parser.add_argument("--stage1-model", type=Path, default=MODELS_DIR / "efficientnet_stage1.keras")
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "efficientnet_finetuned.keras")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build data pipeline ──────────────────────
    print("\n" + "=" * 60)
    print("  EfficientNetB0 — Data Loading")
    print("=" * 60)
    train_ds, val_ds, test_ds, class_names = build_datasets(
        img_size=IMG_SIZE,
        batch_size=args.batch_size,
    )
    num_classes = len(class_names)

    hist1, hist2 = None, None
    total_start = time.time()

    # ── Stage 1: Head-only training ──────────────
    if not args.skip_stage1:
        print("\n" + "=" * 60)
        print("  EfficientNetB0 — STAGE 1: Head-Only Training")
        print("  (Backbone frozen — only classification head learns)")
        print("=" * 60)

        model, backbone = build_efficientnet_stage1(num_classes, IMG_SIZE)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.stage1_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        total_params = model.count_params()
        print(f"Total parameters     : {total_params:,}")
        print(f"Trainable parameters : {int(trainable_params):,}  (head only)")

        callbacks_s1 = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(args.stage1_model),
                monitor="val_accuracy", save_best_only=True, verbose=1,
            ),
        ]

        t1 = time.time()
        hist_obj1 = model.fit(
            train_ds,
            epochs=args.stage1_epochs,
            validation_data=val_ds,
            callbacks=callbacks_s1,
            verbose=1,
        )
        stage1_time = time.time() - t1
        hist1 = hist_obj1.history

        best_s1_val = max(hist1["val_accuracy"])
        print(f"\n✓ Stage 1 complete — Best val accuracy: {best_s1_val:.4f} ({best_s1_val:.2%})")
        print(f"  Stage 1 time: {stage1_time/60:.1f} min")

        if args.stage1_only:
            (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
            summary = {
                "stage": "stage1_only",
                "best_val_accuracy": float(best_s1_val),
                "history": {k: [float(v) for v in vals] for k, vals in hist1.items()},
            }
            (RESULTS_DIR / "metrics" / "efficientnet_stage1_metrics.json").write_text(
                json.dumps(summary, indent=2)
            )
            _plot_history(hist1, None, RESULTS_DIR / "plots")
            print(f"Stage-1 model saved → {args.stage1_model}")
            return

    else:
        print(f"\nSkipping Stage 1 — loading model from {args.stage1_model}")
        model = keras.models.load_model(str(args.stage1_model))
        # Recover backbone reference for fine-tuning
        backbone = None
        for layer in model.layers:
            if isinstance(layer, keras.Model) or "efficientnet" in layer.name.lower() or "functional" in layer.name.lower():
                backbone = layer
                break
        if backbone is None:
            raise RuntimeError("Cannot locate EfficientNetB0 backbone in loaded model")

    # ── Stage 2: Fine-tuning ─────────────────────
    print("\n" + "=" * 60)
    print("  EfficientNetB0 — STAGE 2: Fine-Tuning")
    print(f"  (Unfreezing last {args.unfreeze_layers} backbone layers, lr={args.stage2_lr:.0e})")
    print("=" * 60)

    model = prepare_for_finetuning(
        model, backbone,
        unfreeze_last_n=args.unfreeze_layers,
        lr=args.stage2_lr,
    )

    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f"Trainable parameters after unfreezing: {int(trainable_params):,}")

    callbacks_s2 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=4, min_lr=1e-8, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_path),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    t2 = time.time()
    hist_obj2 = model.fit(
        train_ds,
        epochs=args.stage2_epochs,
        validation_data=val_ds,
        callbacks=callbacks_s2,
        verbose=1,
    )
    stage2_time = time.time() - t2
    hist2 = hist_obj2.history

    total_time = time.time() - total_start
    best_s2_val = max(hist2["val_accuracy"])
    best_s2_loss = min(hist2["val_loss"])

    print(f"\n✓ Stage 2 complete — Best val accuracy: {best_s2_val:.4f} ({best_s2_val:.2%})")
    print(f"  Stage 2 time: {stage2_time/60:.1f} min")
    print(f"  Total training time: {total_time/60:.1f} min")

    # ── Test evaluation ───────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy : {test_acc:.4f}  ({test_acc:.2%})")
    print(f"Test loss     : {test_loss:.4f}")

    # ── Save metrics ──────────────────────────────
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "EfficientNetB0_FineTuned",
        "stage1_best_val_accuracy": float(max(hist1["val_accuracy"])) if hist1 else None,
        "stage2_best_val_accuracy": float(best_s2_val),
        "stage2_best_val_loss": float(best_s2_loss),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "stage2_epochs_trained": len(hist2["accuracy"]),
        "unfreeze_last_n": args.unfreeze_layers,
        "stage1_lr": args.stage1_lr,
        "stage2_lr": args.stage2_lr,
        "batch_size": args.batch_size,
        "total_training_time_min": round(total_time / 60, 2),
        "stage1_history": {k: [float(v) for v in vals] for k, vals in hist1.items()} if hist1 else {},
        "stage2_history": {k: [float(v) for v in vals] for k, vals in hist2.items()},
    }
    (metrics_dir / "efficientnet_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Metrics saved → {metrics_dir}/efficientnet_metrics.json")

    # ── Plots ─────────────────────────────────────
    _plot_history(hist1, hist2, RESULTS_DIR / "plots")

    print(f"\nFine-tuned model saved → {args.model_path}")
    print("\n✓ EfficientNetB0 training complete.")


if __name__ == "__main__":
    main()
