"""
Quick trainer for both Baseline CNN and EfficientNetB0.

CPU-optimised:
  - Baseline CNN  : 5,000-image stratified subset, 128×128, up to 20 epochs
  - EfficientNetB0: full dataset, 224×224, Stage-1 (10 ep) + Stage-2 (20 ep)

This script is designed to complete in a reasonable time on CPU-only hardware.
The Baseline CNN uses a subset purely to demonstrate the concept quickly;
EfficientNetB0 trains on the full dataset.

Usage:
    python -m src.train_quick              # baseline CNN + EfficientNetB0
    python -m src.train_quick --cnn-only   # baseline CNN only
    python -m src.train_quick --eff-only   # EfficientNetB0 only
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0

from src.data_loader import (
    IMG_SIZE,
    IMG_SIZE_BASELINE,
    TRAIN_DIR,
    VALID_DIR,
    get_class_names,
    save_class_labels,
    load_class_labels,
    _collect_split_paths,
    _stratified_split,
    _build_ds,
    _make_augmenter,
    AUTOTUNE,
)
from src.gpu_utils import setup_gpu

RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")


# ──────────────────────────────────────────────
# Subset sampler (for baseline CNN only)
# ──────────────────────────────────────────────
def _sample_paths(
    paths: list[str],
    labels: list[int],
    n_per_class: int,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Return at most n_per_class images per class, stratified."""
    rng = random.Random(seed)
    from collections import defaultdict
    buckets: dict[int, list[str]] = defaultdict(list)
    for p, lbl in zip(paths, labels):
        buckets[lbl].append(p)
    s_paths, s_labels = [], []
    for cls_id, cls_paths in sorted(buckets.items()):
        shuffled = cls_paths[:]
        rng.shuffle(shuffled)
        chosen = shuffled[:n_per_class]
        s_paths.extend(chosen)
        s_labels.extend([cls_id] * len(chosen))
    return s_paths, s_labels


# ──────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────
def build_baseline_cnn(num_classes: int) -> keras.Model:
    inp = keras.Input(shape=(IMG_SIZE_BASELINE[0], IMG_SIZE_BASELINE[1], 3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="BaselineCNN")


def build_efficientnet_stage1(num_classes: int) -> tuple[keras.Model, keras.Model]:
    backbone = EfficientNetB0(include_top=False, weights="imagenet",
                              input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    backbone.trainable = False
    inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = keras.layers.Rescaling(255.0)(inp)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inp, out, name="EfficientNetB0_Stage1")
    return model, backbone


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def _save_plots(history: dict, name: str, stage1_epochs: int = 0) -> None:
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    acc      = history.get("accuracy", [])
    val_acc  = history.get("val_accuracy", [])
    loss     = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    ep = range(1, len(acc) + 1)

    safe = name.lower().replace(" ", "_").replace("-", "_")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ep, acc, label="Train", linewidth=2)
    axes[0].plot(ep, val_acc, label="Validation", linewidth=2)
    if stage1_epochs > 0 and len(acc) > stage1_epochs:
        axes[0].axvline(x=stage1_epochs + 0.5, color="gray", linestyle="--",
                        alpha=0.7, label="Fine-tuning starts")
    axes[0].set_title(f"{name} — Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(ep, loss, label="Train", linewidth=2)
    axes[1].plot(ep, val_loss, label="Validation", linewidth=2)
    if stage1_epochs > 0 and len(loss) > stage1_epochs:
        axes[1].axvline(x=stage1_epochs + 0.5, color="gray", linestyle="--",
                        alpha=0.7, label="Fine-tuning starts")
    axes[1].set_title(f"{name} — Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(plots_dir / f"{safe}_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Plots → {plots_dir}/{safe}_training_curves.png")


# ──────────────────────────────────────────────
# Baseline CNN training
# ──────────────────────────────────────────────
def train_baseline_cnn(
    class_names: list[str],
    n_per_class: int = 200,
    batch_size: int = 32,
    max_epochs: int = 25,
) -> dict:
    num_classes = len(class_names)
    print("\n" + "=" * 60)
    print("  BASELINE CNN — Training")
    print(f"  Subset: {n_per_class} images/class × {num_classes} = {n_per_class*num_classes} train")
    print("=" * 60)

    train_paths, train_labels = _collect_split_paths(TRAIN_DIR, class_names)
    valid_paths, valid_labels = _collect_split_paths(VALID_DIR, class_names)
    (val_paths, val_labels), (test_paths, test_labels) = _stratified_split(
        valid_paths, valid_labels, val_fraction=0.80, seed=42)

    # Subsample training data for speed
    s_paths, s_labels = _sample_paths(train_paths, train_labels, n_per_class)
    print(f"  Sampled training : {len(s_paths):,}")
    print(f"  Validation       : {len(val_paths):,}")
    print(f"  Test (held-out)  : {len(test_paths):,}")

    train_ds = _build_ds(s_paths, s_labels, num_classes, IMG_SIZE_BASELINE,
                         batch_size, shuffle=True, augment=True)
    val_ds   = _build_ds(val_paths, val_labels, num_classes, IMG_SIZE_BASELINE,
                         batch_size, shuffle=False, augment=False)
    test_ds  = _build_ds(test_paths, test_labels, num_classes, IMG_SIZE_BASELINE,
                         batch_size, shuffle=False, augment=False)

    model = build_baseline_cnn(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  Parameters: {model.count_params():,}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "baseline_cnn.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(save_path), monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, epochs=max_epochs, validation_data=val_ds,
                     callbacks=callbacks, verbose=1)
    elapsed = time.time() - t0
    history = hist.history

    best_val = max(history["val_accuracy"])
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    print(f"\n  Best val accuracy : {best_val:.4f}  ({best_val:.2%})")
    print(f"  Test accuracy     : {test_acc:.4f}  ({test_acc:.2%})")
    print(f"  Training time     : {elapsed/60:.1f} min")

    # Save metrics
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "BaselineCNN",
        "epochs_trained": len(history["accuracy"]),
        "best_val_accuracy": float(best_val),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "training_time_min": round(elapsed / 60, 2),
        "total_parameters": model.count_params(),
        "img_size": list(IMG_SIZE_BASELINE),
        "batch_size": batch_size,
        "n_per_class_train": n_per_class,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    (metrics_dir / "baseline_cnn_metrics.json").write_text(json.dumps(summary, indent=2))
    _save_plots(history, "Baseline CNN")
    print(f"  Model saved → {save_path}")
    return summary


# ──────────────────────────────────────────────
# EfficientNetB0 training
# ──────────────────────────────────────────────
def train_efficientnet(
    class_names: list[str],
    batch_size: int = 16,
    stage1_epochs: int = 15,
    stage2_epochs: int = 25,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-5,
    unfreeze_last_n: int = 30,
    n_per_class: int = 0,   # 0 = use full training set
) -> dict:
    num_classes = len(class_names)
    print("\n" + "=" * 60)
    print("  EFFICIENTNETB0 — Two-Stage Transfer Learning")
    print("=" * 60)

    train_paths, train_labels = _collect_split_paths(TRAIN_DIR, class_names)
    valid_paths, valid_labels = _collect_split_paths(VALID_DIR, class_names)
    (val_paths, val_labels), (test_paths, test_labels) = _stratified_split(
        valid_paths, valid_labels, val_fraction=0.80, seed=42)

    if n_per_class > 0:
        train_paths, train_labels = _sample_paths(train_paths, train_labels, n_per_class)
        print(f"  Training images  : {len(train_paths):,} ({n_per_class}/class subset)")
    else:
        print(f"  Training images  : {len(train_paths):,} (full dataset)")
    print(f"  Validation images: {len(val_paths):,}")
    print(f"  Test images      : {len(test_paths):,}")

    train_ds = _build_ds(train_paths, train_labels, num_classes, IMG_SIZE,
                         batch_size, shuffle=True, augment=True)
    val_ds   = _build_ds(val_paths, val_labels, num_classes, IMG_SIZE,
                         batch_size, shuffle=False, augment=False)
    test_ds  = _build_ds(test_paths, test_labels, num_classes, IMG_SIZE,
                         batch_size, shuffle=False, augment=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stage1_path = MODELS_DIR / "efficientnet_stage1.keras"
    final_path  = MODELS_DIR / "efficientnet_finetuned.keras"

    # ── Stage 1: head-only ──────────────────────
    print(f"\n  Stage 1: Head-only (frozen backbone), lr={stage1_lr}, up to {stage1_epochs} epochs")
    model, backbone = build_efficientnet_stage1(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(stage1_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Trainable params (head only): {int(trainable):,} / {model.count_params():,}")

    cb_s1 = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(stage1_path), monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
    ]
    t1 = time.time()
    h1 = model.fit(train_ds, epochs=stage1_epochs, validation_data=val_ds,
                   callbacks=cb_s1, verbose=1)
    s1_time = time.time() - t1
    hist1 = h1.history
    best_s1 = max(hist1["val_accuracy"])
    print(f"\n  ✓ Stage 1 complete — best val_acc: {best_s1:.4f}  time: {s1_time/60:.1f} min")

    # ── Stage 2: fine-tuning ─────────────────────
    print(f"\n  Stage 2: Fine-tuning (last {unfreeze_last_n} layers), lr={stage2_lr}, up to {stage2_epochs} epochs")
    backbone.trainable = True
    total_layers = len(backbone.layers)
    freeze_until = total_layers - unfreeze_last_n
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    for layer in backbone.layers[freeze_until:]:
        layer.trainable = True

    # Keep BatchNormalization layers frozen during fine-tuning
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization) or "batchnormalization" in layer.__class__.__name__.lower():
            layer.trainable = False
    unfrozen = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Backbone: {total_layers} total layers, {unfrozen} unfrozen")

    model.compile(
        optimizer=keras.optimizers.Adam(stage2_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    trainable2 = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Trainable params (with backbone): {int(trainable2):,}")

    cb_s2 = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=4, min_lr=1e-8, verbose=1),
        keras.callbacks.ModelCheckpoint(str(final_path), monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
    ]
    t2 = time.time()
    h2 = model.fit(train_ds, epochs=stage2_epochs, validation_data=val_ds,
                   callbacks=cb_s2, verbose=1)
    s2_time = time.time() - t2
    hist2 = h2.history
    best_s2 = max(hist2["val_accuracy"])
    print(f"\n  ✓ Stage 2 complete — best val_acc: {best_s2:.4f}  time: {s2_time/60:.1f} min")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test accuracy: {test_acc:.4f}  ({test_acc:.2%})")

    # Combined history for plots
    combined = {
        "accuracy":     list(hist1["accuracy"])     + list(hist2["accuracy"]),
        "val_accuracy": list(hist1["val_accuracy"]) + list(hist2["val_accuracy"]),
        "loss":         list(hist1["loss"])          + list(hist2["loss"]),
        "val_loss":     list(hist1["val_loss"])      + list(hist2["val_loss"]),
    }

    # Save metrics
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "EfficientNetB0_FineTuned",
        "stage1_best_val_accuracy": float(best_s1),
        "stage2_best_val_accuracy": float(best_s2),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "stage1_epochs_trained": len(hist1["accuracy"]),
        "stage2_epochs_trained": len(hist2["accuracy"]),
        "total_training_time_min": round((s1_time + s2_time) / 60, 2),
        "stage1_lr": stage1_lr,
        "stage2_lr": stage2_lr,
        "batch_size": batch_size,
        "unfreeze_last_n": unfreeze_last_n,
        "stage1_history": {k: [float(v) for v in vals] for k, vals in hist1.items()},
        "stage2_history": {k: [float(v) for v in vals] for k, vals in hist2.items()},
    }
    (metrics_dir / "efficientnet_metrics.json").write_text(json.dumps(summary, indent=2))
    _save_plots(combined, "EfficientNetB0", stage1_epochs=len(hist1["accuracy"]))
    print(f"  Model saved → {final_path}")
    return summary


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Quick CPU-friendly training for both models")
    parser.add_argument("--cnn-only", action="store_true", help="Train baseline CNN only")
    parser.add_argument("--eff-only", action="store_true", help="Train EfficientNetB0 only")
    parser.add_argument("--cnn-subset", type=int, default=200,
                        help="Images per class for baseline CNN subset (default: 200)")
    parser.add_argument("--eff-subset", type=int, default=400,
                        help="Images per class for EfficientNet (0=full dataset, default: 400)")
    parser.add_argument("--eff-batch", type=int, default=32,
                        help="Batch size for EfficientNetB0 (default: 32)")
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2-epochs", type=int, default=25)
    args = parser.parse_args()

    # ── Initialize GPU ────────────────────────────
    setup_gpu(verbose=True)

    class_names = get_class_names()
    save_class_labels(class_names)

    total_start = time.time()

    if not args.eff_only:
        cnn_summary = train_baseline_cnn(
            class_names, n_per_class=args.cnn_subset, batch_size=32, max_epochs=25
        )

    if not args.cnn_only:
        eff_summary = train_efficientnet(
            class_names,
            batch_size=args.eff_batch,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            n_per_class=args.eff_subset,
        )

    total_time = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  All training complete in {total_time:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
