"""
Hyperparameter Experiment — EfficientNetB0 Classification Head.

Runs a small grid search over learning rate and dropout, using VALIDATION
performance for model selection. The test set is NEVER touched here.

Hyperparameters explored:
    Stage-1 learning rate : [1e-3, 5e-4]
    Dropout               : [0.2, 0.3, 0.5]

Each configuration is trained for a fixed 10 epochs (Stage 1 only).
The configuration with the best val_accuracy is reported.

Usage:
    python -m src.hyperparam_search
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0

from src.data_loader import (
    IMG_SIZE,
    TRAIN_DIR,
    VALID_DIR,
    get_class_names,
    save_class_labels,
    _collect_split_paths,
    _stratified_split,
    _build_ds,
    _sample_paths,
)

RESULTS_DIR = Path("results")
METRICS_DIR = RESULTS_DIR / "metrics"


def _build_model(num_classes: int, dropout: float, lr: float) -> keras.Model:
    backbone = EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    backbone.trainable = False
    inp = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = keras.layers.Rescaling(255.0)(inp)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    class_names = get_class_names()
    save_class_labels(class_names)
    num_classes = len(class_names)

    # Prepare small training subset (150/class = 3750 images) for speed
    n_per_class = 150
    train_paths, train_labels = _collect_split_paths(TRAIN_DIR, class_names)
    valid_paths, valid_labels = _collect_split_paths(VALID_DIR, class_names)
    (val_paths, val_labels), _ = _stratified_split(valid_paths, valid_labels, 0.80, 42)

    from src.train_quick import _sample_paths
    s_paths, s_labels = _sample_paths(train_paths, train_labels, n_per_class)

    BATCH = 32
    EPOCHS = 10

    train_ds = _build_ds(s_paths, s_labels, num_classes, IMG_SIZE,
                         BATCH, shuffle=True, augment=True)
    val_ds   = _build_ds(val_paths, val_labels, num_classes, IMG_SIZE,
                         BATCH, shuffle=False, augment=False)

    # Hyperparameter grid
    lrs      = [1e-3, 5e-4]
    dropouts = [0.2, 0.3, 0.5]

    results = []

    print("\n" + "=" * 65)
    print("  HYPERPARAMETER SEARCH — EfficientNetB0 Head")
    print(f"  Grid: {len(lrs)} LRs × {len(dropouts)} dropouts = {len(lrs)*len(dropouts)} configs")
    print(f"  Training: {len(s_paths)} images, {EPOCHS} epochs each")
    print("=" * 65)

    for lr, dropout in product(lrs, dropouts):
        label = f"lr={lr:.0e}, dropout={dropout}"
        print(f"\n  Config: {label}")
        t0 = time.time()

        tf.keras.backend.clear_session()
        model = _build_model(num_classes, dropout, lr)

        cb = [keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True)]
        hist = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                         callbacks=cb, verbose=0)
        elapsed = time.time() - t0

        best_val = max(hist.history["val_accuracy"])
        print(f"    Best val_accuracy: {best_val:.4f}  ({best_val:.2%})  [{elapsed:.0f}s]")
        results.append({
            "lr": lr,
            "dropout": dropout,
            "best_val_accuracy": float(best_val),
            "epochs_trained": len(hist.history["accuracy"]),
            "time_sec": round(elapsed, 1),
        })

    # Sort by best val accuracy
    results.sort(key=lambda r: r["best_val_accuracy"], reverse=True)

    print("\n" + "=" * 65)
    print("  RESULTS (sorted by val accuracy)")
    print("=" * 65)
    print(f"  {'LR':<10} {'Dropout':<10} {'Val Accuracy':<15} {'Epochs':<10}")
    print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*10}")
    for r in results:
        flag = " ← BEST" if r == results[0] else ""
        print(f"  {r['lr']:<10.0e} {r['dropout']:<10.1f} "
              f"{r['best_val_accuracy']:<15.4f} {r['epochs_trained']:<10}{flag}")

    best = results[0]
    print(f"\n  Best configuration:")
    print(f"    Learning rate : {best['lr']}")
    print(f"    Dropout       : {best['dropout']}")
    print(f"    Val accuracy  : {best['best_val_accuracy']:.4f}  ({best['best_val_accuracy']:.2%})")

    # Save results
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "hyperparam_search_results.json").write_text(
        json.dumps({"grid": results, "best": best}, indent=2)
    )
    print(f"\n  Results saved → {METRICS_DIR}/hyperparam_search_results.json")
    print("\n✓ Hyperparameter search complete.")


if __name__ == "__main__":
    main()
