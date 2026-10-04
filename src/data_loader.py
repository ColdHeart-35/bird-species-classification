"""
TensorFlow/Keras data pipeline for the Birds-25 dataset.

Responsibilities:
  - Discover class labels from train/ folder names (sorted alphabetically → class IDs)
  - Build tf.data.Dataset for train, validation, and test splits
  - Apply data augmentation ONLY to the training split
  - Save/load the class-label mapping as JSON for inference
  - Split the official valid/ folder 80/20 into val + test (seeded, stratified)

Usage:
    from src.data_loader import build_datasets, IMG_SIZE
    train_ds, val_ds, test_ds, class_names = build_datasets()
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────
# Global configuration constants
# ──────────────────────────────────────────────
IMG_SIZE: tuple[int, int] = (224, 224)      # EfficientNetB0 native resolution
IMG_SIZE_BASELINE: tuple[int, int] = (128, 128)  # Smaller for baseline CNN speed

DATASET_ROOT = Path("data/archive/Birds_25")
TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
LABEL_MAP_PATH = Path("models/class_labels.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

AUTOTUNE = tf.data.AUTOTUNE

# ──────────────────────────────────────────────
# Augmentation layer (training only)
# ──────────────────────────────────────────────
def _make_augmenter() -> tf.keras.Sequential:
    """Return a Keras augmentation pipeline applied during training."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),          # ±~15° rotation
        tf.keras.layers.RandomZoom((-0.05, 0.10)),     # slight zoom
        tf.keras.layers.RandomTranslation(0.05, 0.05), # small shift
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)), # mild brightness jitter
        tf.keras.layers.RandomContrast(0.10),          # mild contrast jitter
    ], name="augmentation")


# ──────────────────────────────────────────────
# Image file discovery helpers
# ──────────────────────────────────────────────
def _collect_split_paths(
    split_dir: Path, class_names: list[str]
) -> tuple[list[str], list[int]]:
    """Return (image_paths, labels) for a given split directory."""
    paths: list[str] = []
    labels: list[int] = []
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    for species_dir in sorted(split_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        class_id = name_to_id[species_dir.name]
        imgs = [
            str(p)
            for p in sorted(species_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        paths.extend(imgs)
        labels.extend([class_id] * len(imgs))
    return paths, labels


def _stratified_split(
    paths: list[str],
    labels: list[int],
    val_fraction: float = 0.80,
    seed: int = 42,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    """
    Split paths/labels into two stratified subsets.

    val_fraction of each class → first subset (validation)
    remainder                  → second subset (test)
    """
    rng = random.Random(seed)
    from collections import defaultdict

    class_buckets: dict[int, list[str]] = defaultdict(list)
    for p, lbl in zip(paths, labels):
        class_buckets[lbl].append(p)

    val_paths: list[str] = []
    val_labels: list[int] = []
    test_paths: list[str] = []
    test_labels: list[int] = []

    for class_id, class_paths in sorted(class_buckets.items()):
        shuffled = class_paths[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_fraction)
        val_paths.extend(shuffled[:n_val])
        val_labels.extend([class_id] * n_val)
        test_paths.extend(shuffled[n_val:])
        test_labels.extend([class_id] * (len(shuffled) - n_val))

    return (val_paths, val_labels), (test_paths, test_labels)


# ──────────────────────────────────────────────
# tf.data pipeline builders
# ──────────────────────────────────────────────
def _load_and_preprocess(
    path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor]:
    """Read JPEG/PNG, resize, and scale to [0, 1]."""
    raw = tf.io.read_file(path)
    # decode_image handles JPEG, PNG, BMP, GIF
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0   # scale [0, 1]
    return image, label


def _build_ds(
    paths: list[str],
    labels: list[int],
    num_classes: int,
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int = 42,
) -> tf.data.Dataset:
    """Construct a batched, optionally augmented tf.data.Dataset."""
    augmenter = _make_augmenter() if augment else None
    n = len(paths)

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.one_hot(labels, depth=num_classes)
    )
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(n, 10_000), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, lbl: _load_and_preprocess(p, lbl, img_size),
        num_parallel_calls=AUTOTUNE,
    )

    if augment:
        ds = ds.map(
            lambda img, lbl: (tf.clip_by_value(augmenter(img, training=True), 0.0, 1.0), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def get_class_names() -> list[str]:
    """Return sorted class names from the train/ directory."""
    return sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())


def save_class_labels(class_names: list[str], path: Path = LABEL_MAP_PATH) -> None:
    """Save class-name list as JSON (index → name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {str(i): name for i, name in enumerate(class_names)}
    path.write_text(json.dumps(mapping, indent=2))
    print(f"Class labels saved → {path}")


def load_class_labels(path: Path = LABEL_MAP_PATH) -> dict[int, str]:
    """Load class label mapping from JSON."""
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def build_datasets(
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = 32,
    val_fraction: float = 0.80,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Build train / validation / test tf.data.Datasets.

    Split strategy
    ──────────────
    • All of TRAIN_DIR/   →  training set  (with augmentation)
    • VALID_DIR/ 80%      →  validation set (no augmentation)
    • VALID_DIR/ 20%      →  test set       (no augmentation, held out until final eval)

    Returns
    ───────
    train_ds, val_ds, test_ds, class_names
    """
    class_names = get_class_names()
    num_classes = len(class_names)
    save_class_labels(class_names)

    # ── Training split ──────────────────────────
    train_paths, train_labels = _collect_split_paths(TRAIN_DIR, class_names)

    # ── Valid/ → split into val + test ──────────
    valid_paths, valid_labels = _collect_split_paths(VALID_DIR, class_names)
    (val_paths, val_labels), (test_paths, test_labels) = _stratified_split(
        valid_paths, valid_labels, val_fraction=val_fraction, seed=seed
    )

    print(f"Classes         : {num_classes}")
    print(f"Training images : {len(train_paths):,}")
    print(f"Validation images: {len(val_paths):,}")
    print(f"Test images     : {len(test_paths):,}")
    print(f"Image size      : {img_size[0]}×{img_size[1]}")
    print(f"Batch size      : {batch_size}")

    train_ds = _build_ds(
        train_paths, train_labels, num_classes, img_size,
        batch_size=batch_size, shuffle=True, augment=True, seed=seed,
    )
    val_ds = _build_ds(
        val_paths, val_labels, num_classes, img_size,
        batch_size=batch_size, shuffle=False, augment=False,
    )
    test_ds = _build_ds(
        test_paths, test_labels, num_classes, img_size,
        batch_size=batch_size, shuffle=False, augment=False,
    )

    return train_ds, val_ds, test_ds, class_names


def get_test_arrays(
    img_size: tuple[int, int] = IMG_SIZE,
    val_fraction: float = 0.80,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return (X_test, y_test_integer, class_names) as numpy arrays.
    Used for sklearn metrics (confusion matrix, classification report).
    """
    class_names = get_class_names()
    valid_paths, valid_labels = _collect_split_paths(VALID_DIR, class_names)
    _, (test_paths, test_labels) = _stratified_split(
        valid_paths, valid_labels, val_fraction=val_fraction, seed=seed
    )
    # Build an unbatched dataset, collect into arrays
    path_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(test_labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(
        lambda p, lbl: (_load_and_preprocess(p, lbl, img_size)[0], lbl),
        num_parallel_calls=AUTOTUNE,
    ).batch(64).prefetch(AUTOTUNE)

    images_list, labels_list = [], []
    for batch_imgs, batch_lbls in ds:
        images_list.append(batch_imgs.numpy())
        labels_list.append(batch_lbls.numpy())

    X = np.concatenate(images_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    return X, y, class_names
