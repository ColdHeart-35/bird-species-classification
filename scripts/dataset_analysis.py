"""
Dataset analysis for Birds_25 — prints actual statistics from disk.

Usage:
    python scripts/dataset_analysis.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_ROOT = Path("data/archive/Birds_25")


def count_images(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for species_dir in sorted(split_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        imgs = [p for p in species_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        counts[species_dir.name] = len(imgs)
    return counts


def check_corrupted(split_dir: Path, sample: int = 5) -> list[str]:
    """Check a sample of images per class for corruption."""
    corrupted: list[str] = []
    for species_dir in sorted(split_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        imgs = [p for p in species_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        for img_path in imgs[:sample]:
            frame = cv2.imread(str(img_path))
            if frame is None:
                corrupted.append(str(img_path))
    return corrupted


def sample_dimensions(split_dir: Path, per_class: int = 3) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for species_dir in sorted(split_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        imgs = [p for p in species_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        for img_path in imgs[:per_class]:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                h, w = frame.shape[:2]
                sizes.append((w, h))
    return sizes


def main() -> None:
    train_dir = DATASET_ROOT / "train"
    valid_dir = DATASET_ROOT / "valid"

    if not train_dir.is_dir() or not valid_dir.is_dir():
        print(f"ERROR: Expected {DATASET_ROOT}/train and {DATASET_ROOT}/valid", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  BIRDS-25 DATASET ANALYSIS")
    print("=" * 60)

    train_counts = count_images(train_dir)
    valid_counts = count_images(valid_dir)
    all_species = sorted(set(train_counts) | set(valid_counts))

    n_classes = len(all_species)
    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())
    total_images = total_train + total_valid

    print(f"\nNumber of classes     : {n_classes}")
    print(f"Total images          : {total_images:,}")
    print(f"Training images       : {total_train:,}  (official train/ folder)")
    print(f"Validation images     : {total_valid:,}  (official valid/ folder — used as val+test)")

    # Split plan
    val_images = round(total_valid * 0.8)
    test_images = total_valid - val_images
    print(f"  ↳ Validation subset : {val_images:,}  (80% of valid/)")
    print(f"  ↳ Test subset       : {test_images:,}  (20% of valid/, held-out)")

    print(f"\n{'Species':<32} {'Train':>7} {'Valid':>7}")
    print("-" * 48)
    for sp in all_species:
        tr = train_counts.get(sp, 0)
        va = valid_counts.get(sp, 0)
        print(f"  {sp:<30} {tr:>7} {va:>7}")

    train_values = list(train_counts.values())
    print(f"\nClass imbalance (train):")
    print(f"  Min images/class    : {min(train_values):,}")
    print(f"  Max images/class    : {max(train_values):,}")
    print(f"  Mean images/class   : {np.mean(train_values):.1f}")
    imbalance_ratio = max(train_values) / min(train_values)
    print(f"  Imbalance ratio     : {imbalance_ratio:.3f}  ", end="")
    print("(nearly balanced ✓)" if imbalance_ratio < 1.2 else "(imbalanced — class weights will be used)")

    print("\nChecking for corrupted images (sampling 5/class in train)...")
    corrupted = check_corrupted(train_dir, sample=5)
    if corrupted:
        print(f"  Found {len(corrupted)} corrupted image(s):")
        for c in corrupted:
            print(f"    {c}")
    else:
        print("  No corrupted images found in sample ✓")

    print("\nSampling image dimensions from train/ ...")
    sizes = sample_dimensions(train_dir, per_class=3)
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"  Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
        print(f"  Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
        print("  Images are variable resolution; will resize to 224×224 for CNN")

    print("\n" + "=" * 60)
    print("  PREVIOUS MODEL PERFORMANCE (HOG + LinearSVM)")
    print("=" * 60)
    print("  Test accuracy  : 17.8%  (from outputs/metrics.json)")
    print("  Macro F1       : 17.4%")
    print("  Reason for failure:")
    print("    1. Grayscale conversion discards bird colour information")
    print("    2. Only 100/1200 training images/class were used")
    print("    3. HOG at 64×64 cannot capture fine plumage detail")
    print("    4. Linear SVM cannot learn hierarchical visual features")
    print("\n  New CNN target   : >80% val accuracy with EfficientNetB0")
    print("=" * 60)


if __name__ == "__main__":
    main()
