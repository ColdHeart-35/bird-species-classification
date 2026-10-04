"""Prepare the supplied Birds_25 archive without leaking validation images."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def readable_images(folder: Path, required: int | None) -> list[Path]:
    """Return the requested count, skipping corrupt or unsupported image files."""
    selected: list[Path] = []
    skipped = 0
    for path in images_in(folder):
        if cv2.imread(str(path)) is None:
            skipped += 1
            continue
        selected.append(path)
        if required is not None and len(selected) == required:
            break
    if skipped:
        print(f"Skipped {skipped} unreadable file(s) in {folder}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the supplied 25 Indian-bird dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/archive/Birds_25"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--train-per-class", type=int, default=100,
                        help="Balanced training images per species; use 100 for practical HOG + SVM training")
    parser.add_argument("--valid-per-class", type=int, default=30,
                        help="Balanced unseen validation images per species")
    parser.add_argument("--all", action="store_true",
                        help="Use every readable image in both supplied train and valid folders")
    args = parser.parse_args()
    train_root, valid_root = args.dataset_dir / "train", args.dataset_dir / "valid"
    if not train_root.is_dir() or not valid_root.is_dir():
        raise FileNotFoundError("Expected train/ and valid/ inside data/archive/Birds_25")

    train_species = sorted(folder.name for folder in train_root.iterdir() if folder.is_dir())
    valid_species = sorted(folder.name for folder in valid_root.iterdir() if folder.is_dir())
    if train_species != valid_species or len(train_species) != 25:
        raise ValueError("Expected identical 25 species folders in train/ and valid/.")

    rows: list[dict[str, object]] = []
    for class_id, species in enumerate(train_species):
        train_limit = None if args.all else args.train_per_class
        valid_limit = None if args.all else args.valid_per_class
        train_images = readable_images(train_root / species, train_limit)
        valid_images = readable_images(valid_root / species, valid_limit)
        if ((train_limit is not None and len(train_images) < train_limit)
                or (valid_limit is not None and len(valid_images) < valid_limit)):
            raise ValueError(f"{species}: insufficient images ({len(train_images)} train, {len(valid_images)} valid)")
        rows.extend({"image_path": str(path), "class_id": class_id, "species": species, "split": "train"}
                    for path in train_images[:args.train_per_class])
        rows.extend({"image_path": str(path), "class_id": class_id, "species": species, "split": "test"}
                    for path in valid_images[:args.valid_per_class])

    manifest = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(train_species)} species, "
          f"{(manifest.split == 'train').sum()} train and "
          f"{(manifest.split == 'test').sum()} unseen validation images.")


if __name__ == "__main__":
    main()
