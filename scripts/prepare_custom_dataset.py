"""Build a train/test manifest from user-defined bird-species folders.

Expected layout:
data/user_defined/<species_name>/<image files>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare user-defined bird species images")
    parser.add_argument("--input-dir", type=Path, default=Path("data/user_defined"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Create species folders inside: {args.input_dir}")

    rows: list[dict[str, object]] = []
    species_dirs = sorted(folder for folder in args.input_dir.iterdir() if folder.is_dir())
    for class_id, folder in enumerate(species_dirs):
        images = sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) < 5:
            raise ValueError(f"'{folder.name}' has {len(images)} images; add at least 5 images per species.")
        rows.extend({"image_path": str(path), "class_id": class_id, "species": folder.name} for path in images)
    if len(species_dirs) < 2:
        raise ValueError("Add folders for at least two bird species.")

    frame = pd.DataFrame(rows)
    train, test = train_test_split(frame, test_size=args.test_size, random_state=args.seed,
                                  stratify=frame["class_id"])
    train = train.assign(split="train")
    test = test.assign(split="test")
    manifest = pd.concat([train, test]).sort_values(["class_id", "image_path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(train)} train, {len(test)} test images.")
    print("Your species:", ", ".join(species_dirs[i].name for i in range(len(species_dirs))))


if __name__ == "__main__":
    main()
