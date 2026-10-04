"""Create a balanced, reproducible multi-species CUB manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_space_table(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", names=names, engine="python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/CUB_200_2011"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--classes", type=int, default=10, help="Number of species (2-200)")
    parser.add_argument("--samples-per-class", type=int, default=80, help="Maximum images per species")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.classes <= 200:
        parser.error("--classes must be between 2 and 200")
    if not args.dataset_dir.exists():
        raise FileNotFoundError("Dataset missing. Run: python scripts/download_cub.py")

    images = read_space_table(args.dataset_dir / "images.txt", ["image_id", "relative_path"])
    labels = read_space_table(args.dataset_dir / "image_class_labels.txt", ["image_id", "class_id"])
    split = read_space_table(args.dataset_dir / "train_test_split.txt", ["image_id", "official_train"])
    class_names = read_space_table(args.dataset_dir / "classes.txt", ["class_id", "species"])
    frame = images.merge(labels, on="image_id").merge(split, on="image_id").merge(class_names, on="class_id")
    selected = class_names.head(args.classes)
    frame = frame[frame.class_id.isin(selected.class_id)].copy()

    balanced = (frame.groupby("class_id", group_keys=False)
                .sample(n=min(args.samples_per_class, frame.groupby("class_id").size().min()), random_state=args.seed))
    train, test = train_test_split(balanced, test_size=args.test_size, random_state=args.seed,
                                  stratify=balanced["class_id"])
    balanced["split"] = "train"
    balanced.loc[balanced.image_id.isin(test.image_id), "split"] = "test"
    balanced["image_path"] = balanced.relative_path.map(lambda p: str(args.dataset_dir / "images" / p))
    output = balanced[["image_id", "image_path", "class_id", "species", "split"]].sort_values("image_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(train)} training and {len(test)} test images, {args.classes} species.")
    print("Selected species:")
    print("\n".join(f"  {i}. {name}" for i, name in enumerate(selected.species, 1)))


if __name__ == "__main__":
    main()
