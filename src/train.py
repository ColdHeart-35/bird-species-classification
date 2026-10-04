"""Train and evaluate a multi-class SVM bird classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from src.features import HOG, IMAGE_SIZE, extract_hog


def features_for(frame: pd.DataFrame, cache: Path) -> np.ndarray:
    if cache.exists():
        saved = np.load(cache)
        expected_paths = frame.image_path.to_numpy(dtype=str)
        if (np.array_equal(saved["image_paths"], expected_paths)
                and saved["features"].shape[1] == HOG.getDescriptorSize()):
            return saved["features"]
    features = np.vstack([extract_hog(path) for path in tqdm(frame.image_path, desc="Extracting HOG")])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, image_paths=frame.image_path.to_numpy(dtype=str))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a HOG + SVM bird classifier")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/bird_svm.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    parser.add_argument("--fast", action="store_true",
                        help="Train a scalable linear SVM directly (recommended for Birds_25)")
    args = parser.parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError("Manifest missing. Run: python scripts/prepare_dataset.py")

    frame = pd.read_csv(args.manifest)
    train, test = frame[frame.split == "train"], frame[frame.split == "test"]
    x_train = features_for(train, Path("data/processed/train_hog.npz"))
    x_test = features_for(test, Path("data/processed/test_hog.npz"))
    y_train, y_test = train.class_id.to_numpy(), test.class_id.to_numpy()

    if args.fast:
        # LinearSVC scales far better than kernel SVC for 25 classes and 8K HOG features.
        model = Pipeline([("scaler", StandardScaler()),
                          ("svm", SGDClassifier(loss="hinge", alpha=0.0001, class_weight="balanced",
                                                max_iter=2000, tol=1e-3, random_state=42))])
        model.fit(x_train, y_train)
        best_params = {"svm": "SGDClassifier (linear SVM / hinge loss)", "alpha": 0.0001}
    else:
        pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True, class_weight="balanced"))])
        search = GridSearchCV(pipeline, {
            "svm__kernel": ["linear", "rbf", "poly"], "svm__C": [1, 10],
            "svm__gamma": ["scale"], "svm__degree": [2, 3],
        }, scoring="f1_macro", cv=args.cv, n_jobs=-1, refit=True)
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    predictions = model.predict(x_test)
    label_map = train.drop_duplicates("class_id").set_index("class_id").species.to_dict()
    labels = sorted(label_map)
    names = [label_map[label] for label in labels]
    report = classification_report(y_test, predictions, labels=labels, target_names=names, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, predictions)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "label_map": label_map, "image_size": list(IMAGE_SIZE)}, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "best_params": best_params, "report": report}, indent=2))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title(f"SVM Confusion Matrix (accuracy: {accuracy:.2%})")
    plt.xlabel("Predicted species"); plt.ylabel("Actual species"); plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=180); plt.close()
    print(f"Selected SVM configuration: {best_params}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
