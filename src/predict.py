"""Predict a bird species from one image using a trained SVM model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from src.features import extract_hog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a bird image")
    parser.add_argument("--model", type=Path, default=Path("models/bird_svm.joblib"))
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError("Trained model missing. Run: python -m src.train")
    bundle = joblib.load(args.model)
    features = extract_hog(str(args.image)).reshape(1, -1)
    pipeline = bundle["pipeline"]
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)[0]
        score_label = "Confidence"
    else:
        # Fast linear SVM has no calibrated probabilities; show relative decision scores.
        scores = pipeline.decision_function(features)[0]
        shifted = scores - scores.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        score_label = "Relative decision score (not calibrated confidence)"
    class_ids = pipeline.classes_
    best = probabilities.argmax()
    print(f"Predicted species: {bundle['label_map'][class_ids[best]]}")
    print(f"{score_label}: {probabilities[best]:.2%}")
    print("\nTop 3 predictions:")
    for index in probabilities.argsort()[-3:][::-1]:
        print(f"  {bundle['label_map'][class_ids[index]]}: {probabilities[index]:.2%}")


if __name__ == "__main__":
    main()
