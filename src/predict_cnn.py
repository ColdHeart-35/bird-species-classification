"""
CNN Inference Script — Bird Species Identification.

Loads a trained Keras model and predicts the species of a given bird image.
Outputs: predicted species, confidence, and top-3 alternatives.

Usage:
    python -m src.predict_cnn path/to/bird_image.jpg
    python -m src.predict_cnn path/to/bird.jpg --model baseline_cnn
    python -m src.predict_cnn path/to/bird.jpg --model efficientnet --top 5

Example output:
    Predicted species  : Indian Roller
    Confidence         : 91.4%

    Top 3 predictions:
      1. Indian Roller          91.4%
      2. Common Kingfisher       4.8%
      3. Blue Jay                2.1%

The SAME preprocessing used during training is applied here:
    - Resize to the model's input size (224×224 for EfficientNetB0)
    - Scale pixels to [0, 1]
    - No augmentation (inference uses the raw image)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_loader import IMG_SIZE, IMG_SIZE_BASELINE, load_class_labels, LABEL_MAP_PATH

MODELS_DIR = Path("models")

MODEL_CONFIGS: dict[str, dict] = {
    "efficientnet": {
        "path": MODELS_DIR / "efficientnet_finetuned.keras",
        "img_size": IMG_SIZE,
        "label": "EfficientNetB0 (fine-tuned)",
    },
    "baseline_cnn": {
        "path": MODELS_DIR / "baseline_cnn.keras",
        "img_size": IMG_SIZE_BASELINE,
        "label": "Baseline CNN",
    },
}


def preprocess_image_bytes(raw_bytes: bytes, img_size: tuple[int, int]) -> np.ndarray:
    """
    Preprocess raw image bytes using the exact training pipeline:
      1. Decode RGB image
      2. Resize to model input size (e.g. 224x224)
      3. Scale pixels from [0, 255] -> [0, 1]
      4. Add batch dimension (1, H, W, 3)
    """
    image = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # same as training pipeline
    return image.numpy()[np.newaxis, ...]        # add batch dim: (1, H, W, 3)


def load_and_preprocess(image_path: Path, img_size: tuple[int, int]) -> np.ndarray:
    """
    Load an image from disk and apply the same preprocessing as training.
    """
    raw = tf.io.read_file(str(image_path))
    return preprocess_image_bytes(raw.numpy(), img_size)


def run_inference(
    image_input: str | Path | bytes,
    model: keras.Model,
    class_labels: dict[int, str],
    img_size: tuple[int, int] = IMG_SIZE,
    top_k: int = 3,
) -> dict:
    """
    Run inference on an image input (file path or bytes) using a pre-loaded model & labels.

    Returns:
        dict containing:
            - 'predicted_species': Human-readable species name
            - 'raw_class': Exact model class label (e.g., 'Asian-Green-Bee-Eater')
            - 'confidence': Prediction probability (float)
            - 'top_predictions': List of top-K prediction dicts
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        raw_bytes = p.read_bytes()
    elif isinstance(image_input, bytes):
        raw_bytes = image_input
    else:
        raise TypeError("image_input must be a file path or bytes")

    x = preprocess_image_bytes(raw_bytes, img_size)
    probabilities = model.predict(x, verbose=0)[0]

    top_indices = probabilities.argsort()[::-1][:top_k]
    predicted_idx = top_indices[0]
    raw_class = class_labels[predicted_idx]
    predicted_species = raw_class.replace("-", " ")
    confidence = float(probabilities[predicted_idx])

    top_predictions = []
    for rank, idx in enumerate(top_indices, start=1):
        c_raw = class_labels[idx]
        top_predictions.append({
            "rank": rank,
            "raw_class": c_raw,
            "species": c_raw.replace("-", " "),
            "confidence": float(probabilities[idx]),
            "index": int(idx),
        })

    return {
        "predicted_species": predicted_species,
        "raw_class": raw_class,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "probabilities": probabilities,
    }


def predict(
    image_path: str | Path,
    model_key: str = "efficientnet",
    top_k: int = 3,
) -> None:
    """Load model, run inference, and print results."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = MODEL_CONFIGS.get(model_key)
    if cfg is None:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_CONFIGS)}")
    if not cfg["path"].exists():
        raise FileNotFoundError(
            f"Trained model not found at {cfg['path']}.\n"
            f"Run training first:  python -m src.train_efficientnet"
        )
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Class label map not found at {LABEL_MAP_PATH}.\n"
            f"Run training first to generate the label map."
        )

    # Load model and labels
    model = keras.models.load_model(str(cfg["path"]))
    class_labels = load_class_labels()           # {int → "species_name"}

    result = run_inference(
        image_input=image_path,
        model=model,
        class_labels=class_labels,
        img_size=cfg["img_size"],
        top_k=top_k,
    )

    # ── Print results ──────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Model              : {cfg['label']}")
    print(f"  Image              : {image_path.name}")
    print(f"{'─' * 50}")
    print(f"  Predicted species  : {result['predicted_species']}")
    print(f"  Confidence         : {result['confidence']:.2%}")
    print(f"\n  Top {top_k} predictions:")
    for item in result["top_predictions"]:
        species = item["species"]
        prob = item["confidence"]
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"    {item['rank']}. {species:<30} {prob:>6.2%}  {bar}")
    print(f"{'─' * 50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict bird species from an image using a trained CNN model"
    )
    parser.add_argument("image", type=Path, help="Path to the bird image")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="efficientnet",
        help="Which trained model to use (default: efficientnet)",
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Number of top predictions to display (default: 3)",
    )
    args = parser.parse_args()
    predict(args.image, model_key=args.model, top_k=args.top)


if __name__ == "__main__":
    main()
