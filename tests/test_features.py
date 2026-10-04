import numpy as np
import cv2

from src.features import extract_hog


def test_hog_features_are_one_dimensional(tmp_path):
    image_path = tmp_path / "bird.jpg"
    cv2.imwrite(str(image_path), np.full((50, 80, 3), 127, dtype=np.uint8))
    features = extract_hog(str(image_path))
    assert features.ndim == 1
    assert len(features) > 0
    assert np.isfinite(features).all()
