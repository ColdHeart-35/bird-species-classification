"""Image preprocessing and HOG feature extraction."""
from __future__ import annotations

import cv2
import numpy as np

IMAGE_SIZE = (64, 64)
HOG = cv2.HOGDescriptor(
    _winSize=IMAGE_SIZE, _blockSize=(16, 16), _blockStride=(8, 8),
    _cellSize=(8, 8), _nbins=9,
)


def extract_hog(image_path: str) -> np.ndarray:
    """Load an image and return normalized HOG features for the SVM."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return HOG.compute(equalized).ravel().astype(np.float32)
