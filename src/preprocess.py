"""Pre-processing: Gaussian smoothing followed by adaptive thresholding (paper Eq. 2-3)."""
import cv2
import numpy as np


def preprocess_image(path: str, size: int = 224, blur_ksize: int = 5, sigma: float = 1.0,
                     block_size: int = 11, c: int = 2) -> np.ndarray:
    """Read an image and return a binary (0/255) handwriting image of shape (size, size)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), sigma)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )
    return binary
