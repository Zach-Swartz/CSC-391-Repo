import numpy as np
import cv2
from calculate_gradient import calculate_gradient


# Apply Sobel operator and threshold the gradient magnitude to produce a binary edge map
def sobel_edge_detector(img: np.ndarray, threshold: float) -> np.ndarray:
    # img: grayscale input image
    # threshold: scalar threshold on gradient magnitude (0..255)
    mag, _ = calculate_gradient(img)
    # binary map: 255 where mag >= threshold else 0
    edge_map = np.zeros_like(mag, dtype=np.uint8)
    edge_map[mag >= threshold] = 255
    return edge_map
