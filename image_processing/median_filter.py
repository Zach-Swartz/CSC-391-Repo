import numpy as np


# apply a median filter of given odd window size to grayscale image
def median_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    # Ensure size is odd and >= 1
    if size <= 1:
        return img.copy()
    if size % 2 == 0:
        raise ValueError('size must be odd')

    h, w = img.shape  # image dimensions
    pad = size // 2  # padding size for neighborhood
    # padded image to handle borders using edge reflection
    padded = np.pad(img, pad, mode='edge')  # padded image
    out = np.empty_like(img)

    # iterate over every pixel and compute median in window
    for y in range(h):
        for x in range(w):
            # flattened neighbor pixels
            window = padded[y:y + size, x:x + size].ravel()
            out[y, x] = np.median(window)  # assign median value

    return out
