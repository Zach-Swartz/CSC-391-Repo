import numpy as np


# compute per-bin counts and normalized distribution
def calculate_histogram(img: np.ndarray, bins: int = 256):
    # flatten the image to 1D for histogram computation
    flat = img.ravel()  # flattened pixel array
    # compute counts per bin over range [0,255]
    counts, edges = np.histogram(flat, bins=bins, range=(
        0, 255))  # array of bin counts
    # total number of pixels
    total = counts.sum()  # number of pixels accounted for
    # normalized distribution (avoid division by zero)
    dist = counts.astype(np.float32) / float(total) if total > 0 else np.zeros_like(
        counts, dtype=np.float32)  # normalized histogram
    return counts, dist


if __name__ == "__main__":
    # sample usage/test
    a = np.array([[0, 0, 255], [128, 128, 128]],
                 dtype=np.uint8)  # test image
    print(calculate_histogram(a, bins=4))
