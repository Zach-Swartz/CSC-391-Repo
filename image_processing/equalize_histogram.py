import numpy as np


# perform histogram equalization on uint8 images
def equalize_histogram(img: np.ndarray) -> np.ndarray:
    # ensure input is uint8
    if img.dtype != np.uint8:
        arr = img.astype(np.uint8)  # uint8 copy of input
    else:
        arr = img  # alias to input when already uint8

    # flatten pixels for histogram computation
    flat = arr.ravel()  # flattened pixel array
    # compute histogram counts across 256 bins
    counts, _ = np.histogram(flat, bins=256, range=(
        0, 255))  # per-intensity counts
    # cumulative distribution function
    cdf = counts.cumsum()  # cumulative counts
    # smallest non-zero cdf value used for normalization
    # minimum non-zero cumulative count
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    total = flat.size  # total number of pixels
    # build lookup table mapping old intensities to new
    lut = np.round((cdf - cdf_min) / float(total - cdf_min) *
                   255.0).astype(np.uint8)  # mapping array
    lut = np.clip(lut, 0, 255)  # LUT clipped to valid range
    out = lut[arr]  # equalized image via LUT indexing
    return out


if __name__ == "__main__":
    # sample test image
    a = np.array([[0, 0, 255], [128, 128, 128]],
                 dtype=np.uint8)  # test input
    print(equalize_histogram(a))
