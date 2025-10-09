import numpy as np


# Linearly map input intensity range [r_min, r_max]
# to [0,255] and return an uint8 image.
def contrast_stretch(img: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    # convert to float for numeric operations
    arr = img.astype(np.float32)  # working float image
    # denominator for linear mapping, avoid divide-by-zero
    denom = float(r_max - r_min) if (r_max -
                                     r_min) != 0 else 1.0  # denom: mapping scale
    # apply linear mapping to [0,255]
    # stretched: mapped float image
    stretched = (arr - float(r_min)) * (255.0 / denom)
    # clip to valid range and convert back to uint8
    stretched = np.clip(stretched, 0, 255).astype(
        np.uint8)  # final output
    return stretched


if __name__ == "__main__":
    # tiny smoke test input array
    a = np.array([[50, 60], [70, 80]], dtype=np.uint8)  # small sample image
    print(contrast_stretch(a, 50, 80))
