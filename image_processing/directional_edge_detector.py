import numpy as np
from calculate_gradient import calculate_gradient


# direction_range: tuple (min_deg, max_deg) in degrees, e.g., (40,50)
# returns binary map where edges have directions within the given angular window
def directional_edge_detector(img: np.ndarray, direction_range: tuple, mag_threshold: float = 20.0) -> np.ndarray:
    # img: grayscale input
    # direction_range: (min_deg, max_deg) in degrees, interpreted modulo 180
    mag, angle = calculate_gradient(img)
    # Normalize angles to 0..180 for undirected edges
    ang = np.abs(angle)  # -180..180 -> 0..180 symmetric
    # Build mask for direction range (handle wrap-around)
    a_min, a_max = direction_range
    if a_min < a_max:
        dir_mask = (ang >= a_min) & (ang <= a_max)
    else:
        # wrap-around (e.g., 170..10)
        dir_mask = (ang >= a_min) | (ang <= a_max)

    # Also require a minimum gradient magnitude to avoid weak directions
    mag_f = mag.astype(np.float32)
    mag_mask = mag_f >= mag_threshold

    out = np.zeros_like(mag, dtype=np.uint8)
    out[dir_mask & mag_mask] = 255
    return out
