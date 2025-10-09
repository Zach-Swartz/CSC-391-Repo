import numpy as np
import importlib.util
import os

# Load apply_convolution from project_1's real_time_filters.py by file
_rt_path = os.path.abspath(os.path.join(os.path.dirname(
    __file__), '..', 'project_1', 'part_3', 'real_time_filters.py'))
if os.path.exists(_rt_path):
    spec = importlib.util.spec_from_file_location('rt_filters', _rt_path)
    rt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt_mod)
    apply_convolution = getattr(rt_mod, 'apply_convolution')
else:
    # implement a simple convolution using OpenCV if file not found
    import cv2

    def apply_convolution(image, kernel):
        return cv2.filter2D(image, -1, kernel)


# apply Sobel Sx,Sy via apply_convolution and return gradient magnitude and angle
def calculate_gradient(img: np.ndarray):
    """Compute gradient magnitude and direction using Sobel filters.

    Returns:
        mag: uint8 image with gradient magnitude (0..255)
        angle: float32 array with gradient direction in degrees (range -180..180)
    """
    # define Sobel kernels
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                  dtype=np.float32)  # horizontal gradient kernel
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                  dtype=np.float32)  # vertical gradient kernel

    gx = apply_convolution(img, sx).astype(np.float32)  # gradient along x
    gy = apply_convolution(img, sy).astype(np.float32)  # gradient along y

    # magnitude and angle (radians -> degrees)
    mag_f = np.sqrt(gx * gx + gy * gy)
    angle = np.degrees(np.arctan2(gy, gx))  # range -180..180

    mag = np.clip(mag_f, 0, 255).astype(np.uint8)
    return mag, angle
