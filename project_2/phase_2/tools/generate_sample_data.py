"""Generate small sample calibration data and a few tiny chessboard images.

This script is safe to run locally and creates files under:
  project_2/phase_1/sample_data/

It produces:
 - calibration_results_example.npz  (keys: mtx, dist)
 - calibrated_chessboard_01.jpg, _02.jpg, _03.jpg

Intended for quick smoke tests and CI examples — not for real calibration.
"""
import os
import numpy as np
import cv2


def generate_sample_data(out_dir=None):
    base = out_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'phase_1', 'sample_data')
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)

    # simple synthetic intrinsic matrix
    fx = 800.0
    fy = 800.0
    cx = 320.0
    cy = 240.0
    mtx = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((5,), dtype=np.float64)  # assume zero distortion for synthetic sample

    npz_path = os.path.join(base, 'calibration_results_example.npz')
    np.savez(npz_path, mtx=mtx, dist=dist)

    # generate three small chessboard-like images
    def make_chessboard(w=640, h=480, squares_x=8, squares_y=6, square_px=40, shift=(0,0)):
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        origin_x = 50 + shift[0]
        origin_y = 40 + shift[1]
        for r in range(squares_y):
            for c in range(squares_x):
                x0 = origin_x + c * square_px
                y0 = origin_y + r * square_px
                x1 = x0 + square_px
                y1 = y0 + square_px
                if (r + c) % 2 == 0:
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
        return img

    imgs = [make_chessboard(640, 480, 8, 6, 40, shift=(0,0)),
            make_chessboard(640, 480, 8, 6, 40, shift=(5,3)),
            make_chessboard(640, 480, 8, 6, 40, shift=(-4,6))]

    for i, im in enumerate(imgs, start=1):
        fname = os.path.join(base, f'calibrated_chessboard_{i:02d}.jpg')
        cv2.imwrite(fname, im)

    return base


if __name__ == '__main__':
    out = generate_sample_data()
    print('Wrote sample data to', out)
