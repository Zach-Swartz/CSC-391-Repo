"""Phase 2 extra: calibrated AR overlay using the chessboard and solvePnP

Detect the chessboard in a single image, compute pose with the saved
camera intrinsics and distortion, and draw a projected cube anchored to
the board using cv2.projectPoints().

Saves an output image with the overlay and prints reprojection statistics.
"""
import os
import argparse
import math
import numpy as np
import cv2
try:
    import common_utils as cu
except Exception:
    from project_2.phase_2 import common_utils as cu



def draw_cube(img, imgpts):
    img = img.copy()
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # base 0-3, top 4-7
    # draw base in green
    cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), 2)
    # draw pillars in blue
    for i in range(4):
        cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (255, 0, 0), 2)
    # draw top in red
    cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)
    return img


def reprojection_error(objp, imgpts_detected, rvec, tvec, K, dist):
    # project object points using solved pose and compute mean pixel error
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    imgpts = imgpts_detected.reshape(-1, 2)
    diffs = proj - imgpts
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.mean(dists)), float(np.max(dists))


def run(args):
    img_path = os.path.abspath(args.image)
    calib_path = os.path.abspath(args.calibration)
    out_path = os.path.abspath(args.out)

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    if not os.path.exists(calib_path):
        raise FileNotFoundError(calib_path)

    mtx, dist = cu.load_calibration(calib_path)

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError('Could not read image ' + img_path)

    und, newK = cu.undistort_image(img, mtx, dist)

    pattern = (args.board_cols, args.board_rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    # Try detection on the undistorted image first (simpler geometry)
    gray_und = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray_und, pattern, flags)

    used_K = None
    used_dist = None
    vis_img = und

    if found:
        corners2 = cv2.cornerSubPix(gray_und, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        used_K = newK
        used_dist = None
        vis_img = und
    else:
        # fallback: try detection on the original (possibly-distorted) image and use original K/dist
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray_orig, pattern, flags)
        if not found:
            print('Chessboard not found in', img_path)
            return 1
        corners2 = cv2.cornerSubPix(gray_orig, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        used_K = mtx
        used_dist = dist
        vis_img = img

    # build object points in the same ordering
    objp = cu.build_object_points(args.board_cols, args.board_rows, args.square_size)

    # solvePnP: pass the appropriate distortion parameter depending on whether we undistorted
    success, rvec, tvec = cv2.solvePnP(objp, corners2, used_K, used_dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        print('solvePnP failed')
        return 1

    # choose cube size: half the smaller dimension of the board in world units
    s = args.square_size * min(args.board_cols - 1, args.board_rows - 1) * 0.5

    # cube base (placed at the first inner corner origin)
    cube_pts = np.array([
        [0, 0, 0],
        [s, 0, 0],
        [s, s, 0],
        [0, s, 0],
        [0, 0, -s],
        [s, 0, -s],
        [s, s, -s],
        [0, s, -s]
    ], dtype=np.float32)

    imgpts, _ = cv2.projectPoints(cube_pts, rvec, tvec, newK, None)

    vis = draw_cube(und, imgpts)

    # compute reprojection error for chessboard points (use original dist = None because undistorted)
    mean_err, max_err = reprojection_error(objp, corners2, rvec, tvec, newK, None)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)

    print(f'Wrote AR visual to {out_path}')
    print(f'Reprojection error (mean px): {mean_err:.3f}, max px: {max_err:.3f}')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Input chessboard image')
    parser.add_argument('--calibration', default=os.path.join(os.path.dirname(__file__), '..', 'phase_1', 'results', 'calibration_results.npz'))
    parser.add_argument('--square-size', type=float, default=0.025, help='Chessboard square size in meters (default 0.025)')
    parser.add_argument('--board-cols', type=int, default=7, help='Number of internal corners per chessboard row (columns)')
    parser.add_argument('--board-rows', type=int, default=6, help='Number of internal corners per chessboard column (rows)')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results', 'ar_visuals', 'ar_overlay.jpg'))
    args = parser.parse_args()
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
