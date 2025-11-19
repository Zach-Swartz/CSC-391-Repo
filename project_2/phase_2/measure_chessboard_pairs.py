"""Measure chessboard inner corners in pair images and produce metric triangulation.

For each consecutive image pair in the calibrated images folder (phase_1/results/calibrated images), this script:
- detects chessboard corners in both images
- refines corners and runs solvePnP to obtain camera poses
- builds projection matrices P = K * [R|t] for each camera (object/world frame = chessboard)
- triangulates the corresponding corners to recover 3D points in the chessboard frame
- computes RMS error vs the known object points and reports measured extents

Outputs a JSON report and per-pair NPZ containing triangulated 3D corners.
"""
import os
import glob
import json
import numpy as np
import cv2
import argparse


try:
    import common_utils as cu
except Exception:
    from project_2.phase_2 import common_utils as cu



def process_pair(img1_path, img2_path, mtx, dist, cols, rows, square_size, out_dir):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        return {'error': 'could not read images'}

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    f1, c1 = cv2.findChessboardCorners(gray1, pattern, flags)
    f2, c2 = cv2.findChessboardCorners(gray2, pattern, flags)
    res = {'img1': img1_path, 'img2': img2_path, 'found1': bool(f1), 'found2': bool(f2)}
    if not (f1 and f2):
        return res

    c1r = cv2.cornerSubPix(gray1, c1, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    c2r = cv2.cornerSubPix(gray2, c2, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    objp = cu.build_object_points(cols, rows, square_size)

    # solvePnP (use RANSAC for robustness)
    ok1, rvec1, tvec1, inliers1 = cv2.solvePnPRansac(objp, c1r, mtx, dist)
    ok2, rvec2, tvec2, inliers2 = cv2.solvePnPRansac(objp, c2r, mtx, dist)
    if not ok1 or not ok2:
        return {'img1': img1_path, 'img2': img2_path, 'error': 'solvePnPRansac_failed'}

    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)

    # Projection matrices mapping object/world points to image pixels
    P1 = mtx.dot(np.hstack((R1, tvec1)))
    P2 = mtx.dot(np.hstack((R2, tvec2)))

    # Undistort the detected image points to pixel coordinates consistent with P matrices
    c1_und = cv2.undistortPoints(c1r, mtx, dist, P=mtx).reshape(-1, 2)
    c2_und = cv2.undistortPoints(c2r, mtx, dist, P=mtx).reshape(-1, 2)

    # triangulate
    pts3d = cu.triangulate_points(P1, P2, c1_und.reshape(-1,2), c2_und.reshape(-1,2))

    # compare to ground-truth object points (objp)
    # objp is Nx3 (z=0); compute per-point error
    if pts3d.shape[0] == objp.shape[0]:
        diffs = pts3d - objp
        dists = np.linalg.norm(diffs, axis=1)
        rms = float(np.sqrt(np.mean(dists ** 2)))
        max_err = float(np.max(dists))
    else:
        rms = None
        max_err = None

    # measured extents (axis-aligned bounding box)
    mins = pts3d.min(axis=0)
    maxs = pts3d.max(axis=0)
    dims = (maxs - mins).tolist()

    # save npz
    base1 = os.path.splitext(os.path.basename(img1_path))[0]
    base2 = os.path.splitext(os.path.basename(img2_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f'chess_{base1}_vs_{base2}.npz')
    np.savez(npz_path, pts3d=pts3d, objp=objp, corners1=c1r, corners2=c2r, R1=R1, t1=tvec1, R2=R2, t2=tvec2)

    res.update({'npz': npz_path, 'rms_m': rms, 'max_err_m': max_err, 'dims_m': {'dx': dims[0], 'dy': dims[1], 'dz': dims[2]}})
    return res


def main():
    parser = argparse.ArgumentParser()
    # calibrated images location under phase_1/results
    parser.add_argument('--images_dir', default=os.path.join('..', 'phase_1', 'results', 'calibrated images'))
    parser.add_argument('--calibration', default=os.path.join('..', 'phase_1', 'results', 'calibration_results.npz'))
    parser.add_argument('--board-cols', type=int, default=7)
    parser.add_argument('--board-rows', type=int, default=6)
    parser.add_argument('--square-size', type=float, default=0.025)
    parser.add_argument('--out_dir', default=os.path.join('results', 'chess_measurements'))
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    calib = os.path.abspath(args.calibration)
    out_dir = os.path.abspath(args.out_dir)

    mtx, dist = cu.load_calibration(calib)

    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg') and f.startswith('calibrated_chessboard_') and not f.endswith('_visual.jpg')])
    pairs = []
    for i in range(len(files)-1):
        pairs.append((os.path.join(images_dir, files[i]), os.path.join(images_dir, files[i+1])))

    report = []
    for a, b in pairs:
        print('Processing', os.path.basename(a), '<->', os.path.basename(b))
        r = process_pair(a, b, mtx, dist, args.board_cols, args.board_rows, args.square_size, out_dir)
        report.append(r)

    # write report
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'chess_measure_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2)
    print('Wrote report to', os.path.join(out_dir, 'chess_measure_report.json'))


if __name__ == '__main__':
    main()
