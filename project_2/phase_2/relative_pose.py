"""Compute relative pose (R, t) between two images and triangulate points.

This script detects and matches features, estimates F and E, recovers pose,
triangulates inlier correspondences, and writes results to the given output
directory.
"""
import os
import cv2
import json
import numpy as np
import argparse
import shutil
try:
    import common_utils as cu
except Exception:
    from project_2.phase_2 import common_utils as cu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True)
    parser.add_argument('--img2', required=True)
    parser.add_argument('--calib', required=True, help='npz file with mtx and dist (dist optional)')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--feature', choices=['sift', 'orb'], default='sift')
    parser.add_argument('--ratio', type=float, default=0.75)
    parser.add_argument('--ransac-thresh', type=float, default=1.0)
    parser.add_argument('--min-matches', type=int, default=20)
    parser.add_argument('--board-cols', type=int, required=False)
    parser.add_argument('--board-rows', type=int, required=False)
    parser.add_argument('--square-size', type=float, required=False)
    args = parser.parse_args()

    img_a = cv2.imread(args.img1)
    img_b = cv2.imread(args.img2)
    if img_a is None or img_b is None:
        print('Could not read images')
        return

    os.makedirs(args.out_dir, exist_ok=True)

    detector, use_sift = cu.get_detector(args.feature)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    kpa, desa = detector.detectAndCompute(gray_a, None)
    kpb, desb = detector.detectAndCompute(gray_b, None)
    if desa is None or desb is None:
        raise RuntimeError('No descriptors found')

    matches = cu.match_features(desa, desb, use_sift, ratio=args.ratio)
    print(f'Good matches: {len(matches)}')
    if len(matches) < args.min_matches:
        raise RuntimeError(f'Not enough matches: {len(matches)} < {args.min_matches}')

    pts_a = np.float32([kpa[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kpb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate fundamental matrix with RANSAC to remove outliers
    F, fmask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, args.ransac_thresh, 0.99)
    inliers = int(fmask.sum()) if fmask is not None else 0
    print(f'Fundamental inliers: {inliers}/{len(matches)}')

    # load intrinsics
    K, _ = cu.load_calibration(args.calib)

    # essential matrix from calibrated F
    E = K.T.dot(F).dot(K)

    # recover pose (R,t) from E using matched points
    pts_a_n = pts_a.reshape(-1, 2)
    pts_b_n = pts_b.reshape(-1, 2)
    _, R, t, pose_mask = cv2.recoverPose(E, pts_a_n, pts_b_n, K)
    pose_inliers = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0
    print(f'Pose inliers: {pose_inliers}/{len(matches)}')

    # select inlier correspondences for triangulation
    if pose_mask is not None:
        mask_bool = pose_mask.ravel().astype(bool)
    else:
        mask_bool = np.ones(len(matches), dtype=bool)

    if mask_bool.sum() < 6:
        print('Warning: few inliers for triangulation:', int(mask_bool.sum()))

    pts_a_in = pts_a_n[mask_bool]
    pts_b_in = pts_b_n[mask_bool]

    # build projection matrices and triangulate
    P0 = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    P1 = K.dot(np.hstack((R, t)))

    # Triangulate corresponding inlier points to get 3D points in the camera frame.
    # cu.triangulate_points expects pixel-domain projection matrices and pixel coords.
    pts_3d = cu.triangulate_points(P0, P1, pts_a_in, pts_b_in)

    errs_a = cu.reprojection_errors(pts_3d, P0, pts_a_in)
    errs_b = cu.reprojection_errors(pts_3d, P1, pts_b_in)

    summary = {
        'num_matches': len(matches),
        'fundamental_inliers': int(inliers),
        'pose_inliers': int(pose_inliers),
        'reproj_mean_img1': float(np.mean(errs_a)) if errs_a.size else None,
        'reproj_mean_img2': float(np.mean(errs_b)) if errs_b.size else None,
    }

    # optional metric scale recovery using chessboard solvePnP
    if args.board_cols and args.board_rows and args.square_size:
        pattern_size = (args.board_cols, args.board_rows)
        objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * args.square_size

        def find_and_solve(img, K_):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if not found:
                return None
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 30, 0.001))
            ok, rvec, tvec = cv2.solvePnP(objp, corners2, K_, None)
            if not ok:
                return None
            R_mat, _ = cv2.Rodrigues(rvec)
            return {'rvec': rvec, 'tvec': tvec.reshape(3,), 'R': R_mat, 'corners': corners2}

        sol1 = find_and_solve(img_a, K)
        sol2 = find_and_solve(img_b, K)
        if sol1 is not None and sol2 is not None:
            R1 = sol1['R']
            t1 = sol1['tvec'].reshape(3, 1)
            R2 = sol2['R']
            t2 = sol2['tvec'].reshape(3, 1)
            R_rel = R2.dot(R1.T)
            t_rel = t2 - R_rel.dot(t1)
            t_rel = t_rel.reshape(3,)
            summary['metric_t_norm_m'] = float(np.linalg.norm(t_rel))
        else:
            print('Chessboard not found in both images; metric scale recovery skipped')

    # save outputs
    np.save(os.path.join(args.out_dir, 'R.npy'), R)
    np.save(os.path.join(args.out_dir, 't.npy'), t)
    tri_path = os.path.join(args.out_dir, 'triangulated.npz')
    np.savez(tri_path, points_3d=pts_3d, errs1=errs_a, errs2=errs_b)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # save match visualization (use fmask if available)
    cu.save_match_vis(img_a, kpa, img_b, kpb, matches, fmask.flatten() if 'fmask' in locals() and fmask is not None else None, os.path.join(args.out_dir, 'matches_inliers.png'))

    # copy triangulation to central points directory
    try:
        p = os.path.abspath(os.path.dirname(__file__))
        while os.path.basename(p) != 'phase_2' and os.path.dirname(p) != p:
            p = os.path.dirname(p)
        points_dir = os.path.join(p, 'results', 'points')
        os.makedirs(points_dir, exist_ok=True)
        parent = os.path.basename(os.path.normpath(args.out_dir))
        newname = f"{parent}_tri.npz"
        dst = os.path.join(points_dir, newname)
        shutil.copy(tri_path, dst)
        print('Also copied triangulation to', dst)
    except Exception as e:
        print('Could not copy triangulation to central points dir:', e)

    print('Saved results to', args.out_dir)


if __name__ == '__main__':
    main()
