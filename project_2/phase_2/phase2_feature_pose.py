"""Feature matching and relative pose estimation (Phase 2)."""
import os
import argparse
import json
import numpy as np
import cv2
import shutil
import re
try:
    import common_utils as cu
except Exception:
    from project_2.phase_2 import common_utils as cu



def run(args):
    images_dir = os.path.abspath(args.images_dir)
    calib_path = os.path.abspath(args.calibration)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, 'match_visuals')
    os.makedirs(vis_dir, exist_ok=True)

    mtx, dist = cu.load_calibration(calib_path)

    image_files = sorted(
        [f for f in os.listdir(images_dir)
         if f.lower().endswith('.jpg') and f.startswith('calibrated_chessboard_') and not f.endswith('_visual.jpg')]
    )
    if len(image_files) < 2:
        print('Need at least two images to run Phase 2')
        return

    detector, use_sift = cu.get_detector(args.detector)

    summary = []

    # load, undistort and compute keypoints/descriptors
    image_data = []  # (filename, original_bgr, undistorted_bgr, undistort_mtx)
    keypoints = []
    descriptors = []
    for fname in image_files:
        path = os.path.join(images_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print('Could not read', path)
            continue
        undistorted, und_mtx = cu.undistort_image(img, mtx, dist)
        image_data.append((fname, img, undistorted, und_mtx))
        kp, des = detector.detectAndCompute(undistorted, None)
        keypoints.append(kp)
        descriptors.append(des)

    # iterate consecutive image pairs
    for i in range(len(image_data) - 1):
        name1, orig1, und1, und_mtx1 = image_data[i]
        name2, orig2, und2, und_mtx2 = image_data[i + 1]
        kp1, des1 = keypoints[i], descriptors[i]
        kp2, des2 = keypoints[i + 1], descriptors[i + 1]

        good_matches = cu.match_features(des1, des2, use_sift)
        print(f'Pair {name1} <-> {name2}: {len(good_matches)} good matches')

        result = {'img1': name1, 'img2': name2, 'matches': len(good_matches)}

        if len(good_matches) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            H, mask_h = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            result['homography_inliers'] = int(np.count_nonzero(mask_h)) if (H is not None and mask_h is not None) else 0

            K_used = und_mtx1 if und_mtx1 is not None else mtx
            try:
                E, maskE = cv2.findEssentialMat(pts1, pts2, K_used, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            except Exception:
                E = None
                maskE = None

            if E is not None and E.shape != ():
                pose_ret, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K_used)
                result['E_found'] = True
                result['R'] = R.tolist()
                result['t'] = t.tolist()
                result['essential_inliers'] = int(np.count_nonzero(pose_mask)) if pose_mask is not None else 0

                try:
                    pose_mask_arr = pose_mask.ravel() if pose_mask is not None else None
                    if pose_mask_arr is not None:
                        pts1_inliers = pts1[pose_mask_arr != 0]
                        pts2_inliers = pts2[pose_mask_arr != 0]
                    else:
                        pts1_inliers = pts1
                        pts2_inliers = pts2

                    result['triangulation_candidate_count'] = int(pts1_inliers.shape[0])

                    if pts1_inliers.shape[0] >= 6:
                        P_cam0 = np.hstack((np.eye(3), np.zeros((3, 1))))
                        P_cam1 = np.hstack((R, t))
                        P0 = K_used.dot(P_cam0)
                        P1 = K_used.dot(P_cam1)

                        # Triangulate inlier point correspondences. We build projection matrices P0/P1
                        # in camera coordinates then multiply by K_used to get pixel-domain projection matrices.
                        # cu.triangulate_points expects pixel-domain points/projections and returns 3D points
                        # in the camera coordinate frame.
                        pts3d = cu.triangulate_points(P0, P1, pts1_inliers, pts2_inliers)

                        b1 = os.path.splitext(name1)[0]
                        b2 = os.path.splitext(name2)[0]
                        ply_name = os.path.join(out_dir, f'points_{i+1:02d}_{b1}_vs_{b2}.ply')
                        with open(ply_name, 'w') as pf:
                            pf.write('ply\nformat ascii 1.0\n')
                            pf.write(f'element vertex {pts3d.shape[0]}\n')
                            pf.write('property float x\nproperty float y\nproperty float z\nend_header\n')
                            for p in pts3d:
                                pf.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
                        result['point_cloud'] = os.path.relpath(ply_name, start=out_dir)

                        npz_name = os.path.join(out_dir, f'points_{i+1:02d}_{b1}_vs_{b2}.npz')
                        try:
                            np.savez(npz_name, pts1_in=pts1_inliers, pts2_in=pts2_inliers, pts3d=pts3d, K_used=K_used, R=R, t=t)
                            result['triangulation_npz'] = os.path.relpath(npz_name, start=out_dir)
                            try:
                                p = os.path.abspath(os.path.dirname(__file__))
                                while os.path.basename(p) != 'phase_2' and os.path.dirname(p) != p:
                                    p = os.path.dirname(p)
                                points_dir = os.path.join(p, 'results', 'points')
                                os.makedirs(points_dir, exist_ok=True)
                                m1 = re.search(r'calibrated_chessboard_(\d+)', b1)
                                m2 = re.search(r'calibrated_chessboard_(\d+)', b2)
                                if m1 and m2:
                                    newname = f"p{i+1:02d}_c{int(m1.group(1))}_vs_c{int(m2.group(1))}.npz"
                                else:
                                    newname = f"p{i+1:02d}_{b1}_vs_{b2}.npz"
                                dst = os.path.join(points_dir, newname)
                                shutil.copy2(npz_name, dst)
                                result['triangulation_npz_canonical'] = os.path.join('points', newname)
                            except Exception as ex_cp:
                                result['triangulation_npz_copy_error'] = str(ex_cp)

                            try:
                                if hasattr(args, 'square_size') and args.square_size is not None:
                                    pattern = (args.board_cols, args.board_rows) if hasattr(args, 'board_cols') else (7, 6)
                                    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                                    g1 = cv2.cvtColor(orig1, cv2.COLOR_BGR2GRAY)
                                    g2 = cv2.cvtColor(orig2, cv2.COLOR_BGR2GRAY)
                                    f1, c1 = cv2.findChessboardCorners(g1, pattern, flags)
                                    f2, c2 = cv2.findChessboardCorners(g2, pattern, flags)
                                    if f1 and f2:
                                        c1r = cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                                        c2r = cv2.cornerSubPix(g2, c2, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                                        cols = args.board_cols if hasattr(args, 'board_cols') else 7
                                        rows = args.board_rows if hasattr(args, 'board_rows') else 6
                                        objp = []
                                        for rr in range(rows):
                                            for cc in range(cols):
                                                objp.append([cc * args.square_size, rr * args.square_size, 0.0])
                                        objp = np.array(objp, dtype=np.float32)
                                        ok1, rvec1, tvec1 = cv2.solvePnP(objp, c1r, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                                        ok2, rvec2, tvec2 = cv2.solvePnP(objp, c2r, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                                        if ok1 and ok2:
                                            R1, _ = cv2.Rodrigues(rvec1)
                                            R2, _ = cv2.Rodrigues(rvec2)
                                            R_rel = R2.dot(R1.T)
                                            t_rel = tvec2 - R_rel.dot(tvec1)
                                            t_est_norm = float(np.linalg.norm(t))
                                            t_rel_norm = float(np.linalg.norm(t_rel))
                                            if t_est_norm > 1e-8 and t_rel_norm > 0:
                                                scale = t_rel_norm / t_est_norm
                                                result['scale'] = float(scale)
                                                pts3d_metric = pts3d * scale
                                                metric_npz = os.path.join(out_dir, f'points_{i+1:02d}_{b1}_vs_{b2}_metric.npz')
                                                np.savez(metric_npz, pts3d_metric=pts3d_metric, scale=scale)
                                                result['triangulation_metric_npz'] = os.path.relpath(metric_npz, start=out_dir)
                                                ply_metric = os.path.join(out_dir, f'points_{i+1:02d}_{b1}_vs_{b2}_metric.ply')
                                                with open(ply_metric, 'w') as pfm:
                                                    pfm.write('ply\nformat ascii 1.0\n')
                                                    pfm.write(f'element vertex {pts3d_metric.shape[0]}\n')
                                                    pfm.write('property float x\nproperty float y\nproperty float z\nend_header\n')
                                                    for p in pts3d_metric:
                                                        pfm.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
                                                result['point_cloud_metric'] = os.path.relpath(ply_metric, start=out_dir)
                                                mins = pts3d_metric.min(axis=0)
                                                maxs = pts3d_metric.max(axis=0)
                                                dims = (maxs - mins).tolist()
                                                result['dimensions_m'] = {'dx_m': float(dims[0]), 'dy_m': float(dims[1]), 'dz_m': float(dims[2])}
                                    else:
                                        result['scale_attempt'] = 'chessboard_not_found_in_both'
                                else:
                                    result['scale_attempt'] = 'no_square_size_provided'
                            except Exception as ex_scale:
                                result['scale_error'] = str(ex_scale)
                        except Exception as ex_npz:
                            result['triangulation_npz_error'] = str(ex_npz)
                    else:
                        result['point_cloud'] = None
                except Exception as ex:
                    result['point_cloud_error'] = str(ex)
            else:
                result['E_found'] = False
                result['essential_inliers'] = 0
        else:
            result['homography_inliers'] = 0
            result['E_found'] = False
            result['essential_inliers'] = 0

        # draw inlier-only matches when available (prefer essential mask), else top matches
        matches_to_draw = good_matches[:200]
        try:
            if 'pose_mask' in locals() and pose_mask is not None:
                inlier_idxs = [idx for idx, m in enumerate(pose_mask.ravel()) if m == 1]
                matches_inliers = [good_matches[idx] for idx in inlier_idxs if idx < len(good_matches)]
                if matches_inliers:
                    matches_to_draw = matches_inliers[:200]
        except Exception:
            pass

        vis_matches = cv2.drawMatches(und1, kp1, und2, kp2, matches_to_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        b1 = os.path.splitext(name1)[0]
        b2 = os.path.splitext(name2)[0]
        vis_path = os.path.join(vis_dir, f'matches_{i+1:02d}_{b1}_vs_{b2}.jpg')
        cv2.imwrite(vis_path, vis_matches)
        result['match_vis'] = os.path.relpath(vis_path, start=out_dir)

        summary.append(result)

    # save summary
    with open(os.path.join(out_dir, 'phase2_pose_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote summary to', os.path.join(out_dir, 'phase2_pose_summary.json'))


def main():
    parser = argparse.ArgumentParser()
    # image defaults now point to the phase_1 results folder where calibrated images live
    parser.add_argument('--images_dir', default=os.path.join('..', 'phase_1', 'results', 'calibrated images'))
    # defaults anchored to this script's folder so outputs land under project_2/phase_2/results
    base_dir = os.path.dirname(__file__)
    parser.add_argument('--calibration', default=os.path.join(base_dir, '..', 'phase_1', 'results', 'calibration_results.npz'))
    parser.add_argument('--out_dir', default=os.path.join(base_dir, 'results'))
    parser.add_argument('--detector', default='sift', choices=['sift', 'orb'])
    parser.add_argument('--square-size', type=float, default=None, help='Chessboard square size in meters (optional, used to resolve metric scale)')
    parser.add_argument('--board-cols', type=int, default=7, help='Number of internal corners per chessboard row (columns)')
    parser.add_argument('--board-rows', type=int, default=6, help='Number of internal corners per chessboard column (rows)')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
