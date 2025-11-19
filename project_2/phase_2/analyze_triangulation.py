"""Analyze triangulation NPZ files produced by phase2_feature_pose.py

Usage: run from repository or provide absolute path to npz.

This script loads a .npz file containing:
- pts1_in: (N,2) 2D points in image1 coordinates
- pts2_in: (N,2) 2D points in image2 coordinates
- pts3d: (N,3) triangulated 3D points
- K_used: (3,3) camera intrinsic used for projection
- R: (3,3) rotation from cam0->cam1
- t: (3,1) translation vector

It computes reprojection errors of pts3d into both images and prints mean/rms.
"""
import sys
import numpy as np
import os


def reprojection_stats(npz_path):
    data = np.load(npz_path)
    pts1 = data['pts1_in']
    pts2 = data['pts2_in']
    pts3d = data['pts3d']
    K = data['K_used']
    R = data['R']
    t = data['t']

    # ensure shapes
    assert pts3d.shape[0] == pts1.shape[0] == pts2.shape[0]

    def project_points(X, P):
        # X: (N,3); P: (3,4)
        Xh = np.hstack((X, np.ones((X.shape[0], 1))))
        proj = (P @ Xh.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj

    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t.reshape(3,1) if t.ndim == 1 else t))

    p1 = project_points(pts3d, P0)
    p2 = project_points(pts3d, P1)

    err1 = np.linalg.norm(p1 - pts1, axis=1)
    err2 = np.linalg.norm(p2 - pts2, axis=1)

    def stats(err):
        return {'count': int(err.size), 'mean': float(np.mean(err)), 'rms': float(np.sqrt(np.mean(err**2))), 'max': float(np.max(err))}

    return stats(err1), stats(err2)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python analyze_triangulation.py path/to/points_XX_pair.npz')
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print('File not found:', path)
        sys.exit(2)
    s1, s2 = reprojection_stats(path)
    print('Reprojection stats for image1:', s1)
    print('Reprojection stats for image2:', s2)
