"""Common utilities shared by Phase 2 scripts.

This module centralizes small helper functions used across the Phase 2 tools:
- calibration loading
- undistortion helpers
- object point builders
- detector/matcher helpers
- triangulation / reprojection helpers
- simple visualization helpers

Keep implementations small and dependency-free (only NumPy/OpenCV/os).
"""
import os
import numpy as np
import cv2


def load_calibration(npz_path):
    """Load a calibration .npz and return (mtx, dist).

    Accepts common key names used across the repo ('mtx'/'dist' or
    'camera_matrix'/'dist_coeffs' or the first two arrays in the file).
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = data.files
    if 'mtx' in data and 'dist' in data:
        return data['mtx'], data['dist']
    if 'camera_matrix' in data and 'dist_coeffs' in data:
        return data['camera_matrix'], data['dist_coeffs']
    # fallback: common variants
    if 'camera_matrix' in data and 'dist' in data:
        return data['camera_matrix'], data['dist']
    if len(keys) >= 2:
        return data[keys[0]], data[keys[1]]
    raise RuntimeError('Cannot find calibration arrays in ' + npz_path)


def undistort_image(img, mtx, dist):
    """Return undistorted image and the new camera matrix for pixel coords."""
    h, w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    und = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return und, newcameramtx


def build_object_points(cols, rows, square_size):
    objp = []
    for r in range(rows):
        for c in range(cols):
            objp.append([c * square_size, r * square_size, 0.0])
    return np.array(objp, dtype=np.float32)


def get_detector(name='sift', nfeatures=2000):
    """Return a detector instance and a flag whether it's L2 (SIFT-like) or Hamming."""
    if name == 'sift':
        try:
            return cv2.SIFT_create(), True
        except Exception:
            # OpenCV without contrib -> fall back
            pass
    # default ORB
    return cv2.ORB_create(nfeatures), False


def match_features(desc1, desc2, use_sift=True, ratio=0.75):
    """Match descriptors with a ratio test and return list of good matches (cv2.DMatch)."""
    if desc1 is None or desc2 is None:
        return []
    if use_sift:
        # FLANN may give faster SIFT matching for larger descriptor sets; BF is fine too
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = matcher.knnMatch(desc1, desc2, k=2)
    except Exception:
        return []
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def triangulate_points(P1, P2, pts1, pts2, mtx=None, dist=None):
    """Triangulate correspondences and return Nx3 points and projection matrices P0,P1.

    pts1/pts2 are expected as Nx2 in pixel coords. If mtx/dist provided, the function
    will undistort the points into pixel coords consistent with P matrices.
    """
    if mtx is not None and dist is not None:
        u1 = cv2.undistortPoints(pts1.reshape(-1,1,2), mtx, dist, P=mtx).reshape(-1,2)
        u2 = cv2.undistortPoints(pts2.reshape(-1,1,2), mtx, dist, P=mtx).reshape(-1,2)
    else:
        u1 = pts1.reshape(-1,2)
        u2 = pts2.reshape(-1,2)
    # Ensure P1 and P2 are 3x4
    pts4d = cv2.triangulatePoints(P1, P2, u1.T, u2.T)
    pts3d = (pts4d[:3, :] / (pts4d[3:4, :] + 1e-12)).T
    return pts3d


def reprojection_errors(pts_3d, P, pts_2d):
    # project 3D points with P (3x4) and compare to pts_2d (Nx2)
    pts_3d_h = np.hstack((pts_3d, np.ones((pts_3d.shape[0],1))))
    proj = (P.dot(pts_3d_h.T)).T
    proj_xy = proj[:, :2] / (proj[:, 2:3] + 1e-12)
    errs = np.linalg.norm(proj_xy - pts_2d.reshape(-1,2), axis=1)
    return errs


def save_match_vis(img1, kp1, img2, kp2, matches, mask, outpath, max_draw=200):
    """Create and save a matches visualization image. mask is a 1D array where 1=inlier."""
    matches_to_draw = matches[:max_draw]
    if mask is not None:
        # mask expected as 1D boolean/0-1 array aligned with matches
        matchesMask = [[int(bool(m))] for m in mask[:len(matches_to_draw)]] if hasattr(mask, '__iter__') else None
    else:
        matchesMask = None
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    cv2.imwrite(outpath, vis)
