# Phase 1 — Camera Calibration (draft)

Date: 2025-11-14

## Introduction

This document summarizes Phase 1 (Camera Calibration and Geometric Modeling). The aim is to compute a robust intrinsic camera matrix K and distortion coefficients using OpenCV and a chessboard calibration target. These calibration results will be used in Phase 2 for geometric measurements / feature-based tasks.

## Data capture

- Camera: (record your camera model & resolution here)
- Chessboard: interior corners pattern used = 7 x 6 (7 columns, 6 rows of interior corners)
- Number of images used for calibration: 21 (images stored in `results/calibrated images/` as `calibrated_chessboard_1.jpg` ... `calibrated_chessboard_21.jpg`).
- Capture advice: varied angles and distances; avoid motion blur; ensure full board coverage in frame.

## Phase 1: Methodology

1. Corner detection: `cv2.findChessboardCorners()` followed by `cv2.cornerSubPix()` for subpixel refinement.
2. Object points: the board is modeled as a planar grid; object points arranged in (x,y,0) using the known interior-corner pattern and an arbitrary square size of 1.0 unit (square size can be scaled later to real-world units).
3. Calibration: used `cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)` to compute intrinsic matrix `K`, distortion coefficients, and per-image extrinsic rotations/translations.
4. Validation: used `cv2.undistort()` to undistort images and computed per-image reprojection errors.

## Results

- RMS reprojection error (cv2.calibrateCamera output): 2.125413144513621
- Mean reprojection error (per-image average): 0.2879678552576454 px

Camera matrix (K):

```
[[443.57302947,   0.        , 297.91344013],
 [  0.        , 446.05543623, 199.2438857 ],
 [  0.        ,   0.        ,   1.        ]]
```

Distortion coefficients (k1, k2, p1, p2, k3):

```
[-0.03890023, -0.19832575, -0.00845366, -0.00453602, 0.17875964]
```

Saved artifacts (folder `project_2/phase_1/results`):

- `calibration_results.npz` — binary arrays (camera matrix, dist, rvecs, tvecs)
- `calibration_summary.json` — numeric summary
- `undistorted_sample.jpg` and `undistort_compare.jpg` — example undistort images
- `undistort_examples/compare_calibrated_chessboard_*.jpg` — side-by-side originals and undistorted versions for all calibrated images

## Discussion / Analysis

- The mean reprojection error ≈ 0.29 px is low and indicates a reliable intrinsics estimate for many computer-vision tasks.
- RMS reported by `calibrateCamera` is 2.125; the per-image mean reprojection error (0.29 px) is a more intuitive measure for pixel-level accuracy.
- If needed, accuracy can be improved by:
  - Increasing the number and diversity of calibration images (various poses & distances).
  - Accurately providing the physical square size to compute focal length in metric units.
  - Using a larger chessboard pattern (more corners) to increase constraints.

## Phase 2 (next steps)

- Use `calibration_results.npz` to undistort images before running feature detection and matching.
- For scene-specific extrinsics: if you need the camera pose relative to a particular scene, compute `rvecs`/`tvecs` from calibration or re-solve for the scene using solvePnP on known markers.

## Conclusion

Phase 1 calibration is complete: intrinsic matrix and distortion coefficients are saved, validation images are available. The dataset contains 21 good calibration images and undistortion examples. The next step is to perform Phase 2 (feature matching + geometric estimation) using the undistorted images.

## Files & commands used

Run these from the `project_2/phase_1` folder:

```powershell
python intrinsic_parameters.py --images_dir "results/calibrated images" --pattern 7 6
python tools/generate_undistort_examples.py --images_dir "results/calibrated images" --calibration results/calibration_results.npz --out_dir results/undistort_examples
```
