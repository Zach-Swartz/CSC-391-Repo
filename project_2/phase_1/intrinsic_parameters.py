import os
import json
import argparse
import numpy as np
import cv2 as cv
import glob


# termination criteria for cornerSubPix
termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# find the script base directory
base_dir = os.path.dirname(__file__)

# parse command-line arguments for images folder, pattern and square size
parser = argparse.ArgumentParser(description='Compute camera intrinsics from chessboard images')
parser.add_argument('--images_dir', type=str, default=os.path.join(base_dir, 'images'))
parser.add_argument('--pattern', type=int, nargs=2, default=[7, 6])
parser.add_argument('--square_size', type=float, default=1.0)
args = parser.parse_args()


# select images directory: prefer user-provided, otherwise try common folders
if args.images_dir and os.path.isdir(args.images_dir):
    images_dir = args.images_dir
else:
    cand_a = os.path.join(base_dir, 'calibration_image')
    cand_b = os.path.join(base_dir, 'calibtartion_image')
    cand_c = os.path.join(base_dir, 'images')
    if os.path.isdir(cand_a):
        images_dir = cand_a
    elif os.path.isdir(cand_b):
        images_dir = cand_b
    elif os.path.isdir(cand_c):
        images_dir = cand_c
    else:
        images_dir = args.images_dir or cand_c


# prepare object points for the chessboard pattern and scale by square size
pattern = (int(args.pattern[0]), int(args.pattern[1]))
square_size = float(args.square_size)
obj_template = np.zeros((pattern[0] * pattern[1], 3), np.float32)
obj_template[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
obj_template *= square_size


# collect image file paths from the images directory
image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
if len(image_paths) == 0:
    print('No images found in', images_dir)
    print('Place your chessboard images in that folder and re-run this script.')
    raise SystemExit(1)


# storage for detected object points and image points
object_points_list = []
image_points_list = []
image_size = None


# iterate over images and detect chessboard corners
for path in image_paths:
    img = cv.imread(path)
    if img is None:
        print('Warning: could not read', path)
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    found, corners = cv.findChessboardCorners(gray, pattern, None)
    if found:
        object_points_list.append(obj_template.copy())
        refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
        image_points_list.append(refined)

        # save a visual copy with drawn corners if possible
        cv.drawChessboardCorners(img, pattern, refined, found)
        try:
            visual_name = os.path.join(images_dir, f'visual_{os.path.basename(path)}')
            cv.imwrite(visual_name, img)
        except Exception:
            pass
        try:
            cv.imshow('img', img)
            cv.waitKey(200)
        except Exception:
            pass
    else:
        print('Corners not found for', os.path.basename(path))


try:
    cv.destroyAllWindows()
except Exception:
    pass


# ensure we detected something before calibrating
if len(object_points_list) == 0:
    print('No chessboard corners detected in any image; cannot calibrate.')
    raise SystemExit(2)


# calibrate the camera using detected correspondences
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(object_points_list, image_points_list, image_size, None, None)

print('\nCalibration results:')
print('RMS reprojection error:', ret)
print('Camera matrix (K):\n', camera_matrix)
print('Distortion coefficients:\n', dist_coeffs.ravel())


# compute per-image reprojection error and report
mean_error = 0.0
for i in range(len(object_points_list)):
    projected, _ = cv.projectPoints(object_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv.norm(image_points_list[i], projected, cv.NORM_L2) / len(projected)
    mean_error += error
    print(f'Image {i+1} reprojection error: {error:.4f} px')
mean_error /= len(object_points_list)
print(f'Mean reprojection error: {mean_error:.4f} px')


# save calibration results to a .npz file
out_path = os.path.join(base_dir, 'calibration_results.npz')
np.savez(out_path, mtx=camera_matrix, dist=dist_coeffs, rvecs=rvecs, tvecs=tvecs, rms=ret)
print('Saved calibration to', out_path)


# undistort and save a sample image for quick verification
sample_img = None
for path in image_paths:
    tmp = cv.imread(path)
    if tmp is not None:
        sample_img = tmp
        break

if sample_img is not None:
    h, w = sample_img.shape[:2]
    new_cam, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv.undistort(sample_img, camera_matrix, dist_coeffs, None, new_cam)
    undist_path = os.path.join(images_dir, 'undistorted_sample.jpg')
    cv.imwrite(undist_path, dst)
    print('Saved undistorted sample to', undist_path)

    # save a simple side-by-side comparison image
    side = np.hstack((cv.resize(sample_img, (w//2, h//2)), cv.resize(dst, (w//2, h//2))))
    cmp_path = os.path.join(images_dir, 'undistort_compare.jpg')
    cv.imwrite(cmp_path, side)
    print('Saved comparison to', cmp_path)

    # write a small json summary with numeric results
    summary = {
        'rms': float(ret),
        'mean_reprojection_error': float(mean_error),
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.ravel().tolist()
    }
    with open(os.path.join(base_dir, 'calibration_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Wrote calibration_summary.json')
