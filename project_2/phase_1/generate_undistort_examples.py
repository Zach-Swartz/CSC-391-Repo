import os
import argparse
import numpy as np
import cv2


# load camera matrix and distortion coefficients from .npz file
def load_calibration_data(npz_path):
    arr = np.load(npz_path, allow_pickle=True)
    # try common keys first
    if 'mtx' in arr:
        camera_matrix = arr['mtx']
        dist_coeffs = arr['dist']
    elif 'camera_matrix' in arr:
        camera_matrix = arr['camera_matrix']
        dist_coeffs = arr['dist_coeffs']
    else:
        # fall back to positional arrays if saved without names
        camera_matrix = arr[arr.files[0]]
        dist_coeffs = arr[arr.files[1]]
    return camera_matrix, dist_coeffs


# create a side-by-side comparison image showing original and undistorted
def make_comparison_image(original, undistorted):
    height, width = original.shape[:2]
    left = cv2.resize(original, (width // 2, height // 2))
    right = cv2.resize(undistorted, (width // 2, height // 2))
    combined = np.hstack((left, right))
    # add simple labels to the combined image
    try:
        cv2.putText(combined, 'ORIGINAL', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, 'UNDISTORTED', (width // 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception:
        pass
    return combined


# main program: undistort each calibrated image and write comparison images
def main():
    parser = argparse.ArgumentParser()
    # calibrated images were moved into the phase_1 results folder and renamed
    parser.add_argument('--images_dir', default=os.path.join('results', 'calibrated images'))
    parser.add_argument('--calibration', default=os.path.join('results', 'calibration_results.npz'))
    parser.add_argument('--out_dir', default=os.path.join('results', 'undistort_examples'))
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    calib_file = os.path.abspath(args.calibration)
    output_dir = os.path.abspath(args.out_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(calib_file):
        print('Calibration file not found:', calib_file)
        return

    camera_matrix, dist_coeffs = load_calibration_data(calib_file)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith('.jpg') and f.startswith('calibrated_chessboard_') and not f.endswith('_visual.jpg')
    ])
    if not image_files:
        print('No calibrated images found in', images_dir)
        return

    for name in image_files:
        src_path = os.path.join(images_dir, name)
        img = cv2.imread(src_path)
        if img is None:
            print('Could not read', src_path)
            continue
        h, w = img.shape[:2]
        # compute an optimal new camera matrix for undistortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        und = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        # basic difference metrics to detect if undistortion changed the image
        diff = cv2.absdiff(img, und)
        mean_diff = float(diff.mean())
        max_diff = int(diff.max())
        combined = make_comparison_image(img, und)
        out_name = os.path.join(output_dir, f'compare_{name}')
        cv2.imwrite(out_name, combined)
        print(f'Wrote {out_name}  mean_diff={mean_diff:.3f} max_diff={max_diff}')
        if mean_diff < 1.0:
            print('  Warning: very small mean difference (undistort appears nearly identical)')


if __name__ == '__main__':
    main()
