import os
import argparse
import json
import glob
import cv2
import numpy as np
from datetime import datetime


# compute sharpness using Laplacian variance
def laplacian_variance(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


# attempt chessboard detection with standard flags
def detect_chessboard(gray_image, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    try:
        found, corners = cv2.findChessboardCorners(gray_image, pattern, flags)
    except Exception:
        found, corners = False, None
    return found, corners


# main processing: try multiple preprocessing methods to rescue detections
def main():
    parser = argparse.ArgumentParser(description='Process chessboard images and save ones with detectable corners')
    parser.add_argument('--images_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'images'))
    default_out = os.path.join(os.path.dirname(__file__), 'calibration_image')
    parser.add_argument('--out_dir', type=str, default=default_out)
    parser.add_argument('--pattern', type=int, nargs=2, default=[7, 6])
    parser.add_argument('--save_visuals', action='store_true')
    args = parser.parse_args()

    images_dir = args.images_dir
    out_dir = args.out_dir
    pattern = (args.pattern[0], args.pattern[1])

    os.makedirs(out_dir, exist_ok=True)

    # gather image files from the source folder
    files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
    if not files:
        print('No images found in', images_dir)
        return

    results = []
    success_count = 0

    for fpath in files:
        base = os.path.basename(fpath)
        img = cv2.imread(fpath)
        if img is None:
            print('Could not read', fpath)
            results.append({'file': base, 'status': 'read-fail'})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = laplacian_variance(gray)

        # build a list of preprocessing candidates to try
        candidates = []
        candidates.append(('orig', gray))
        try:
            hist_eq = cv2.equalizeHist(gray)
            candidates.append(('equalize', hist_eq))
        except Exception:
            pass
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            candidates.append(('clahe', clahe_img))
        except Exception:
            pass
        try:
            adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            candidates.append(('adapt', adapt))
        except Exception:
            pass

        detected = False
        detected_label = None
        detected_corners = None

        # try each preprocessing candidate until we find corners
        for label, proc_gray in candidates:
            found, corners = detect_chessboard(proc_gray, pattern)
            if found:
                # choose a gray image for subpixel refinement
                refine_gray = proc_gray if proc_gray.dtype == np.uint8 and proc_gray.ndim == 2 else gray
                try:
                    corners_refined = cv2.cornerSubPix(refine_gray, corners, (11, 11), (-1, -1),
                                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                except Exception:
                    corners_refined = corners

                detected = True
                detected_label = label
                detected_corners = corners_refined
                break

        if detected:
            success_count += 1
            out_path = os.path.join(out_dir, base)
            cv2.imwrite(out_path, img)

            visual_name = None
            if args.save_visuals:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, pattern, detected_corners, True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                visual_name = os.path.join(out_dir, f'visual_{detected_label}_{ts}_{base}')
                cv2.imwrite(visual_name, vis)

            results.append({'file': base, 'status': 'ok', 'method': detected_label, 'sharp': float(sharpness), 'visual': os.path.basename(visual_name) if visual_name else None})
            print(f'[OK] {base} via {detected_label} (sharp={sharpness:.1f})')
        else:
            results.append({'file': base, 'status': 'fail', 'sharp': float(sharpness)})
            print(f'[FAIL] {base} (sharp={sharpness:.1f})')

    # write a summary JSON with processing results
    summary = {
        'processed': len(files),
        'success': success_count,
        'fail': len(files) - success_count,
        'results': results
    }
    with open(os.path.join(out_dir, 'processing_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\nDone. Summary saved to', os.path.join(out_dir, 'processing_summary.json'))


if __name__ == '__main__':
    main()
