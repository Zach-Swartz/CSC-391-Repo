import os
import time
import argparse
import threading
import cv2
import numpy as np


# Background reader: continuously capture frames from a camera device
class CameraBackgroundReader:
    def __init__(self, camera_index=0, requested_width=None, requested_height=None, backend_flag=None):
        self.camera_index = camera_index
        self.backend_flag = backend_flag
        try:
            if backend_flag is not None:
                self.capture = cv2.VideoCapture(camera_index, int(backend_flag))
            else:
                self.capture = cv2.VideoCapture(camera_index)
        except Exception:
            self.capture = cv2.VideoCapture(camera_index)
        if requested_width:
            try:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(requested_width))
            except Exception:
                pass
        if requested_height:
            try:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(requested_height))
            except Exception:
                pass
        self.lock = threading.Lock()
        self.latest_frame = None
        self.is_paused = False
        self.is_stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        failures = 0
        while not self.is_stopped:
            if getattr(self, 'is_paused', False):
                time.sleep(0.05)
                continue
            # read a frame from the capture device
            try:
                ok, frame = self.capture.read()
            except Exception:
                ok = False
                frame = None
            if not ok or frame is None:
                failures += 1
                time.sleep(min(0.05 + failures * 0.01, 0.5))
                if failures > 200:
                    try:
                        self.capture.release()
                    except Exception:
                        pass
                    try:
                        if self.backend_flag is not None:
                            self.capture = cv2.VideoCapture(self.camera_index, int(self.backend_flag))
                        else:
                            self.capture = cv2.VideoCapture(self.camera_index)
                    except Exception:
                        pass
                    failures = 0
                continue
            failures = 0
            with self.lock:
                self.latest_frame = frame

    def read(self):
        # return a copy of the latest frame in a thread-safe way
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def release(self):
        self.is_stopped = True
        try:
            self.capture.release()
        except Exception:
            pass


# Compute Laplacian variance for sharpness
def laplacian_variance(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


# Compute coverage ratio for a set of corners over the image area
def compute_coverage_ratio(corners, image_width, image_height):
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    bbox_area = max(0.0, (maxx - minx)) * max(0.0, (maxy - miny))
    img_area = float(image_width) * float(image_height) if (image_width > 0 and image_height > 0) else 1.0
    return bbox_area / img_area


# Capture a single frame from a fresh VideoCapture instance
def capture_single_frame_direct(camera_index, requested_width, requested_height, backend_flag):
    try:
        if backend_flag is not None:
            cap = cv2.VideoCapture(int(camera_index), int(backend_flag))
        else:
            cap = cv2.VideoCapture(int(camera_index))
    except Exception:
        cap = cv2.VideoCapture(int(camera_index))
    if requested_width:
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(requested_width))
        except Exception:
            pass
    if requested_height:
        try:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(requested_height))
        except Exception:
            pass
    # discard a few warm-up frames to allow exposure and auto adjustments
    for _ in range(3):
        cap.read()
    ok, frame = cap.read()
    try:
        cap.release()
    except Exception:
        pass
    if not ok:
        return None
    return frame


# Determine the next chessboard image index in the output folder
def next_filename_index(output_dir):
    # determine the next numeric index by scanning existing chessboard files
    existing = [f for f in os.listdir(output_dir) if f.startswith('chessboard_') and f.lower().endswith('.jpg')]
    index = 1
    if existing:
        nums = []
        for f in existing:
            try:
                nums.append(int(f.split('_')[-1].split('.')[0]))
            except Exception:
                pass
        if nums:
            index = max(nums) + 1
    return index


# CLI loop: live preview, detect chessboard corners, and save images on keypress or auto conditions
def main():
    # main entry: parse CLI arguments and prepare camera reader
    parser = argparse.ArgumentParser(description='Live capture chessboard images for calibration')
    parser.add_argument('--pattern', type=int, nargs=2, default=[7, 6])
    default_out = os.path.join(os.path.dirname(__file__), 'calibration_image')
    parser.add_argument('--outdir', type=str, default=default_out)
    parser.add_argument('--min_sharp', type=float, default=50.0)
    parser.add_argument('--min_coverage', type=float, default=0.2)
    parser.add_argument('--stable_frames', type=int, default=3)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', type=str, default='dshow', choices=['dshow', 'msmf', 'any'])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--use_clahe', action='store_true')
    parser.add_argument('--preview_fps', type=float, default=6.0)
    parser.add_argument('--use_direct_capture', action='store_true')
    args = parser.parse_args()

    pattern = (args.pattern[0], args.pattern[1])
    os.makedirs(args.outdir, exist_ok=True)

    backend_flag = None
    if args.backend == 'dshow' and hasattr(cv2, 'CAP_DSHOW'):
        backend_flag = cv2.CAP_DSHOW
    elif args.backend == 'msmf' and hasattr(cv2, 'CAP_MSMF'):
        backend_flag = cv2.CAP_MSMF
    else:
        backend_flag = cv2.CAP_ANY

    reader = CameraBackgroundReader(args.camera, requested_width=args.width, requested_height=args.height, backend_flag=backend_flag)
    time.sleep(0.2)

    print('Press SPACE to force-save, s to save on detection/sharpness, q to quit')

    last_detection_time = 0.0
    detection_interval = 1.0 / max(0.1, float(args.preview_fps))
    consecutive_good_frames = 0
    saved_count = 0
    last_corners_time = 0.0
    last_corners = None

    window_created = False

    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = laplacian_variance(gray)
            height, width = gray.shape[:2]

            found = False
            corners = None
            now = time.time()
            if now - last_detection_time >= detection_interval:
                last_detection_time = now
                proc_gray = gray
                if args.use_clahe:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    proc_gray = clahe.apply(proc_gray)
                try:
                    found, corners = cv2.findChessboardCorners(proc_gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                except Exception:
                    found = False
                    corners = None

            if found:
                try:
                    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                except Exception:
                    refined = corners
                cv2.drawChessboardCorners(display_frame, pattern, refined, True)
                coverage = compute_coverage_ratio(refined, width, height)
                cv2.putText(display_frame, f'Sharp={sharpness:.1f} Cov={coverage:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                last_corners_time = time.time()
                last_corners = refined.copy()
                if sharpness >= args.min_sharp and coverage >= args.min_coverage:
                    consecutive_good_frames += 1
                else:
                    consecutive_good_frames = 0
            else:
                cv2.putText(display_frame, f'Sharp={sharpness:.1f} Corners=0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                consecutive_good_frames = 0

            if not window_created:
                cv2.namedWindow('capture', cv2.WINDOW_NORMAL)
                window_created = True
            cv2.imshow('capture', display_frame)

            # poll for user keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # user pressed SPACE — force save current frame
            elif key == 32:
                f = None
                if args.use_direct_capture:
                    try:
                        reader.pause()
                        time.sleep(0.12)
                        f = capture_single_frame_direct(args.camera, args.width, args.height, backend_flag)
                    finally:
                        reader.resume()
                    if f is None:
                        ok2, pf = reader.read()
                        if ok2:
                            f = pf
                else:
                    f = frame
                if f is not None:
                    idx = next_filename_index(args.outdir)
                    fname = os.path.join(args.outdir, f'chessboard_{idx}.jpg')
                    try:
                        cv2.imwrite(fname, f)
                        saved_count += 1
                        print('Saved (manual):', fname)
                    except Exception as e:
                        print('Save failed (write error):', e)
                else:
                    print('Save failed: could not acquire frame')

            elif key == ord('s'):
                recent_ok = (time.time() - last_corners_time) <= 1.0
                sharp_ok = (sharpness >= args.min_sharp)
                if found or recent_ok or sharp_ok:
                    f = None
                    if args.use_direct_capture:
                        try:
                            reader.pause()
                            time.sleep(0.12)
                            f = capture_single_frame_direct(args.camera, args.width, args.height, backend_flag)
                        finally:
                            reader.resume()
                        if f is None:
                            ok2, pf = reader.read()
                            if ok2:
                                f = pf
                    else:
                        f = frame
                    if f is not None:
                        idx = next_filename_index(args.outdir)
                        fname = os.path.join(args.outdir, f'chessboard_{idx}.jpg')
                        try:
                            cv2.imwrite(fname, f)
                            saved_count += 1
                            mode = 'manual-found' if (found or recent_ok) else 'manual-sharp'
                            print(f'Saved ({mode}):', fname)
                        except Exception as e:
                            print('Save failed (write error):', e)
                    else:
                        print('Save failed: could not acquire frame')
                else:
                    print('No recent corners and sharpness below threshold; not saving')
            # auto-save condition based on consecutive good frames
            elif consecutive_good_frames >= args.stable_frames:
                f = None
                if args.use_direct_capture:
                    try:
                        reader.pause()
                        time.sleep(0.12)
                        f = capture_single_frame_direct(args.camera, args.width, args.height, backend_flag)
                    finally:
                        reader.resume()
                    if f is None:
                        ok2, pf = reader.read()
                        if ok2:
                            f = pf
                else:
                    f = frame
                if f is not None:
                    idx = next_filename_index(args.outdir)
                    fname = os.path.join(args.outdir, f'chessboard_{idx}.jpg')
                    try:
                        cv2.imwrite(fname, f)
                        saved_count += 1
                        print('Saved (auto):', fname)
                    except Exception as e:
                        print('Auto-save failed (write error):', e)
                else:
                    print('Auto-save failed: could not acquire frame')
                consecutive_good_frames = 0
    except KeyboardInterrupt:
        pass
    finally:
        try:
            reader.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print('Done. Saved', saved_count, 'images to', args.outdir)


if __name__ == '__main__':
    main()
