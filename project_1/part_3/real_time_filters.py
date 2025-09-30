import argparse
import cv2
import numpy as np


# Apply convolution to a single-channel image using a manual pixel loop
def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape  # kernel height/width

    h, w = image.shape  # image height and width
    pad = kh // 2  # padding size

    padded = np.pad(image.astype(np.float32), ((pad, pad),
                    (pad, pad)), mode='edge')  # padded image
    out = np.zeros_like(image, dtype=np.float32)  # accumulator
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)  # flipped kernel

    for y in range(h):
        for x in range(w):
            region = padded[y:y + kh, x:x + kw]  # region under kernel
            out[y, x] = np.sum(region * k)

    out = np.clip(out, 0, 255)  # clamp to [0,255]
    return out.astype(np.uint8)


# Create a side-by-side image for display
def stack_side_by_side(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    if img_a.ndim == 2:
        a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2BGR)  # convert gray to BGR
    else:
        a = img_a
    if img_b.ndim == 2:
        b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2BGR)  # convert gray to BGR
    else:
        b = img_b
    if a.shape[0] != b.shape[0]:
        h = min(a.shape[0], b.shape[0])  # normalize height
        a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    return np.hstack((a, b))


# Tile multiple filtered images (plus original) into a grid for display
def tile_filters(original: np.ndarray, filtered_map: dict, cols: int = 3) -> np.ndarray:
    names = list(filtered_map.keys())
    imgs = [filtered_map[n] for n in names]
    bgrs = [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim ==
            2 else im for im in imgs]  # convert each to BGR
    orig_bgr = original if original.ndim == 3 else cv2.cvtColor(
        original, cv2.COLOR_GRAY2BGR)  # original as BGR
    tiles = [orig_bgr] + bgrs
    h = min(t.shape[0] for t in tiles)  # normalize height
    norm = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
            for t in tiles]
    labeled = []
    for i, img in enumerate(norm):
        name = 'Original' if i == 0 else names[i - 1]
        out = img.copy()
        cv2.putText(out, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)  # label
        labeled.append(out)
    rows = []
    for r in range(0, len(labeled), cols):
        row_imgs = labeled[r:r + cols]
        if len(row_imgs) < cols:
            for _ in range(cols - len(row_imgs)):
                row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)
    return grid


# Probe camera indices/backends and return an opened VideoCapture or None
def open_camera(preferred_index=0, max_index=5, try_backends=True):
    cap = cv2.VideoCapture(preferred_index)  # try preferred index
    if cap.isOpened():
        return cap
    for idx in range(0, max_index + 1):
        cap = cv2.VideoCapture(idx)  # try other indices
        if cap.isOpened():
            print(f"Opened camera at index {idx}")
            return cap
    if try_backends and hasattr(cv2, 'CAP_DSHOW'):
        for idx in range(0, max_index + 1):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # try DirectShow
            if cap.isOpened():
                print(f"Opened camera at index {idx} with CAP_DSHOW")
                return cap
        if hasattr(cv2, 'CAP_MSMF'):
            for idx in range(0, max_index + 1):
                # try Media Foundation
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                if cap.isOpened():
                    print(f"Opened camera at index {idx} with CAP_MSMF")
                    return cap
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proc-width', type=int, default=320,
                   help='width in pixels to downscale frames for processing (speeds up convolution)')
    p.add_argument('--manual', action='store_true',
                   help='use the manual Python convolution implementation (slower)')
    p.add_argument('--filter', type=str, default='box',
                   help='name of the filter to show in single-view (box, gaussian, sobel_h, sobel_v, sharpen, emboss)')
    args = p.parse_args()

    # Define kernels
    box = (1.0 / 9.0) * np.array([[1, 1, 1],
                                  [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    gaussian = (1.0 / 16.0) * \
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

    # initial filter selection
    filters = {
        'box': box,
        'gaussian': gaussian,
        'sobel_h': sobel_h,
        'sobel_v': sobel_v,
        'sharpen': sharpen,
        'emboss': emboss,
    }
    filter_names = list(filters.keys())
    # determine which filter to show in single-view; validate CLI value
    if args.filter not in filters:
        print(
            f"Warning: requested filter '{args.filter}' not found. Falling back to 'box'. Available: {', '.join(filter_names)}")
        current_idx = 0
    else:
        current_idx = filter_names.index(args.filter)

    cap = open_camera(preferred_index=0, max_index=5, try_backends=True)
    if cap is None:
        return

    show_all = True
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downscale for processing to speed up convolution
        orig_h, orig_w = gray.shape
        proc_w = int(min(args.proc_width, orig_w))
        scale = proc_w / float(orig_w) if orig_w > 0 else 1.0
        if scale < 1.0:
            proc_h = int(orig_h * scale)
            gray_proc = cv2.resize(
                gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            gray_proc = gray

        # Compute only the needed filtered images to save time
        filtered_map = {}
        if show_all:
            for name, kernel in filters.items():
                if args.manual:
                    # manual Python implementation (slower)
                    f = apply_convolution(gray_proc, kernel)
                else:
                    # use OpenCV's optimized filter2D for speed
                    f = cv2.filter2D(gray_proc, -1, kernel)
                filtered_map[name] = f

        # Always compute the currently selected filter for single view
        if filter_names[current_idx] not in filtered_map:
            if args.manual:
                filtered = apply_convolution(
                    gray_proc, filters[filter_names[current_idx]])
            else:
                filtered = cv2.filter2D(
                    gray_proc, -1, filters[filter_names[current_idx]])
        else:
            filtered = filtered_map[filter_names[current_idx]]

        # For Sobel, scale result for visibility (abs and normalize)
        if filter_names[current_idx] in ('sobel_h', 'sobel_v'):
            filtered = cv2.convertScaleAbs(filtered)

        if show_all:
            # upsample filtered_map images to a reasonable display size (use half height)
            display_map = {k: cv2.resize(v, (int(orig_w * 0.5), int(orig_h * 0.5)),
                                         interpolation=cv2.INTER_LINEAR) for k, v in filtered_map.items()}
            side = tile_filters(frame, display_map, cols=3)
        else:
            # upsample filtered to original frame size for side-by-side display
            filtered_disp = cv2.resize(
                filtered, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            side = stack_side_by_side(frame, filtered_disp)
            label = f"Filter: {filter_names[current_idx]}"
            cv2.putText(side, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        win_name = 'Original | Filtered'
        cv2.imshow(win_name, side)

        key = cv2.waitKey(1) & 0xFF
        # If the user closed the window using the window manager, exit loop
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                # If user closed the single-view window, toggle to all-filters view instead of exiting
                if not show_all:
                    show_all = True
                    # continue loop to open the tiled view
                    continue
                else:
                    break
        except Exception:
            # If getWindowProperty is unsupported on some builds, ignore
            pass
        if key == ord('q'):
            break
        elif key == ord('a'):
            show_all = not show_all

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
