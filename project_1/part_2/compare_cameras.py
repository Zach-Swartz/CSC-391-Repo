import argparse
from pathlib import Path
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import rawpy

BASE = Path(__file__).resolve().parent  # BASE: this script's directory
IMG_DIR = BASE / 'images'  # IMG_DIR: input DNG images
RESULTS = BASE / 'results'  # RESULTS: outputs directory for CSV/summary
FIGS = RESULTS / 'figs'  # FIGS: subfolder to save PNG figures
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


# Compute horizontal field-of-view (degrees) from focal length and sensor width
def horizontal_fov(focal_mm, sensor_width_mm):
    # compute horizontal field-of-view in degrees
    return 2.0 * np.degrees(np.arctan(sensor_width_mm / (2.0 * focal_mm)))


# Read a raw DNG file and return the raw image as a float32 numpy array
def read_raw(path):
    # read DNG and return raw sensor image as float32
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image.copy().astype(np.float32)  # raw image array


# Extract a square center patch of size `patch_size` from image `img`
def center_patch(img, patch_size):
    h, w = img.shape  # h,w: image height and width
    cx, cy = w // 2, h // 2  # cx,cy: center coordinates
    half = patch_size // 2  # half: half-size of square patch
    x0 = max(0, cx - half)  # x0: left coordinate
    y0 = max(0, cy - half)  # y0: top coordinate
    x1 = min(w, x0 + patch_size)  # x1: right coordinate (exclusive)
    y1 = min(h, y0 + patch_size)  # y1: bottom coordinate (exclusive)
    return img[y0:y1, x0:x1]  # return: cropped center patch


# Save a patch as a grayscale PNG after normalizing its range to 0-255
def save_patch_png(patch, outpath):
    # normalize for PNG visibility
    m, M = float(patch.min()), float(patch.max())  # min and max pixel values
    if M - m <= 0:
        img = np.zeros_like(patch, dtype=np.uint8)  # flat patch -> black image
    else:
        img = ((patch - m) / (M - m) * 255.0).astype(np.uint8)  # scaled to 0-255
    plt.imsave(outpath, img, cmap='gray')


# Create an annotated histogram from a patch and save it to `outpath`
def annotate_and_save_hist(patch, title, outpath, exposures_text=None):
    data = patch.ravel()  # flattened pixel values from the patch
    fig, ax = plt.subplots(figsize=(6, 4))  # figure and axis for plotting
    ax.hist(data, bins=80, range=(float(data.min()),
                                  # histogram
                                  float(data.max())), color='#4c72b0')
    ax.set_title(title)  # set plot title
    ax.set_xlabel('Pixel value')  # x-axis label
    ax.set_ylabel('Frequency')  # y-axis label
    mu = float(np.mean(patch))  # patch mean
    sigma = float(np.std(patch))  # patch standard deviation
    ax.text(0.98, 0.95, f'μ={mu:.2f}\nσ={sigma:.2f}', transform=ax.transAxes,
            ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    if exposures_text:
        ax.text(0.01, 0.95, exposures_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(outpath)  # save annotated histogram
    plt.close(fig)


# Create a side-by-side histogram figure comparing main and tele patches
def combined_hist_figure(patch_main, patch_tele, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # two-panel figure
    for ax, patch, name in zip(axes, (patch_main, patch_tele), ('main', 'tele')):
        data = patch.ravel()  # flattened pixel values
        ax.hist(data, bins=80, color='#4c72b0')  # draw histogram
        ax.set_title(name)
        ax.set_xlabel('Pixel value')
        ax.set_ylabel('Frequency')
        mu = float(np.mean(patch))  # mean for the panel
        sigma = float(np.std(patch))  # std for the panel
        ax.text(0.98, 0.95, f'μ={mu:.1f}\nσ={sigma:.1f}', transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(outpath)  # write combined histogram image
    plt.close(fig)


# Load a DNG file, extract the center patch, and compute its mean and stddev
def analyze_file(path, patch_size):
    arr = read_raw(path)  # full raw array from the DNG
    patch = center_patch(arr, patch_size)  # extracted center patch
    mu = float(np.mean(patch))  # patch mean
    sigma = float(np.std(patch))  # patch standard deviation
    return arr, patch, mu, sigma


# Note: I had AI do this part, since I could not find the specific pixel values
# and had to use the grids on photoshop. The below creates a general grid of the image
# and selects the appropriate region.
# Map Photoshop grid cell ranges to a pixel bounding box and compute a patch inside it
def bbox_from_photoshop_grid(img_shape, col_min, col_max, row_min, row_max, ncols=18, nrows=24):
    """
    img_shape: (h, w)
    col_min/col_max, row_min/row_max: 1-based inclusive grid indices
    ncols, nrows: grid dimensions (columns x rows)
    returns: (x0, y0, x1, y1) integer pixel bbox (x1,y1 exclusive)
    """
    h, w = img_shape
    # size of one grid cell (may be fractional)
    cell_w = w / float(ncols)
    cell_h = h / float(nrows)
    # convert 1-based inclusive grid indices to pixel bbox
    x0 = int(round((col_min - 1) * cell_w))
    x1 = int(round(col_max * cell_w))
    y0 = int(round((row_min - 1) * cell_h))
    y1 = int(round(row_max * cell_h))
    # clamp
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    return x0, y0, x1, y1


def patch_within_bbox(img, bbox, desired_patch):
    x0, y0, x1, y1 = bbox
    h, w = img.shape
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    # choose patch size no larger than bbox dimensions
    ps = int(min(desired_patch, bbox_w, bbox_h))
    if ps <= 0:
        # fallback to center patch
        return center_patch(img, desired_patch)
    # center of bbox
    cx = x0 + bbox_w // 2
    cy = y0 + bbox_h // 2
    px0 = int(max(x0, min(cx - ps // 2, x1 - ps)))
    py0 = int(max(y0, min(cy - ps // 2, y1 - ps)))
    return img[py0:py0 + ps, px0:px0 + ps]


def main():
    p = argparse.ArgumentParser(
        description='Compare cameras: FOV and noise analysis')
    p.add_argument('--patch', type=int, default=128,
                   help='center patch size in pixels')
    p.add_argument('--main-foc', type=float, default=None,
                   help='main focal length (mm)')
    p.add_argument('--main-sensor-w', type=float,
                   default=None, help='main sensor width (mm)')
    p.add_argument('--tele-foc', type=float, default=None,
                   help='tele focal length (mm)')
    p.add_argument('--tele-sensor-w', type=float,
                   default=None, help='tele sensor width (mm)')
    p.add_argument('--main-shutter', type=float, default=1.0 /
                   40.0, help='main shutter seconds')
    p.add_argument('--main-aperture', type=float,
                   default=1.8, help='main aperture f-number')
    p.add_argument('--main-iso', type=float, default=1000.0, help='main ISO')
    p.add_argument('--tele-shutter', type=float, default=1.0 /
                   20.0, help='tele shutter seconds')
    p.add_argument('--tele-aperture', type=float,
                   default=2.8, help='tele aperture f-number')
    p.add_argument('--tele-iso', type=float, default=640.0, help='tele ISO')
    p.add_argument('--use-ps-grid', action='store_true',
                   help='use Photoshop 18x24 grid coordinates provided in the prompt')
    args = p.parse_args()

    main_path = IMG_DIR / 'main_camera.dng'  # expected main DNG input path
    tele_path = IMG_DIR / 'telephoto_camera.dng'  # expected telephoto DNG input path
    if not main_path.exists() or not tele_path.exists():
        print('Required DNGs not found in', IMG_DIR)
        sys.exit(1)

    arr_main = read_raw(main_path)
    arr_tele = read_raw(tele_path)

    if args.use_ps_grid:
        # Photoshop grid mapping provided by user (1-based indices)
        # Telephoto: columns 8-10, rows 16-17
        t_x0, t_y0, t_x1, t_y1 = bbox_from_photoshop_grid(
            arr_tele.shape, col_min=8, col_max=10, row_min=16, row_max=17)
        patch_tele = patch_within_bbox(
            arr_tele, (t_x0, t_y0, t_x1, t_y1), args.patch)
        mu_tele = float(np.mean(patch_tele))
        sigma_tele = float(np.std(patch_tele))

        # Main: columns 8-10, row 19 (interpreted as rows 19-19)
        m_x0, m_y0, m_x1, m_y1 = bbox_from_photoshop_grid(
            arr_main.shape, col_min=8, col_max=10, row_min=19, row_max=19)
        patch_main = patch_within_bbox(
            arr_main, (m_x0, m_y0, m_x1, m_y1), args.patch)
        mu_main = float(np.mean(patch_main))
        sigma_main = float(np.std(patch_main))
    else:
        arr_main, patch_main, mu_main, sigma_main = analyze_file(
            main_path, args.patch)  # analyze main camera image
        arr_tele, patch_tele, mu_tele, sigma_tele = analyze_file(
            tele_path, args.patch)  # analyze telephoto camera image

    # relative exposure proxy: shutter_time / (f_number^2)
    rel_exp_main = args.main_shutter / \
        (args.main_aperture ** 2) if args.main_aperture and args.main_shutter else None
    rel_exp_tele = args.tele_shutter / \
        (args.tele_aperture ** 2) if args.tele_aperture and args.tele_shutter else None
    snr_main = mu_main / \
        sigma_main if sigma_main > 0 else float('inf')  # SNR proxy (μ/σ)
    snr_tele = mu_tele / \
        sigma_tele if sigma_tele > 0 else float('inf')  # SNR proxy (μ/σ)

    # compute horizontal FOV when focal length and sensor width are provided
    fov_main = horizontal_fov(
        args.main_foc, args.main_sensor_w) if args.main_foc and args.main_sensor_w else None
    fov_tele = horizontal_fov(
        args.tele_foc, args.tele_sensor_w) if args.tele_foc and args.tele_sensor_w else None

    # save patch crops for visual inspection
    save_patch_png(patch_main, FIGS / 'main_patch_crop.png')
    save_patch_png(patch_tele, FIGS / 'tele_patch_crop.png')

    # annotated histograms
    # formatted exposure strings for annotation
    expos_main = f'shutter={args.main_shutter:.4f}s\naperture=f/{args.main_aperture:.1f}\nISO={args.main_iso:.0f}\nSNR={snr_main:.2f}'
    expos_tele = f'shutter={args.tele_shutter:.4f}s\naperture=f/{args.tele_aperture:.1f}\nISO={args.tele_iso:.0f}\nSNR={snr_tele:.2f}'
    annotate_and_save_hist(patch_main, 'main_camera.dng', FIGS /
                           'main_patch_hist_annot.png', exposures_text=expos_main)
    annotate_and_save_hist(patch_tele, 'telephoto_camera.dng',
                           FIGS / 'tele_patch_hist_annot.png', exposures_text=expos_tele)
    combined_hist_figure(patch_main, patch_tele, FIGS / 'combined_hist.png')

    # write CSV with expanded columns
    csv_path = RESULTS / 'camera_compare_summary.csv'  # output CSV path
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        # header row for CSV
        w.writerow(['camera', 'patch_mean', 'patch_std', 'snr', 'shutter_s', 'aperture_f',
                   'iso', 'rel_exposure', 'focal_mm', 'sensor_width_mm', 'h_fov_deg'])
        # main camera row
        w.writerow(['main', f'{mu_main:.4f}', f'{sigma_main:.4f}', f'{snr_main:.4f}', f'{args.main_shutter:.6f}', f'{args.main_aperture:.2f}', f'{args.main_iso:.1f}',
                   f'{rel_exp_main:.8f}' if rel_exp_main is not None else '', args.main_foc or '', args.main_sensor_w or '', f'{fov_main:.4f}' if fov_main else ''])
        # telephoto camera row
        w.writerow(['tele', f'{mu_tele:.4f}', f'{sigma_tele:.4f}', f'{snr_tele:.4f}', f'{args.tele_shutter:.6f}', f'{args.tele_aperture:.2f}', f'{args.tele_iso:.1f}',
                   f'{rel_exp_tele:.8f}' if rel_exp_tele is not None else '', args.tele_foc or '', args.tele_sensor_w or '', f'{fov_tele:.4f}' if fov_tele else ''])

    print('\nSaved:')
    print(' -', csv_path)
    for fn in ('main_patch_crop.png', 'tele_patch_crop.png', 'main_patch_hist_annot.png', 'tele_patch_hist_annot.png', 'combined_hist.png'):
        print(' -', FIGS / fn)

    print('\nSummary:')
    print(
        f"Main mean={mu_main:.2f}, std={sigma_main:.2f}, SNR={snr_main:.2f}, rel_exp={rel_exp_main:.6e}")
    print(
        f"Tele mean={mu_tele:.2f}, std={sigma_tele:.2f}, SNR={snr_tele:.2f}, rel_exp={rel_exp_tele:.6e}")
    if fov_main:
        print(f"Main FOV={fov_main:.2f} deg")
    if fov_tele:
        print(f"Tele FOV={fov_tele:.2f} deg")


if __name__ == '__main__':
    main()
