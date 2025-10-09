import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rawpy


# BASE_DIR: folder containing this script
BASE_DIR = Path(__file__).resolve().parent  # BASE_DIR: project part_1 folder
# IMG_DIR: folder with DNG input files
IMG_DIR = BASE_DIR / "images"  # IMG_DIR: input images directory
# RESULTS_DIR: output folder for CSVs/figs
RESULTS_DIR = BASE_DIR / "results"  # RESULTS_DIR: outputs directory
# FIG_DIR: subfolder for PNG figures
FIG_DIR = RESULTS_DIR / "figs"  # FIG_DIR: figures directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# analyze_raw_image: read DNG file and return center-patch mean and std
def analyze_raw_image(file_path, patch_size=100, save_fig=True):
    # print filename being processed
    print(f"\nFile: {os.path.basename(file_path)}")
    try:
        # load raw DNG into float32 numpy array
        with rawpy.imread(file_path) as raw:
            # raw_image: float32 raw sensor image
            raw_image = raw.raw_image.copy().astype(np.float32)
    except Exception as e:
        try:
            fsize = os.path.getsize(file_path)  # fsize: file size in bytes
        except Exception:
            fsize = "unknown"
        print(f"Failed to read raw file: {file_path}")
        print(f"  Error: {e!r}")
        print(f"  File size: {fsize}")
        return None, None

    h, w = raw_image.shape  # image height and width
    cx, cy = w // 2, h // 2  # center coordinates
    half = patch_size // 2  # half-size of square patch
    x0 = max(0, cx - half)  # left edge of patch
    y0 = max(0, cy - half)  # top edge of patch
    x1 = min(w, x0 + patch_size)  # right edge of patch
    y1 = min(h, y0 + patch_size)  # bottom edge of patch

    patch = raw_image[y0:y1, x0:x1]  # center patch array
    mean_val = float(np.mean(patch))  # patch mean
    std_val = float(np.std(patch))  # patch stddev

    print("Shape:", raw_image.shape)
    print("Type:", raw_image.dtype)
    print("Min:", raw_image.min(), "Max:", raw_image.max())
    print(f"Patch mean: {mean_val:.2f}, std: {std_val:.2f}")

    plt.figure(figsize=(6, 4))
    plt.hist(patch.ravel(), bins=60, range=(
        float(patch.min()), float(patch.max())))
    plt.title(os.path.basename(file_path))
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    if save_fig:
        out_path = FIG_DIR / (os.path.basename(file_path) + "_hist.png")
        plt.savefig(out_path)
        print(f"Saved histogram to {out_path}")
    plt.close()

    return mean_val, std_val


def get_center_patch(file_path, patch_size=100):
    """Return the center patch as a numpy array or None on read error."""
    try:
        with rawpy.imread(file_path) as raw:
            raw_image = raw.raw_image.copy().astype(
                np.float32)  # raw sensor image
    except Exception as e:
        print(
            f"Failed to read raw file for patch extraction: {file_path}: {e!r}")
        return None

    h, w = raw_image.shape  # image height and width
    cx, cy = w // 2, h // 2  # center coordinates
    half = patch_size // 2  # half-size of square patch
    x0 = max(0, cx - half)  # left edge
    y0 = max(0, cy - half)  # top edge
    x1 = min(w, x0 + patch_size)  # right edge
    y1 = min(h, y0 + patch_size)  # bottom edge
    return raw_image[y0:y1, x0:x1]


def main():
    """Parse arguments, analyze images, and create/save report plots."""
    parser = argparse.ArgumentParser(
        description="Analyze DNG images in project_1/part_1/images")
    parser.add_argument("--patch", type=int, default=100,
                        help="patch size (square) to analyze at center")
    parser.add_argument("--show", action="store_true",
                        help="show matplotlib figures interactively")
    parser.add_argument("--grid", action="store_true",
                        help="create a single 2x3 grid of all center-patch histograms")
    parser.add_argument("--out-csv", type=str, default=str(RESULTS_DIR / "summary.csv"),
                        help="output CSV filename (path relative to script folder or absolute)")
    args = parser.parse_args()  # parsed CLI arguments

    if not IMG_DIR.exists():
        print(f"Images directory not found: {IMG_DIR}")
        sys.exit(1)

    results = []  # list to hold per-file statistics
    patches_for_grid = []  # list to collect center patches for the 2x3 grid
    for p in sorted(IMG_DIR.iterdir()):
        if not p.name.lower().endswith('.dng'):
            continue
        mean_patch, std_patch = analyze_raw_image(
            str(p), patch_size=args.patch, save_fig=not args.show)

        mean_full = std_full = None  # placeholders for full-image stats
        if rawpy is not None:
            try:
                with rawpy.imread(str(p)) as raw:
                    img = raw.raw_image.copy().astype(np.float32)  # full raw image as float32
                mean_full = float(np.mean(img))  # full-image mean
                std_full = float(np.std(img))  # full-image stddev
            except Exception as e:
                print(
                    f"Failed to compute full-image stats for {p.name}: {e!r}")
                mean_full = std_full = None

        results.append({
            'file': p.name,
            'mean_patch': mean_patch,
            'std_patch': std_patch,
            'mean_full': mean_full,
            'std_full': std_full,
        })
        if args.grid:
            patch = get_center_patch(str(p), patch_size=args.patch)
            patches_for_grid.append((p.name, patch))

    out_csv_path = Path(args.out_csv)  # output CSV path
    if not out_csv_path.is_absolute():
        out_csv_path = BASE_DIR / out_csv_path  # resolve relative path
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(
            cf, fieldnames=['file', 'mean_patch', 'std_patch', 'mean_full', 'std_full'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved CSV summary to {out_csv_path}")
    print("\nSummary:")
    print("{:<20} {:>10} {:>10} {:>12} {:>12}".format(
        "File", "M_patch", "S_patch", "M_full", "S_full"))
    for r in results:
        mf = f"{r['mean_patch']:.2f}" if r['mean_patch'] is not None else "-"
        sf = f"{r['std_patch']:.2f}" if r['std_patch'] is not None else "-"
        mfull = f"{r['mean_full']:.2f}" if r['mean_full'] is not None else "-"
        sfull = f"{r['std_full']:.2f}" if r['std_full'] is not None else "-"
        print("{:<20} {:>10} {:>10} {:>12} {:>12}".format(
            r['file'], mf, sf, mfull, sfull))

    if args.grid:
        n = len(patches_for_grid)  # number of patches collected for the grid
        if n == 0:
            print("No DNG patches available for grid plotting.")
        else:
            cols = 3  # grid columns
            rows = 2  # grid rows
            fig, axes = plt.subplots(
                rows, cols, figsize=(cols * 4.5, rows * 3.2))
            axes = axes.flatten()
            for i in range(rows * cols):
                ax = axes[i]
                if i < n and patches_for_grid[i][1] is not None:
                    patch_name, patch_arr = patches_for_grid[i]
                    data = patch_arr.ravel()  # 1D data for histogram
                    ax.hist(data, bins=60, range=(
                        float(data.min()), float(data.max())))
                    ax.set_title(f"{patch_name}", y=1.03)
                    ax.set_xlabel('Pixel intensity')
                    ax.set_ylabel('Frequency')
                else:
                    ax.axis('off')
            plt.subplots_adjust(top=0.88, hspace=0.45, wspace=0.35)
            combined_path = FIG_DIR / 'combined_hist_grid.png'
            plt.savefig(combined_path)
            print(f"Saved combined grid to {combined_path}")
            if args.show:
                plt.show()
            plt.close()
            try:
                if not args.show:
                    plt.figure()
                img = plt.imread(str(combined_path))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    'Frequencies of Observed Light Intensities')
                plt.show()
            except Exception:
                pass

    try:
        dark_patches = []  # collected dark patches
        light_patches = []  # collected light patches
        for fname, mp, sp, mf, sf in [(r['file'], r['mean_patch'], r['std_patch'], r['mean_full'], r['std_full']) for r in results]:
            if fname.lower().startswith('dark'):
                p = get_center_patch(str(IMG_DIR / fname),
                                     patch_size=args.patch)
                if p is not None:
                    dark_patches.append(p.ravel())
            elif fname.lower().startswith('light'):
                p = get_center_patch(str(IMG_DIR / fname),
                                     patch_size=args.patch)
                if p is not None:
                    light_patches.append(p.ravel())

        if dark_patches or light_patches:
            plt.figure(figsize=(8, 5))
            all_dark = None  # concatenated dark data
            all_light = None  # concatenated light data
            if dark_patches:
                all_dark = np.concatenate(dark_patches)
            if light_patches:
                all_light = np.concatenate(light_patches)

            mins = []
            maxs = []
            if all_dark is not None and all_dark.size > 0:
                mins.append(float(all_dark.min()))
                maxs.append(float(all_dark.max()))
            if all_light is not None and all_light.size > 0:
                mins.append(float(all_light.min()))
                maxs.append(float(all_light.max()))
            if mins and maxs:
                rng = (min(mins), max(maxs))  # common histogram range
            else:
                rng = None

            if all_dark is not None and all_dark.size > 0:
                plt.hist(all_dark, bins=80, alpha=0.5,
                         label='dark (combined)', density=True, range=rng)
            if all_light is not None and all_light.size > 0:
                plt.hist(all_light, bins=80, alpha=0.5,
                         label='light (combined)', density=True, range=rng)
            plt.legend()
            plt.xlabel('Pixel intensity')
            plt.ylabel('Normalized frequency')
            plt.title(
                'Frequencies of Observed Light Intensities — Overlay (dark vs light)')
            plt.tight_layout()
            overlay_path = FIG_DIR / 'overlay_dark_vs_light.png'  # overlay figure path
            plt.savefig(overlay_path)
            print(f"Saved overlay histogram to {overlay_path}")
            try:
                img = plt.imread(str(overlay_path))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    'Frequencies of Observed Light Intensities — Overlay (dark vs light)')
                plt.show()
            except Exception:
                pass
            plt.close()
    except Exception as e:
        print(f"Failed to create overlay histogram: {e!r}")

    try:
        names = [r['file'] for r in results]  # file names for bar chart
        means = [r['mean_full'] if r['mean_full'] is not None else (
            r['mean_patch'] or 0) for r in results]  # per-file means
        errs = [r['std_full'] if r['std_full'] is not None else (
            r['std_patch'] or 0) for r in results]  # per-file stds
        if names:
            x = np.arange(len(names))  # x positions for bars
            plt.figure(figsize=(10, 4))
            plt.bar(x, means, yerr=errs, capsize=5)
            plt.xticks(x, names, rotation=45, ha='right')
            plt.ylabel('Mean pixel value')
            plt.title(
                'Frequencies of Observed Light Intensities — Per-file means')
            plt.tight_layout()
            bar_path = FIG_DIR / 'per_file_means.png'  # bar chart path
            plt.savefig(bar_path)
            print(f"Saved per-file means bar chart to {bar_path}")
            try:
                img = plt.imread(str(bar_path))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    'Frequencies of Observed Light Intensities — Per-file means')
                plt.show()
            except Exception:
                pass
            plt.close()
    except Exception as e:
        print(f"Failed to create per-file means chart: {e!r}")


if __name__ == '__main__':
    main()
