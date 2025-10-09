#!/usr/bin/env python3
# Combined runner for all exercises

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import utilities
from contrast_stretch import contrast_stretch
from equalize_histogram import equalize_histogram
from calculate_histogram import calculate_histogram
from median_filter import median_filter
from calculate_gradient import calculate_gradient
from sobel_edge_detector import sobel_edge_detector
from directional_edge_detector import directional_edge_detector


# Find the first image file inside a directory
# absolute path to first matching image or None
def find_first_image(images_dir):
    # return first match or None
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif',
            '.tiff')  # extensions
    if not os.path.isdir(images_dir):
        return None
    for name in os.listdir(images_dir):
        if name.lower().endswith(exts):
            return os.path.join(images_dir, name)
    return None


# Save histogram counts to a CSV file
def save_hist_csv(counts, out_path):
    # ensure parent dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # write one integer per line
    np.savetxt(out_path, np.asarray(
        counts, dtype=np.int64), fmt='%d', delimiter=',')


def exercise1(images_dir, out_dir):
    # Exercise 1 runner
    print('Exercise 1: Intensity Transformations and Histogram Equalization')
    print('This section applies contrast stretching and histogram equalization to a low-contrast image,')
    print('saves processed images and histogram CSVs, and displays a comparison figure.')

    inp = find_first_image(images_dir)
    if inp is None:
        print('No input image for Exercise 1; skipping')
        return
    im = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)  # input grayscale image
    # percentiles for contrast stretch
    r_min, r_max = np.percentile(im, [2, 98])
    # stretched: contrast-stretched image
    stretched = contrast_stretch(im, float(r_min), float(r_max))
    equalized = equalize_histogram(im)  # histogram-equalized image

    base = os.path.splitext(os.path.basename(inp))[0]
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{base}_stretched.png"), stretched)
    cv2.imwrite(os.path.join(out_dir, f"{base}_equalized.png"), equalized)

    # histogram counts and normalized distributions
    counts_orig, dist_orig = calculate_histogram(im, bins=256)
    counts_str, dist_str = calculate_histogram(stretched, bins=256)
    counts_eq, dist_eq = calculate_histogram(equalized, bins=256)
    save_hist_csv(counts_orig, os.path.join(out_dir, f"{base}_hist_orig.csv"))
    save_hist_csv(counts_str, os.path.join(
        out_dir, f"{base}_hist_stretched.csv"))
    save_hist_csv(counts_eq, os.path.join(
        out_dir, f"{base}_hist_equalized.csv"))

    # Create a composite figure (Can be difficult to visualize if monitor is not large enough)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    imgs = [im, stretched, equalized]
    counts = [counts_orig, counts_str, counts_eq]
    dists = [dist_orig, dist_str, dist_eq]

    bins = np.arange(256)
    titles = ['Original', 'Stretched', 'Equalized']
    for col in range(3):
        # show image
        ax_img = axes[0, col]
        ax_img.imshow(imgs[col], cmap='gray', vmin=0, vmax=255)
        ax_img.set_title(titles[col])
        ax_img.axis('off')
        # annotate pixel count
        total_pixels = int(counts[col].sum())
        ax_img.text(0.02, 0.95, f'pixels: {total_pixels}', color='white', fontsize=10,
                    transform=ax_img.transAxes, va='top', ha='left', bbox=dict(facecolor='black', alpha=0.4))

        # raw counts bar plot
        ax_counts = axes[1, col]
        ax_counts.bar(bins, counts[col], color='tab:gray', width=1)
        ax_counts.set_xlim(0, 255)
        ax_counts.set_ylabel('Counts')
        ax_counts.set_xlabel('Intensity')
        ax_counts.set_title(f'{titles[col]} counts')

        # normalized histogram
        ax_dist = axes[2, col]
        ax_dist.plot(bins, dists[col], color='tab:blue')
        ax_dist.set_xlim(0, 255)
        ax_dist.set_xlabel('Intensity')
        ax_dist.set_ylabel('Probability')
        ax_dist.set_title(f'{titles[col]} (normalized)')

    plt.tight_layout()
    out_fig = os.path.join(out_dir, 'image_processing_contrast_3x3.png')
    fig.savefig(out_fig, dpi=150)
    print('Saved Exercise 1 composite (3x3) to', out_fig)
    plt.show()


# Add salt-and-pepper noise to a grayscale image
# amt = fraction of pixels to corrupt (0..1)
def add_salt_and_pepper(img, amount=0.05):
    out = img.copy()  # noisy output image
    h, w = img.shape
    # number of salt (and number of pepper) pixels to add
    n = int(h * w * amount)
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = 255  # set salt pixels to white
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    out[ys, xs] = 0  # set pepper pixels to black
    return out


def exercise2(images_dir, out_dir):
    # Exercise 2 runner
    print('\nExercise 2: Non-Linear Filtering and Edge Detection')
    print('This section corrupts the image with salt-and-pepper noise, applies a median filter,')
    print('computes gradient magnitudes before and after, and prints a short analysis.')

    inp = find_first_image(images_dir)
    if inp is None:
        print('No input image for Exercise 2; skipping')
        return
    im = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)  # input grayscale image
    # image with salt-and-pepper corruption
    noisy = add_salt_and_pepper(im, amount=0.05)
    denoised = median_filter(noisy, size=3)  # result of median filtering
    # gradient magnitude image for noisy input (calculate_gradient returns mag, angle)
    grad_noisy, _ = calculate_gradient(noisy)
    # gradient magnitude after denoising
    grad_denoised, _ = calculate_gradient(denoised)

    base = os.path.splitext(os.path.basename(inp))[
        0]
    cv2.imwrite(os.path.join(out_dir, f"{base}_noisy.png"), noisy)
    cv2.imwrite(os.path.join(out_dir, f"{base}_denoised.png"), denoised)
    cv2.imwrite(os.path.join(out_dir, f"{base}_grad_noisy.png"), grad_noisy)
    cv2.imwrite(os.path.join(
        out_dir, f"{base}_grad_denoised.png"), grad_denoised)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Noisy')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('Median filtered')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(grad_noisy, cmap='gray')
    axes[1, 0].set_title('Gradient (noisy)')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(grad_denoised, cmap='gray')
    axes[1, 1].set_title('Gradient (denoised)')
    axes[1, 1].axis('off')
    plt.tight_layout()
    out_fig = os.path.join(out_dir, f"{base}_exercise2_comparison.png")
    fig.savefig(out_fig, dpi=150)
    print('Saved Exercise 2 comparison figure to', out_fig)
    plt.show()

    # Compare mean gradient magnitude (simple quantitative metric)
    # scalar mean of gradient magnitudes
    mean_grad_noisy = float(np.mean(grad_noisy))
    # scalar mean after denoising
    mean_grad_denoised = float(np.mean(grad_denoised))
    print('\nExercise 2 quantitative analysis:')
    print(f'  Mean gradient magnitude (noisy): {mean_grad_noisy:.2f}')
    print(f'  Mean gradient magnitude (denoised): {mean_grad_denoised:.2f}')
    print('\nExercise 2 Additional Analysis: Median filtering reduces spurious high-magnitude gradient responses caused by')
    print('salt-and-pepper impulses while preserving the magnitude of true edges, so the mean gradient')
    print('magnitude typically decreases (noise removed) but important edges remain visible in the denoised gradient image.')


def exercise3(images_dir, out_dir):
    # Exercise 3 runner: Sobel magnitude edges, directional edges (~45deg), and Canny
    print('\nExercise 3: Sobel vs Directional vs Canny')
    inp = find_first_image(images_dir)
    if inp is None:
        print('No input image for Exercise 3; skipping')
        return
    im = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)

    # Sobel magnitude edge map
    sobel_map = sobel_edge_detector(im, threshold=50)

    # Directional edges around 45 degrees
    dir_map = directional_edge_detector(im, (35, 55), mag_threshold=30.0)

    # Canny edges using OpenCV
    med = np.median(im)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    canny = cv2.Canny(im, lower, upper)

    base = os.path.splitext(os.path.basename(inp))[0]
    cv2.imwrite(os.path.join(out_dir, f"{base}_sobel_map.png"), sobel_map)
    cv2.imwrite(os.path.join(out_dir, f"{base}_dir45_map.png"), dir_map)
    cv2.imwrite(os.path.join(out_dir, f"{base}_canny.png"), canny)

    # Display side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(sobel_map, cmap='gray')
    axes[0].set_title('Sobel magnitude edges')
    axes[0].axis('off')
    axes[1].imshow(dir_map, cmap='gray')
    axes[1].set_title('Directional ~45deg')
    axes[1].axis('off')
    axes[2].imshow(canny, cmap='gray')
    axes[2].set_title('Canny')
    axes[2].axis('off')
    plt.tight_layout()
    out_fig = os.path.join(out_dir, f"{base}_exercise3_comparison.png")
    fig.savefig(out_fig, dpi=150)
    print('Saved Exercise 3 comparison figure to', out_fig)
    plt.show()

    # Short discussion printed
    print('\nExercise 3 Analysis:')
    print(' - Sobel thresholding gives a clean binary map where gradient magnitude exceeds the threshold.')
    print(' - Directional map isolates edges whose orientation is near 45 degrees; it misses edges at other angles.')
    print(' - Canny combines gradient information and non-maximum suppression and often produces thinner edges and fewer spurious responses.')


def main():
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    exercise1(images_dir, out_dir)
    exercise2(images_dir, out_dir)
    exercise3(images_dir, out_dir)


if __name__ == '__main__':
    main()
