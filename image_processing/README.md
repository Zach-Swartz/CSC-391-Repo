Image Processing — Exercises 1–3

-------
This  project implements image processing exercises used in Project 1:
- Exercise 1: Intensity transformations (contrast stretching) and histogram equalization. Produces processed images, per-bin counts and normalized histograms, and a composite demo figure.
- Exercise 2: Salt-and-pepper noise, median filtering (3×3), and gradient magnitude comparison (before/after). Saves noisy/denoised images and gradient visualizations and prints a simple quantitative analysis (mean gradient magnitude).
- Exercise 3: Simple Sobel-based edge detection and directional filtering, plus comparison with OpenCV's Canny edge detector. Produces Sobel thresholded edge maps, directional (~45°) maps, and Canny output.

What the program produces
-------------------------
- Exercise 1 outputs (in `image_processing/results/`):
	- `<image>_stretched.png`, `<image>_equalized.png`
	- `<image>_hist_orig.csv`, `<image>_hist_stretched.csv`, `<image>_hist_equalized.csv`
	- `image_processing_contrast_3x3.png` — 3×3 composite showing original/stretched/equalized images, raw counts, and normalized histograms

- Exercise 2 outputs:
	- `<image>_noisy.png`, `<image>_denoised.png`
	- `<image>_grad_noisy.png`, `<image>_grad_denoised.png`
	- `<image>_exercise2_comparison.png`
	- Printed quantitative summary: mean gradient magnitude before/after median filtering

- Exercise 3 outputs:
	- `<image>_sobel_map.png` (Sobel magnitude > threshold)
	- `<image>_dir45_map.png` (directional map for ~45°)
	- `<image>_canny.png` (OpenCV Canny output)
	- `<image>_exercise3_comparison.png` (side-by-side figure)

How to run
----------
1. Make sure you have the required packages installed. The scripts use:
	 - numpy
	 - opencv-python (cv2)
	 - matplotlib

2. From the `image_processing/` directory run:

python .\run_exercises.py

This will:
- find the first image file in `image_processing/images/` 
- run Exercise 1 (contrast stretch + equalization), produces and saves image_processing_contrast_3x3.png (3×3 composite: images, raw counts, normalized histograms)
- run Exercise 2 (noise + median filter), save noisy/denoised/gradient outputs and print mean gradient magnitudes
- run Exercise 3 (Sobel, directional, and Canny), save edge maps and a side-by-side comparison image

Notes and interpretation
------------------------
- Exercise 1: The composite demo image shows raw counts and normalized histograms alongside images so you can inspect how stretching and equalization change intensity distributions.
- Exercise 2: Median filtering reduces spurious high gradient responses introduced by salt-and-pepper noise — the program prints mean gradient magnitude before & after to quantify this.
- Exercise 3: Sobel thresholding returns many magnitude-based responses (thicker clusters), directional filtering isolates edges near a chosen angle (we used ~45°), and Canny typically gives thinner, cleaner contours because it includes smoothing, non-maximum suppression, and hysteresis thresholding.

Additional Notes and Citations:
------------------------
I made use of a low-contrast image I found online, rather than capturing my own due to my, respecftully, horrific photography skills you can observe in project_1. I have cited the image below for reference.

1. “Cold winters morning in Sweden with mist and hoar frost.” iStock, Image ID 1272261194, iStock by Getty Images, https://www.istockphoto.com/photo/cold-winters-morning-in-sweden-with-mist-and-hoar-frost-gm1272261194-3745834962. Accessed 8 Oct. 2025.
