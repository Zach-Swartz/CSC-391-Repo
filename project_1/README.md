Project 1 — Image Formation (CSC-391)

Overview
Note that this file is AI generated, since this is technically not required. If the below does not provide enough information on how to set up the program, then let me know and I'll help out.

This folder contains three parts for Project 1: noise analysis (Part 1), camera comparison (Part 2), and a real-time filtering demo with benchmarking (Part 3). Each part includes scripts, input images, and generated results (CSVs and PNGs).

Folder layout

- part_1/
  - analyze_noise.py — analyze center-patch statistics and save histograms
  - images/ — input DNG files used for Part 1
  - results/summary.csv — per-file means/stds and other stats
  - results/figs/ — PNG histograms and combined figures

- part_2/
  - compare_cameras.py — compute patch stats for main vs tele images and save annotated figures
  - images/ — input DNG files (main_camera.dng, telephoto_camera.dng)
  - results/camera_compare_summary.csv — CSV with patch_mean, patch_std, snr, shutter, aperture, iso, rel_exposure
  - results/figs/ — PNGs with patch crops, annotated histograms, and combined histogram

- part_3/
  - real_time_filters.py — real-time OpenCV demo and optional synthetic benchmark mode (if present)
  - Use `--proc-width` to control processing resolution; `--manual` uses the Python convolution; `--bench` runs a synthetic CPU benchmark if available.

How to run (PowerShell on Windows)

1) Part 1 — noise analysis

Open PowerShell, change to the part_1 directory, then run:

python .\analyze_noise.py

Output:
- `project_1/part_1/results/summary.csv` — per-file center-patch mean/std and full-image stats
- PNGs in `project_1/part_1/results/figs/` — per-file histograms and combined figures

2) Part 2 — camera comparison

Open PowerShell, change to the part_2 directory, then run:

python .\compare_cameras.py

Output:
- `project_1/part_2/results/camera_compare_summary.csv` — patch means/stds, SNR proxy, shutter/aperture/ISO, rel_exposure
- PNGs in `project_1/part_2/results/figs/` — patch crops and annotated histograms

3) Part 3 — real-time filtering demo

Open PowerShell, change to the part_3 directory, then run:

# interactive demo (default)
python .\real_time_filters.py

# downscale processing width for speed
python .\real_time_filters.py --proc-width 320

# force manual (Python) convolution (slow)
python .\real_time_filters.py --manual --proc-width 320

# synthetic benchmark mode (if the script supports it)
python .\real_time_filters.py --bench --proc-width 320

Output and where to find calculated figures

- Part 1 results: `project_1/part_1/results/summary.csv` and figure PNGs in `project_1/part_1/results/figs/`.
- Part 2 results: `project_1/part_2/results/camera_compare_summary.csv` and PNGs in `project_1/part_2/results/figs/`.
- Part 3 outputs: bench prints timing summaries to the terminal; the interactive demo displays live windows and does not save images by default.

Notes

- If `--bench` is not present in `real_time_filters.py`, the interactive demo still works. The bench mode was an optional convenience to gather CPU timings without a webcam.
- Keep the results CSVs and PNGs when preparing your submission; they are the evidence files graders expect.
