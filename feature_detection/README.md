# Part 1

Quick deliverable
- SIFT keypoint visualization: `SIFT_default.jpg` (scale-proportional circles over `images/example-image.jpg`).

What I did
- Created a SIFT detector and printed its main parameters.
- Detected keypoints on `image1` and saved the visualization.

Descriptor note
- SIFT descriptors are 128-D vectors from gradient orientation histograms around each keypoint. They are normalized for robustness. In the notebook you can inspect `descriptors.shape` and `descriptors[0][:16]` to see raw values.

Questions from Part 1
- contrastThreshold ~0.03–0.09 (default 0.04), edgeThreshold ~5–20 (default 10), sigma near 1.6.

Files produced
- `SIFT_default.jpg`

# Part 2

What I compared
- Ran a sweep over (contrastThreshold, edgeThreshold) and saved one image per setting: `sift_contrast{contrast}_edge{edge}.jpg`.

Observed example counts
- default (0.04, 10): 387 keypoints
- (0.03, 5): 266
- (0.02, 5): 290
- (0.03, 15): 570

Short explanation
- Lower contrastThreshold means more weak extrema accepted meaning there are more detections (and more noise).
- Higher edgeThreshold means fewer edge-suppression constraints meaning more edge-like detections.

Files produced
- `sift_contrast{contrast}_edge{edge}.jpg`

# Part 3

What it adds

# Part 3

What it adds
- A plot that shows how many keypoints each parameter set finds.

- Descriptor (simple): a descriptor is a 128-number summary of the patch around a keypoint.
- How to view it: run the notebook to plot the 128-number bar chart or a 4x4 grid of small histograms showing edge directions.
- Why it matters: matching compares these lists; small differences mean a good match across images.

# Part 4

What I did
- Created a transformed image by rotating, scaling, applying an affine warp and a small perspective warp. Saved as `images/example-image-transformed.jpg`.
- Computed SIFT keypoints/descriptors for `image1` and `image2` and matched them with a Brute-Force matcher (`cv.BFMatcher(cv.NORM_L2, crossCheck=True)`).
- Saved a visualization of the top matches as `sift_matches.jpg` and display it inline.

Quick interpretation
- Good matches appear as short connecting lines that align corresponding structures. Long or scattered lines indicate mismatches or areas heavily altered by the transform.

Files produced
- `images/example-image-transformed.jpg`
- `sift_matches.jpg`

# Run this project
1. Install: `pip install numpy opencv-python matplotlib`.
2. Open `feature_detection/repo3code.ipynb` and run cells in order: Part 1 → Part 2 (optional) → Part 3 → Part 4.
