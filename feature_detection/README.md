Feature Detection — Part 1 (SIFT)

-------
This mini-project detects blobs using SIFT (Difference of Gaussians) in OpenCV and saves a visualization with circle sizes that reflect scale.

What this does
--------------
- Creates a SIFT detector with `cv2.SIFT_create()`.
- Prints key tunable parameters and typical ranges.
- Loads `images/example-image.jpg`, detects keypoints, and draws them with `cv2.drawKeypoints(..., DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)`.
- Saves the result as `sift_keypoints.jpg`.
- Prints the total number of keypoints and example fields (position, size, angle, response, octave).

What it produces
----------------
- `sift_keypoints.jpg` — image with SIFT keypoints drawn at their detected scales.
- Console output — SIFT parameters, total keypoints, and one sample keypoint’s attributes.

How to run
----------
1. Open `feature_detection/repo3code.ipynb` in VS Code and run all cells.
2. Requirements: `numpy`, `opencv-python`.

SIFT parameters (defaults and typical ranges)
--------------------------------------------
- contrastThreshold: default 0.04 (typical 0.03–0.09)
- edgeThreshold: default 10 (typical 5–20)
- sigma: default 1.6 (initial blur)
- nfeatures: default 0 (unlimited)

Observations
------------
- Large blobs get large circles; small blobs get small circles (scale-invariant behavior).
- Some features are missed or placed on textured backgrounds; adjust `contrastThreshold` and `edgeThreshold` to trade off recall vs. false positives.
- Multiple orientations at the same location are expected for strong features.
---------------------------