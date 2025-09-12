
# CSC-391-Repo

This repository contains assignments and projects for the Computer Vision course at Wake Forest University. 

Contents

- image_formation/geometric_transforms.py: Reverse-engineers 2D geometric transformations between images using OpenCV.
- image_formation/lens_aperture_params.py: Visualizes thin lens law and f-number relationships for camera lenses.
- image_formation/sampling_quantization.py: Demonstrates sampling and quantization of a continuous signal.
- image_formation/error_noise_analysis.py: Models and analyzes the effect of noise on sampled and quantized signals.

Exercise 1: Geometric Transformations

What the code does:
Loads an original and a transformed image, then applies perspective transformations to the original image using OpenCV to create a similar-looking transformed image. The code demonstrates how to use transformation matrices to map pixel coordinates and visualize the result.

Key concepts:
- Translation, rotation, affine, and perspective transformations
- OpenCV functions: cv2.warpAffine, cv2.warpPerspective, cv2.getPerspectiveTransform


Exercise 2: Thin Lens Law and F-Number Plots

What the code does:
Plots the lens-to-image distance (zi) as a function of object distance (z0) for four different focal lengths, and visualizes aperture diameter (D) as a function of focal length for several real-world lenses and f-numbers.

Key concepts:
- Thin lens law: zi = f z0 / (z0 - f)
- Aperture diameter: D = f / N

Subquestion:

What should the aperture diameter be for each lens in order to achieve their stated maximum f-number?

For each lens, the required aperture diameter is calculated as D = f / N, where f is the focal length and N is the f-number. The code prints the needed aperture diameter for each lens in the console output.


Exercise 3: Sampling and Quantization

What the code does:
Simulates the conversion of a sinusoidal signal into a digital one through sampling and quantization. Plots the original, sampled, and quantized signals, and explores the impact of sampling frequency and quantization levels.

Key concepts:
- Sampling frequency and the Nyquist-Shannon theorem
- Quantization and bit depth

Subquestions:

What do you think a reasonable sampling frequency should be to capture the true shape of the signal?
A reasonable sampling frequency should be at least twice the signal frequency (Nyquist rate) to capture the true shape of the signal.

What should be done to minimize error?
To minimize error, increase both the sampling frequency and the number of quantization bits.

Exercise 4: Noise and Error Analysis

What the code does:
Models real-world noise by adding Gaussian random noise to a sampled signal, then quantizes and plots the noisy signal. Computes and prints error metrics: Mean Square Error (MSE), Root Mean Square Error (RMSE), and Peak Signal-to-Noise Ratio (PSNR).

Key concepts:
- Types of noise: thermal, dark current, flicker, shot noise
- Noise modeling
- Error metrics: MSE, RMSE, PSNR



How to Run

1. Install required Python packages:
	pip install numpy matplotlib opencv-python
2. Run each script from the command line or VS Code terminal:
	python image_formation/geometric_transforms.py
	python image_formation/lens_aperture_params.py
	python image_formation/sampling_quantization.py
	python image_formation/error_noise_analysis.py
