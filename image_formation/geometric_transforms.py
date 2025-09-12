import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images():
    "Load original and transformed images from images/ folder."
    img_dir = os.path.join(os.path.dirname(__file__), "images")
    original_path = os.path.join(img_dir, "original_image.jpg")
    transformed_path = os.path.join(img_dir, "transformed_image.jpg")

    original = cv2.imread(original_path)
    transformed = cv2.imread(transformed_path)

    if original is None or transformed is None:
        raise FileNotFoundError(
            "Check images folder for image names")

    return original, transformed


def apply_perspective_transform(img):
    rows, cols = img.shape[:2]

    # indicate 4 corners
    pts1 = np.float32([
        [0, 0],
        [cols - 1, 0],
        [0, rows - 1],
        [cols - 1, rows - 1]
    ])

    # point coordinate maps
    pts2 = np.float32([
        [50, 10],          # top left corner
        [cols - 50, 75],  # top right corner
        [120, rows],   # top left corner
        [cols + 10, rows + 75]  # bottom-right corner
    ])

    # perspective transform
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # warp perspective
    dst = cv2.warpPerspective(img, M, (cols, rows))

    return dst


def display_results(original, transformed, reverse_engineered):
    "Display the three images side by side in RGB for comparison."
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    trans_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
    rev_rgb = cv2.cvtColor(reverse_engineered, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(orig_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(trans_rgb)
    plt.title("Provided Transformed Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(rev_rgb)
    plt.title("New Transformed Image")
    plt.axis("off")

    plt.show()


def main():
    original, transformed = load_images()
    reverse_engineered = apply_perspective_transform(original)

    display_results(original, transformed, reverse_engineered)


if __name__ == "__main__":
    main()
