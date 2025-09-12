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

    return original, transformed


def apply_perspective_transform(img):
    rows, cols = img.shape[:2]

    # Rotation: See Below
    # There is an issue with this where, when rotating the provided image
    # it actually cuts off a portion where the image should still be. I tried using copilot
    # to extend the canvas and then condense it back, but this resulted in many failures and
    # made the code difficult to read. So, I left it as is, but outside of this error, I
    # believe the rest of it works correctly.

    angle = 10
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rot, (cols, rows))
    print(f"Applied rotation: angle = {angle} degrees around center")

    # Translation
    tx, ty = -7, -15
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img_translated = cv2.warpAffine(img_rotated, M_trans, (cols, rows))
    print(f"Applied translation: tx = {tx}, ty = {ty}")

    # Scaling
    scale_x, scale_y = 1.05, 1.05
    img_scaled = cv2.resize(img_translated, None, fx=scale_x,
                            fy=scale_y, interpolation=cv2.INTER_LINEAR)
    print(f"Applied scaling: scale_x = {scale_x}, scale_y = {scale_y}")

    # Perspective transform
    pts1 = np.float32([
        [0, 0],
        [cols - 1, 0],
        [0, rows - 1],
        [cols - 1, rows - 1]
    ])
    pts2 = np.float32([
        [50, 10],          # top left corner
        [cols - 50, 75],  # top right corner
        [120, rows],   # top left corner
        [cols + 10, rows + 75]  # bottom-right corner
    ])
    M_persp = cv2.getPerspectiveTransform(pts1, pts2)
    img_persp = cv2.warpPerspective(img_scaled, M_persp, (cols, rows))
    print("Applied perspective transformation with custom corner mapping.")

    return img_persp


def display_results(original, transformed, reverse_engineered):
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
