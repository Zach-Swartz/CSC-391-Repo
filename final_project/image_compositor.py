import cv2
import numpy as np


class ImageCompositor:
    # Handles image compositing using Porter-Duff alpha blending for seamless foreground/background composition
    
    def __init__(self):
        pass
    
    def blend_foreground_background(self, foreground, background, mask, feather=True):
        # Convert single-channel mask to 3-channel for RGB blending
        mask_3ch = np.stack([mask] * 3, axis=-1) if len(mask.shape) == 2 else mask
        if feather:
            # Apply Gaussian blur to mask for soft edge transitions
            mask_3ch = self.feather_mask(mask_3ch, blur_kernel_size=5)
        # Porter-Duff "over" compositing: result = fg*alpha + bg*(1-alpha)
        composited = (foreground.astype(np.float32) * mask_3ch +
                      background.astype(np.float32) * (1 - mask_3ch)).astype(np.uint8)
        return composited
    
    def blend_with_alpha(self, first_image, second_image, transparency):
        # Uniformly blend two images with transparency (0=first_image only, 1=second_image only)
        blended = cv2.addWeighted(first_image.astype(np.float32), 1 - transparency,
                                   second_image.astype(np.float32), transparency, 0).astype(np.uint8)
        return blended
    
    def apply_masked_effect(self, original_image, filtered_image, mask):
        # Apply filter effect only to masked regions using Porter-Duff compositing
        return self.blend_foreground_background(filtered_image, original_image, mask)
    
    def invert_mask(self, mask):
        # Flip mask values: 0 becomes 1, 1 becomes 0 (swap foreground/background)
        return 1.0 - mask
    
    def feather_mask(self, mask, blur_kernel_size=5):
        # Create soft gradient at mask edges using Gaussian blur with odd kernel size
        kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
        feathered = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        return feathered
    
    def extract_foreground(self, image, mask):
        # Isolate foreground by multiplying image with 3-channel mask (zeros out background)
        mask_3ch = np.stack([mask] * 3, axis=-1) if len(mask.shape) == 2 else mask
        foreground = (image.astype(np.float32) * mask_3ch).astype(np.uint8)
        return foreground
    
    def extract_background(self, image, mask):
        # Isolate background by multiplying image with inverted mask (background=1, foreground=0)
        bg_mask = self.invert_mask(mask)
        bg_mask_3ch = np.stack([bg_mask] * 3, axis=-1) if len(bg_mask.shape) == 2 else bg_mask
        background = (image.astype(np.float32) * bg_mask_3ch).astype(np.uint8)
        return background
    
    def replace_background(self, image, mask, replacement_background):
        # Replace background while preserving foreground subject
        if isinstance(replacement_background, (tuple, list)):
            # Create solid color background from RGB tuple
            bg_img = np.full_like(image, replacement_background, dtype=np.uint8)
        else:
            # Use provided image as background, resize if dimensions don't match
            bg_img = cv2.resize(replacement_background, (image.shape[1], image.shape[0])) if replacement_background.shape[:2] != image.shape[:2] else replacement_background
        # Composite original foreground over new background
        composited = self.blend_foreground_background(image, bg_img, mask)
        return composited
    
    def create_vignette_effect(self, image, darkness_intensity=0.5):
        # Create vignette by darkening image edges while keeping center bright
        rows, cols = image.shape[:2]
        # Generate 1D Gaussian kernels for width and height
        gauss_x = cv2.getGaussianKernel(cols, cols / 2)
        gauss_y = cv2.getGaussianKernel(rows, rows / 2)
        # Create 2D radial gradient by outer product of 1D kernels
        radial_grad = gauss_y * gauss_x.T
        # Normalize gradient to range [0, 1] with center=1, edges=0
        normalized = radial_grad / radial_grad.max()
        # Scale gradient: darkness_intensity controls minimum brightness at edges
        vignette_mask = normalized * (1 - darkness_intensity) + darkness_intensity
        # Convert to 3-channel mask for RGB multiplication
        vignette_3ch = np.stack([vignette_mask] * 3, axis=-1)
        # Multiply image by vignette mask to darken edges
        vignetted = (image.astype(np.float32) * vignette_3ch).astype(np.uint8)
        return vignetted
    
    def overlay_edges(self, image, edge_map, line_color=(0, 0, 0), edge_opacity=0.7):
        # Draw edges on top of image with specified color and opacity
        # Create blank layer to hold edge lines
        edge_layer = np.zeros_like(image)
        # Set edge pixels to specified color (default black)
        edge_layer[edge_map > 0] = line_color
        # Blend edge layer with original image using opacity weight
        result = cv2.addWeighted(image, 1.0, edge_layer, edge_opacity, 0)
        return result
