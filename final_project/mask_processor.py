import cv2
import numpy as np


class MaskProcessor:
    # Post-processes segmentation masks using morphological operations (removes noise, fills holes, smooths edges)
    
    def __init__(self):
        # Small elliptical structuring element for fine operations (5x5 pixels)
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Medium elliptical structuring element for moderate operations (9x9 pixels)
        self.medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Large elliptical structuring element for heavy operations (15x15 pixels)
        self.large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    def refine_mask(self, mask, smoothing_level=2):
        # Convert float mask to uint8 format for morphological operations
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype in [np.float32, np.float64] else mask.copy()
        _, binary_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        # Morphological opening (erosion→dilation) removes small noise specks
        noise_removed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.small_kernel)
        # Morphological closing (dilation→erosion) fills small holes in mask
        holes_filled = cv2.morphologyEx(noise_removed, cv2.MORPH_CLOSE, self.medium_kernel)
        edges_smoothed = self.smooth_mask_edges(holes_filled, smoothing_level)
        # Convert back to float32 in range [0, 1] for alpha blending
        refined_float = edges_smoothed.astype(np.float32) / 255.0
        return refined_float
    
    def smooth_mask_edges(self, mask, smoothing_level=2):
        # Apply graduated edge smoothing based on desired softness level
        if smoothing_level == 1:
            smoothed = cv2.GaussianBlur(mask, (5, 5), 0)  # Light smoothing
        elif smoothing_level == 2:
            smoothed = cv2.GaussianBlur(mask, (9, 9), 0)  # Medium smoothing (default)
        elif smoothing_level >= 3:
            smoothed = cv2.GaussianBlur(mask, (15, 15), 0)  # Heavy smoothing
            smoothed = cv2.dilate(smoothed, self.small_kernel, iterations=1)  # Expand mask slightly
        else:
            smoothed = mask  # No smoothing
        return smoothed
    
    def remove_small_components(self, mask, minimum_pixel_area=500):
        # Find connected components using 8-connectivity (includes diagonal neighbors)
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        # Iterate through components (skip 0=background) and keep only those exceeding minimum area
        for label_id in range(1, num_components):
            if stats[label_id, cv2.CC_STAT_AREA] >= minimum_pixel_area:
                cleaned[labels == label_id] = 255
        return cleaned
    
    def erode_mask(self, mask, iterations=1):
        # Shrink mask boundaries inward by specified iterations
        mask_u8 = (mask * 255).astype(np.uint8)
        eroded = cv2.erode(mask_u8, self.small_kernel, iterations=iterations)
        return eroded.astype(np.float32) / 255.0
    
    def dilate_mask(self, mask, iterations=1):
        # Expand mask boundaries outward by specified iterations
        mask_u8 = (mask * 255).astype(np.uint8)
        dilated = cv2.dilate(mask_u8, self.small_kernel, iterations=iterations)
        return dilated.astype(np.float32) / 255.0
    
    def feather_mask_edges(self, mask, feather_pixels=10):
        # Create soft gradient falloff at mask edges for seamless compositing
        mask_u8 = (mask * 255).astype(np.uint8)
        # Calculate odd kernel size based on desired feather distance
        kernel_size = feather_pixels * 2 + 1
        # Apply Gaussian blur to create smooth alpha gradient
        feathered = cv2.GaussianBlur(mask_u8, (kernel_size, kernel_size), 0)
        return feathered.astype(np.float32) / 255.0
    
    def create_mask_visualization(self, mask):
        # Generate color heatmap for mask visualization (red=high, blue=low)
        mask_u8 = (mask * 255).astype(np.uint8)
        # Apply Jet colormap for intuitive visualization of mask values
        heatmap = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
        return heatmap
