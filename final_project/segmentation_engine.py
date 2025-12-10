import cv2
import numpy as np


class SegmentationEngine:
    # Engine for extracting subject masks using GrabCut segmentation with Gaussian Mixture Models (GMM)
    
    def __init__(self, model_selection=1):
        # Margin ratio for initial bounding rectangle (10% on each side)
        self.rectangle_margin_ratio = 0.1
    
    def extract_subject_mask(self, image):
        # Initialize empty mask for GrabCut algorithm
        initial_mask = np.zeros(image.shape[:2], np.uint8)
        # GMM models for background and foreground (5 Gaussian components with 13 parameters each)
        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        
        # Calculate dimensions for initial bounding rectangle
        height, width = image.shape[:2]
        margin_h = int(height * self.rectangle_margin_ratio)
        margin_w = int(width * self.rectangle_margin_ratio)
        # Create bounding rectangle: (x, y, width, height) with margins
        bounding_rect = (margin_w, margin_h, width - 2*margin_w, height - 2*margin_h)
        
        try:
            # Run GrabCut algorithm with 5 iterations using Markov Random Fields for segmentation refinement
            cv2.grabCut(image, initial_mask, bounding_rect, background_model, foreground_model, 5, cv2.GC_INIT_WITH_RECT)
            # Convert GrabCut mask to binary: probable/definite BG (0,2)→0, probable/definite FG (1,3)→1
            subject_mask = np.where((initial_mask == 2) | (initial_mask == 0), 0, 1).astype('float32')
        except:
            # Fallback to elliptical mask if GrabCut fails
            subject_mask = self._create_fallback_center_mask(image.shape[:2])
        
        return subject_mask
    
    def _create_fallback_center_mask(self, shape):
        # Create elliptical mask centered in image as fallback when GrabCut fails
        height, width = shape
        mask = np.zeros((height, width), dtype=np.float32)
        center_x, center_y = width // 2, height // 2
        radius_x, radius_y = int(width * 0.35), int(height * 0.35)
        y_coords, x_coords = np.ogrid[:height, :width]
        # Standard ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
        ellipse_region = ((x_coords - center_x) / radius_x) ** 2 + ((y_coords - center_y) / radius_y) ** 2 <= 1
        mask[ellipse_region] = 1.0
        # Apply heavy Gaussian blur to create smooth gradient edges
        smooth_mask = cv2.GaussianBlur(mask, (51, 51), 0)
        return smooth_mask
    
    def extract_multi_class_mask(self, image, threshold=0.5):
        # Convert soft float mask to hard binary mask using threshold
        float_mask = self.extract_subject_mask(image)
        # Apply threshold and scale to 0-255 range for visualization
        binary_mask = (float_mask > threshold).astype(np.uint8) * 255
        return binary_mask
    
    def visualize_mask(self, image, mask, transparency=0.7):
        # Create visualization overlay with green tint on masked regions
        green_overlay = image.copy()
        mask_3ch = np.stack([mask] * 3, axis=-1)
        green_overlay[:, :, 0] = green_overlay[:, :, 0] * 0.5  # Reduce blue by 50%
        green_overlay[:, :, 2] = green_overlay[:, :, 2] * 0.5  # Reduce red by 50% (creates green tint)
        # Blend original image with green overlay using mask-weighted transparency
        blended_viz = cv2.addWeighted(image.astype(np.float32), 1 - transparency * mask_3ch,
                                       green_overlay.astype(np.float32), transparency * mask_3ch, 0).astype(np.uint8)
        return blended_viz
