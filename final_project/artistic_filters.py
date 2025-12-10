import cv2
import numpy as np


class ArtisticFilters:
    
    def __init__(self):
        pass
    
    def apply_gaussian_blur(self, image, kernel_size=15):
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def apply_bilateral_blur(self, image, neighborhood_diameter=15, color_sigma=80, space_sigma=80):
        edge_preserved = cv2.bilateralFilter(image, neighborhood_diameter, color_sigma, space_sigma)
        return edge_preserved
    
    def apply_median_filter(self, image, kernel_size=9):
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        noise_removed = cv2.medianBlur(image, kernel_size)
        return noise_removed
    
    def extract_edges(self, image, mask=None, low_threshold=50, high_threshold=150):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            edges = cv2.bitwise_and(edges, edges, mask=mask_uint8)
        
        return edges
    
    def extract_sobel_edges(self, image, mask=None):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        horizontal_gradient = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
        vertical_gradient = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(horizontal_gradient**2 + vertical_gradient**2)
        edge_magnitude = np.clip(edge_magnitude, 0, 255).astype(np.uint8)
        
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            edge_magnitude = cv2.bitwise_and(edge_magnitude, edge_magnitude, mask=mask_uint8)
        
        return edge_magnitude
    
    def create_sketch_effect(self, image, edges):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        edges_inverted = cv2.bitwise_not(edges)
        sketch_gray = cv2.multiply(gray.astype(np.float32), edges_inverted.astype(np.float32) / 255.0)
        sketch_gray = np.clip(sketch_gray, 0, 255).astype(np.uint8)
        sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
        return sketch_bgr
    
    def apply_oil_paint_effect(self, image, size=7, dynRatio=1):
        try:
            oil_painted = cv2.xphoto.oilPainting(image, size, dynRatio)
        except AttributeError:
            oil_painted = cv2.stylization(image, sigma_s=60, sigma_r=0.6)
        return oil_painted
    
    def apply_watercolor_effect(self, image):
        watercolor = image.copy()
        for iteration in range(2):
            watercolor = cv2.bilateralFilter(watercolor, 9, 75, 75)
        watercolor = cv2.medianBlur(watercolor, 5)
        return watercolor
    
    def enhance_foreground_edges(self, image, mask):
        edges = self.extract_edges(image, mask=mask)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.addWeighted(image, 1.0, edges_3ch, 0.3, 0)
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (image * (1 - mask_3ch) + sharpened * mask_3ch).astype(np.uint8)
        return result
    
    def apply_cartoon_effect(self, image):
        color = cv2.bilateralFilter(image, 9, 250, 250)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_3ch)
        return cartoon
    
    def apply_sepia_tone(self, image):
        sepia_matrix = np.array([[0.272, 0.534, 0.131],
                                  [0.349, 0.686, 0.168],
                                  [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, sepia_matrix)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia
    
    def sharpen_image(self, image, strength=1.0):
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    
    def apply_vignette(self, image, intensity=0.6):
        rows, cols = image.shape[:2]
        gaussian_x = cv2.getGaussianKernel(cols, cols / 2)
        gaussian_y = cv2.getGaussianKernel(rows, rows / 2)
        gaussian_mask = gaussian_y * gaussian_x.T
        gaussian_mask = gaussian_mask / gaussian_mask.max()
        vignette_mask = gaussian_mask * (1 - intensity) + intensity
        vignette_mask_3ch = np.stack([vignette_mask] * 3, axis=-1)
        vignetted = (image.astype(np.float32) * vignette_mask_3ch).astype(np.uint8)
        return vignetted
    
    def apply_pixelation(self, image, pixel_size=10):
        height, width = image.shape[:2]
        downscaled = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_NEAREST)
        return pixelated
    
    def apply_emboss(self, image):
        emboss_kernel = np.array([[-2, -1, 0],
                                   [-1,  1, 1],
                                   [ 0,  1, 2]])
        embossed = cv2.filter2D(image, -1, emboss_kernel)
        embossed = cv2.cvtColor(embossed, cv2.COLOR_BGR2GRAY)
        embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
        return embossed
        
    
    def apply_hdr_effect(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Adaptive histogram equalization
        lightness = clahe.apply(lightness)
        lab = cv2.merge([lightness, a_channel, b_channel])
        hdr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Boost saturation
        hdr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return hdr
    
    def apply_vintage_effect(self, image):
        vintage = self.apply_sepia_tone(image)
        noise = np.random.normal(0, 15, vintage.shape).astype(np.int16)
        vintage = np.clip(vintage.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        vintage = self.apply_vignette(vintage, intensity=0.5)
        vintage = cv2.addWeighted(vintage, 0.8, np.full_like(vintage, 128), 0.2, 0)
        return vintage
    
    def apply_cool_tone(self, image):
        cool = image.copy().astype(np.float32)
        cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.15, 0, 255)
        cool[:, :, 2] = np.clip(cool[:, :, 2] * 0.85, 0, 255)
        return cool.astype(np.uint8)
    
    def apply_warm_tone(self, image):
        warm = image.copy().astype(np.float32)
        warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.85, 0, 255)
        warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.05, 0, 255)
        warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.15, 0, 255)
        return warm.astype(np.uint8)
    
    def apply_pencil_drawing(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blurred = cv2.bitwise_not(blurred)
        pencil = cv2.divide(gray, inverted_blurred, scale=256.0)  # Dodge blend mode for sketch effect
        pencil_bgr = cv2.cvtColor(pencil, cv2.COLOR_GRAY2BGR)
        return pencil_bgr
    
    def apply_color_pencil(self, image):
        sketch_gray, sketch_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return sketch_color
    
    def apply_edge_preserve_filter(self, image):
        filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
        return filtered
    
    def apply_detail_enhance(self, image):
        enhanced = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
        return enhanced
    
    def apply_grayscale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray_bgr
    
    def apply_invert(self, image):
        inverted = cv2.bitwise_not(image)
        return inverted
    
    def apply_posterize(self, image, levels=4):
        posterized = image.copy()
        color_step = 256 // levels
        posterized = (posterized // color_step) * color_step
        return posterized
    
    def apply_solarize(self, image, threshold=128):
        solarized = image.copy()
        bright_pixels = solarized > threshold
        solarized[bright_pixels] = 255 - solarized[bright_pixels]
        return solarized
    
    def apply_motion_blur(self, image, kernel_size=15):
        motion_kernel = np.zeros((kernel_size, kernel_size))
        center_row = int((kernel_size - 1) / 2)
        motion_kernel[center_row, :] = np.ones(kernel_size)  # Horizontal motion blur line
        motion_kernel = motion_kernel / kernel_size
        blurred = cv2.filter2D(image, -1, motion_kernel)
        return blurred
    
    def apply_crosshatch(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        diagonal_kernel_1 = np.array([[-1, -1,  2],
                                       [-1,  2, -1],
                                       [ 2, -1, -1]])
        diagonal_kernel_2 = np.array([[ 2, -1, -1],
                                       [-1,  2, -1],
                                       [-1, -1,  2]])
        hatch_layer_1 = cv2.filter2D(gray, -1, diagonal_kernel_1)
        hatch_layer_2 = cv2.filter2D(gray, -1, diagonal_kernel_2)
        crosshatch = cv2.bitwise_and(hatch_layer_1, hatch_layer_2)
        crosshatch_bgr = cv2.cvtColor(crosshatch, cv2.COLOR_GRAY2BGR)
        return crosshatch_bgr
