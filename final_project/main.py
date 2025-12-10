import cv2
import numpy as np
import argparse
import os
from pathlib import Path

from segmentation_engine import SegmentationEngine
from mask_processor import MaskProcessor
from artistic_filters import ArtisticFilters
from image_compositor import ImageCompositor


AVAILABLE_FILTERS = {
    'gaussian_blur': {'desc': 'Gaussian blur', 'target': 'bg', 'func': 'apply_gaussian_blur'},
    'bilateral_blur': {'desc': 'Bilateral blur (edge-preserving)', 'target': 'bg', 'func': 'apply_bilateral_blur'},
    'median_filter': {'desc': 'Median filter', 'target': 'bg', 'func': 'apply_median_filter'},
    'oil_paint': {'desc': 'Oil painting effect', 'target': 'bg', 'func': 'apply_oil_paint_effect'},
    'watercolor': {'desc': 'Watercolor painting', 'target': 'bg', 'func': 'apply_watercolor_effect'},
    'sketch': {'desc': 'Pencil sketch (grayscale)', 'target': 'full', 'func': 'apply_pencil_drawing'},
    'color_sketch': {'desc': 'Color pencil sketch', 'target': 'full', 'func': 'apply_color_pencil'},
    'cartoon': {'desc': 'Cartoon/comic effect', 'target': 'full', 'func': 'apply_cartoon_effect'},
    'edges': {'desc': 'Edge detection', 'target': 'fg', 'func': 'extract_edges'},
    'edge_enhance': {'desc': 'Edge enhancement', 'target': 'fg', 'func': 'enhance_foreground_edges'},
    'sharpen': {'desc': 'Sharpen image', 'target': 'full', 'func': 'sharpen_image'},
    'vignette': {'desc': 'Vignette (darkened corners)', 'target': 'full', 'func': 'apply_vignette'},
    'pixelate': {'desc': 'Pixelation/mosaic', 'target': 'full', 'func': 'apply_pixelation'},
    'emboss': {'desc': 'Emboss effect', 'target': 'full', 'func': 'apply_emboss'},
    'hdr': {'desc': 'HDR effect', 'target': 'full', 'func': 'apply_hdr_effect'},
    'vintage': {'desc': 'Vintage/retro effect', 'target': 'full', 'func': 'apply_vintage_effect'},
    'cool_tone': {'desc': 'Cool color tone (blue tint)', 'target': 'full', 'func': 'apply_cool_tone'},
    'warm_tone': {'desc': 'Warm color tone (orange tint)', 'target': 'full', 'func': 'apply_warm_tone'},
    'sepia': {'desc': 'Sepia tone', 'target': 'full', 'func': 'apply_sepia_tone'},
    'grayscale': {'desc': 'Grayscale conversion', 'target': 'full', 'func': 'apply_grayscale'},
    'invert': {'desc': 'Invert colors', 'target': 'full', 'func': 'apply_invert'},
    'posterize': {'desc': 'Posterize (reduce colors)', 'target': 'full', 'func': 'apply_posterize'},
    'solarize': {'desc': 'Solarization effect', 'target': 'full', 'func': 'apply_solarize'},
    'motion_blur': {'desc': 'Motion blur', 'target': 'full', 'func': 'apply_motion_blur'},
    'crosshatch': {'desc': 'Crosshatch drawing', 'target': 'full', 'func': 'apply_crosshatch'},
    'edge_preserve': {'desc': 'Edge-preserving smoothing', 'target': 'full', 'func': 'apply_edge_preserve_filter'},
    'detail_enhance': {'desc': 'Detail enhancement', 'target': 'full', 'func': 'apply_detail_enhance'},
}


class ArtisticFilterSystem:
    
    def __init__(self):
        self.segmentation = SegmentationEngine()
        self.mask_processor = MaskProcessor()
        self.filters = ArtisticFilters()
        self.compositor = ImageCompositor()
        self.setup_directories()
    
    def setup_directories(self):
        Path("final_project/sample_inputs").mkdir(parents=True, exist_ok=True)
        Path("final_project/filtered_outputs").mkdir(parents=True, exist_ok=True)
        Path("final_project/mask_visualizations").mkdir(parents=True, exist_ok=True)
    
    def process_image(self, image, mode=1, save_intermediate=False):
        mask = self.segmentation.extract_subject_mask(image)
        refined_mask = self.mask_processor.refine_mask(mask)
        
        if save_intermediate:
            mask_vis = cv2.applyColorMap((refined_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite("final_project/mask_visualizations/subject_mask.png", mask_vis)
        
        processed_image = self.apply_mode(image, refined_mask, mode)
        return processed_image
    
    def apply_mode(self, image, mask, mode):
        if mode == 1:
            return self.mode_blurred_background(image, mask)
        elif mode == 2:
            return self.mode_sketch_foreground(image, mask)
        elif mode == 3:
            return self.mode_fully_artistic(image, mask)
        elif mode == 4:
            return self.mode_inverse_stylization(image, mask)
        else:
            print(f"Unknown mode {mode}, defaulting to mode 1")
            return self.mode_blurred_background(image, mask)
    
    def mode_blurred_background(self, image, mask):
        blurred_bg = self.filters.apply_bilateral_blur(image, neighborhood_diameter=15, 
                                                        color_sigma=80, space_sigma=80)
        result = self.compositor.blend_foreground_background(foreground=image, 
                                                              background=blurred_bg, mask=mask)
        return result
    
    def mode_sketch_foreground(self, image, mask):
        edges = self.filters.extract_edges(image, mask)
        sketch = self.filters.create_sketch_effect(image, edges)
        blurred_bg = self.filters.apply_gaussian_blur(image, kernel_size=15)
        result = self.compositor.blend_foreground_background(foreground=sketch, 
                                                              background=blurred_bg, mask=mask)
        return result
    
    def mode_fully_artistic(self, image, mask):
        artistic_bg = self.filters.apply_oil_paint_effect(image)
        edge_enhanced_fg = self.filters.enhance_foreground_edges(image, mask)
        result = self.compositor.blend_foreground_background(foreground=edge_enhanced_fg, 
                                                              background=artistic_bg, mask=mask)
        return result
    
    def mode_inverse_stylization(self, image, mask):
        stylized_fg = self.filters.apply_median_filter(image, kernel_size=9)
        stylized_fg = self.filters.enhance_foreground_edges(stylized_fg, mask)
        result = self.compositor.blend_foreground_background(foreground=stylized_fg, 
                                                              background=image, mask=mask)
        return result
    
    def apply_custom_filters(self, image, mask, filter_names):
        result = image.copy()
        
        for filter_name in filter_names:
            if filter_name not in AVAILABLE_FILTERS:
                print(f"Warning: Unknown filter '{filter_name}', skipping...")
                continue
            
            filter_info = AVAILABLE_FILTERS[filter_name]
            func_name = filter_info['func']
            target = filter_info['target']
            filter_func = getattr(self.filters, func_name)
            
            if target == 'full':  # Apply to entire image
                result = filter_func(result)
            elif target == 'bg':  # Apply only to background region
                filtered_bg = filter_func(result)
                result = self.compositor.blend_foreground_background(foreground=result, 
                                                                      background=filtered_bg, mask=mask)
            elif target == 'fg':  # Apply only to foreground region
                if func_name == 'extract_edges':
                    edges = filter_func(result, mask=mask)
                    sketch = self.filters.create_sketch_effect(result, edges)
                    result = self.compositor.blend_foreground_background(foreground=sketch, 
                                                                          background=result, mask=mask)
                elif func_name == 'enhance_foreground_edges':
                    result = filter_func(result, mask)
                else:
                    filtered_fg = filter_func(result)
                    result = self.compositor.blend_foreground_background(foreground=filtered_fg, 
                                                                          background=result, mask=mask)
        
        return result
    
    def process_static_image(self, input_path, output_path, mode=1, custom_filters=None):
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image from {input_path}")
            return
        
        print(f"Processing image: {input_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        mask = self.segmentation.extract_subject_mask(image)
        refined_mask = self.mask_processor.refine_mask(mask)
        
        if custom_filters:
            print(f"Applying custom filters: {', '.join(custom_filters)}")
            result = self.apply_custom_filters(image, refined_mask, custom_filters)
        else:
            result = self.process_image(image, mode=mode, save_intermediate=True)
        
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {output_path}")
        self.display_comparison(image, result)
    
    def display_comparison(self, original, processed):
        max_height = 800
        if original.shape[0] > max_height:
            scale = max_height / original.shape[0]
            new_width = int(original.shape[1] * scale)
            original_display = cv2.resize(original, (new_width, max_height))
            processed_display = cv2.resize(processed, (new_width, max_height))
        else:
            original_display = original.copy()
            processed_display = processed.copy()
        
        comparison = np.hstack([original_display, processed_display])
        cv2.imshow("Comparison: Original (left) | Processed (right)", comparison)
        print("Press any key to close the comparison window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def interactive_mode(self, input_path):
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image from {input_path}")
            return
        
        print(f"\n{'='*70}")
        print(f"Interactive Filter Mode")
        print(f"{'='*70}")
        print(f"Image: {input_path}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
        
        print("\nExtracting subject mask...")
        mask = self.segmentation.extract_subject_mask(image)
        refined_mask = self.mask_processor.refine_mask(mask)
        print("Mask extracted successfully!")
        
        while True:
            print(f"\n{'-'*70}")
            print("Available Filters:")
            print(f"{'-'*70}")
            
            print("\n📸 FULL IMAGE FILTERS:")
            for name, info in sorted(AVAILABLE_FILTERS.items()):
                if info['target'] == 'full':
                    print(f"  - {name:20s} : {info['desc']}")
            
            print("\n🎨 BACKGROUND FILTERS:")
            for name, info in sorted(AVAILABLE_FILTERS.items()):
                if info['target'] == 'bg':
                    print(f"  - {name:20s} : {info['desc']}")
            
            print("\n✨ FOREGROUND FILTERS:")
            for name, info in sorted(AVAILABLE_FILTERS.items()):
                if info['target'] == 'fg':
                    print(f"  - {name:20s} : {info['desc']}")
            
            print(f"\n{'-'*70}")
            print("Commands:")
            print("  - Type filter names separated by spaces to apply multiple filters")
            print("  - Type 'list' to see filters again")
            print("  - Type 'quit' or 'exit' to quit")
            print(f"{'-'*70}\n")
            
            user_input = input("Enter filter(s) to apply: ").strip().lower()
            
            if user_input in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            if user_input == 'list':
                continue
            
            if not user_input:
                print("No filters specified. Please try again.")
                continue
            
            filter_names = user_input.split()
            valid_filters = []
            for fname in filter_names:
                if fname in AVAILABLE_FILTERS:
                    valid_filters.append(fname)
                else:
                    print(f"Warning: Unknown filter '{fname}', skipping...")
            
            if not valid_filters:
                print("No valid filters specified. Please try again.")
                continue
            
            print(f"\nApplying filters: {', '.join(valid_filters)}")
            result = self.apply_custom_filters(image, refined_mask, valid_filters)
            
            filter_str = '_'.join(valid_filters)
            output_path = f"final_project/filtered_outputs/custom_{filter_str}.png"
            
            cv2.imwrite(output_path, result)
            print(f"✓ Saved result to: {output_path}")
            self.display_comparison(image, result)
            
            continue_choice = input("\nApply more filters? (yes/no): ").strip().lower()
            if continue_choice not in ['yes', 'y']:
                print("Exiting interactive mode...")
                break


def print_filter_list():
    print("\n" + "="*70)
    print("AVAILABLE FILTERS")
    print("="*70)
    
    print("\n📸 FULL IMAGE FILTERS:")
    for name, info in sorted(AVAILABLE_FILTERS.items()):
        if info['target'] == 'full':
            print(f"  {name:20s} - {info['desc']}")
    
    print("\n🎨 BACKGROUND FILTERS:")
    for name, info in sorted(AVAILABLE_FILTERS.items()):
        if info['target'] == 'bg':
            print(f"  {name:20s} - {info['desc']}")
    
    print("\n✨ FOREGROUND FILTERS:")
    for name, info in sorted(AVAILABLE_FILTERS.items()):
        if info['target'] == 'fg':
            print(f"  {name:20s} - {info['desc']}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Augmented Artistic Image Filtering System")
    parser.add_argument("--input", "-i", type=str, help="Input image path (for static image mode)")
    parser.add_argument("--output", "-o", type=str, default="final_project/filtered_outputs/artistic_output.png",
                       help="Output image path")
    parser.add_argument("--mode", "-m", type=int, default=1, choices=[1, 2, 3, 4],
                       help="Filter mode: 1=Blurred BG, 2=Sketch FG, 3=Fully Artistic, 4=Inverse")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode - select filters from menu")
    parser.add_argument("--filters", "-f", type=str, nargs='+',
                       help="Apply custom filters (space-separated list)")
    parser.add_argument("--list-filters", action="store_true",
                       help="List all available filters")
    
    args = parser.parse_args()
    
    # List filters and exit
    if args.list_filters:
        print_filter_list()
        return
    
    # Initialize system
    system = ArtisticFilterSystem()
    
    if args.interactive:
        if not args.input:
            print("Error: Interactive mode requires --input argument")
            print("Example: python main.py -i sample_inputs/photo.jpg --interactive")
            return
        system.interactive_mode(args.input)
    elif args.input:
        # Static image mode with optional custom filters
        if args.filters:
            # Custom filters specified
            system.process_static_image(args.input, args.output, custom_filters=args.filters)
        else:
            # Use predefined mode
            system.process_static_image(args.input, args.output, mode=args.mode)
    else:
        print("Error: Please specify a processing mode")
        print("\nExamples:")
        print("  Predefined mode:   python main.py -i sample_inputs/photo.jpg -m 1")
        print("  Custom filters:    python main.py -i sample_inputs/photo.jpg -f gaussian_blur vignette")
        print("  Interactive mode:  python main.py -i sample_inputs/photo.jpg --interactive")
        print("  List filters:      python main.py --list-filters")


if __name__ == "__main__":
    main()
