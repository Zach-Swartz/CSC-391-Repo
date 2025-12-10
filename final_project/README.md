# Artistic Filter System

An advanced image processing system that applies artistic filters with intelligent foreground/background segmentation using GrabCut and Gaussian Mixture Models.

## Setup

### Requirements
```bash
pip install opencv-python numpy
```

### Quick Start
```bash
# List all available filters
python main.py --list-filters

# Apply a single filter
python main.py -i sample_inputs/photo.jpg -o output.png -f vintage

# Chain multiple filters
python main.py -i sample_inputs/photo.jpg -o output.png -f gaussian_blur vignette sharpen

# Interactive mode (test different filters)
python main.py -i sample_inputs/photo.jpg --interactive
```

## Filter Categories

### Full Image Filters
Apply to the entire image:
- `sharpen` - Enhance edges and details
- `vignette` - Darken image corners
- `pixelate` - Create mosaic effect
- `hdr` - High dynamic range enhancement
- `vintage` - Retro photo effect
- `sepia` - Brown-toned vintage look
- `grayscale` - Convert to black and white
- `warm_tone` - Orange/yellow tint
- `cool_tone` - Blue tint
- `posterize` - Reduce color levels
- `solarize` - Invert bright regions
- `motion_blur` - Directional blur effect
- `emboss` - 3D raised effect
- `invert` - Negative colors
- `sketch` - Pencil drawing (grayscale)
- `color_sketch` - Pencil drawing (color)
- `cartoon` - Comic book style
- `crosshatch` - Pen hatching effect
- `edge_preserve` - Smooth while keeping edges
- `detail_enhance` - Amplify fine details

### Background-Only Filters
Apply only to background (preserves subject):
- `gaussian_blur` - Smooth bokeh effect
- `bilateral_blur` - Edge-preserving blur
- `median_filter` - Remove noise while blurring
- `oil_paint` - Thick paint strokes
- `watercolor` - Soft painting effect

### Foreground-Only Filters
Apply only to subject:
- `edges` - Edge detection overlay
- `edge_enhance` - Sharpen subject edges

## Creative Filter Combinations

### Portrait Photography
```bash
# Professional bokeh effect
python main.py -i input.jpg -o output.png -f gaussian_blur edge_enhance sharpen

# Dramatic cinematic look
python main.py -i input.jpg -o output.png -f hdr cool_tone vignette

# Soft dreamy portrait
python main.py -i input.jpg -o output.png -f bilateral_blur warm_tone
```

### Artistic Styles
```bash
# Oil painting effect
python main.py -i input.jpg -o output.png -f oil_paint edge_enhance

# Watercolor painting
python main.py -i input.jpg -o output.png -f watercolor vignette

# Comic book style
python main.py -i input.jpg -o output.png -f cartoon posterize edge_enhance

# Pencil sketch
python main.py -i input.jpg -o output.png -f sketch vignette
```

### Vintage & Retro
```bash
# Classic vintage photo
python main.py -i input.jpg -o output.png -f vintage vignette

# Sepia-toned nostalgia
python main.py -i input.jpg -o output.png -f sepia warm_tone vignette

# Old film look
python main.py -i input.jpg -o output.png -f vintage grayscale vignette
```

### High-Impact Edits
```bash
# Ultra sharp and detailed
python main.py -i input.jpg -o output.png -f sharpen detail_enhance edge_enhance

# Dramatic black and white
python main.py -i input.jpg -o output.png -f grayscale sharpen vignette

# HDR landscape
python main.py -i input.jpg -o output.png -f hdr detail_enhance sharpen

# Motion blur action
python main.py -i input.jpg -o output.png -f motion_blur sharpen
```

### Abstract & Creative
```bash
# Embossed metal effect
python main.py -i input.jpg -o output.png -f emboss grayscale

# Crosshatch drawing
python main.py -i input.jpg -o output.png -f crosshatch vignette

# Solarized poster
python main.py -i input.jpg -o output.png -f solarize posterize

# Pixelated art
python main.py -i input.jpg -o output.png -f pixelate posterize
```
