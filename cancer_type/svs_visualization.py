"""
SVS Visualization Module

This module provides functions for visualizing SVS (Aperio Slide Virtual Slide) files,
which are commonly used for storing whole slide images in digital pathology.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import openslide
from openslide import OpenSlide
from PIL import Image
import cv2
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

def load_svs_slide(slide_path):
    """
    Load an SVS slide file using OpenSlide

    Args:
        slide_path: Path to the SVS file

    Returns:
        OpenSlide object or None if loading fails
    """
    try:
        slide = OpenSlide(slide_path)
        return slide
    except Exception as e:
        print(f"Error loading slide {slide_path}: {e}")
        return None

def get_slide_info(slide):
    """
    Get information about a slide

    Args:
        slide: OpenSlide object

    Returns:
        Dictionary containing slide information
    """
    info = {
        'dimensions': slide.dimensions,
        'level_count': slide.level_count,
        'level_dimensions': slide.level_dimensions,
        'level_downsamples': slide.level_downsamples,
        'properties': dict(slide.properties)
    }
    return info

def visualize_slide_thumbnail(slide, title='Slide Thumbnail', figsize=(10, 10),
                             save_path=None, max_size=1000):
    """
    Visualize a thumbnail of the slide

    Args:
        slide: OpenSlide object
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization
        max_size: Maximum size of the thumbnail (to avoid memory issues)

    Returns:
        Thumbnail image
    """
    # Get slide dimensions
    width, height = slide.dimensions

    # Calculate thumbnail size
    if width > height:
        thumb_w = min(width, max_size)
        thumb_h = int(height * thumb_w / width)
    else:
        thumb_h = min(height, max_size)
        thumb_w = int(width * thumb_h / height)

    # Get thumbnail
    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))

    # Display thumbnail
    plt.figure(figsize=figsize)
    plt.imshow(thumbnail)
    plt.title(f"{title}\nDimensions: {width}x{height}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

    return thumbnail

def visualize_slide_region(slide, x=0, y=0, level=0, width=1024, height=1024,
                          title='Slide Region', figsize=(10, 10), save_path=None):
    """
    Visualize a region of the slide

    Args:
        slide: OpenSlide object
        x, y: Coordinates of the top-left corner of the region
        level: Magnification level (0 is highest resolution)
        width, height: Dimensions of the region
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization

    Returns:
        Region image
    """
    # Read region
    region = slide.read_region((x, y), level, (width, height))
    region = region.convert('RGB')  # Remove alpha channel

    # Display region
    plt.figure(figsize=figsize)
    plt.imshow(region)
    plt.title(f"{title}\nPosition: ({x}, {y}), Level: {level}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

    return region

def visualize_slide_grid(slide, grid_size=(3, 3), level=0, tile_size=1024,
                        title='Slide Grid', figsize=(15, 15), save_path=None):
    """
    Visualize a grid of regions from the slide

    Args:
        slide: OpenSlide object
        grid_size: Tuple (rows, cols) specifying the grid size
        level: Magnification level (0 is highest resolution)
        tile_size: Size of each tile in the grid
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization

    Returns:
        List of region images
    """
    # Get slide dimensions
    width, height = slide.dimensions

    # Calculate step size
    step_x = (width - tile_size) // (grid_size[1] - 1) if grid_size[1] > 1 else 0
    step_y = (height - tile_size) // (grid_size[0] - 1) if grid_size[0] > 1 else 0

    # Create figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Initialize list to store regions
    regions = []

    # Generate grid of regions
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Calculate coordinates
            x = j * step_x
            y = i * step_y

            # Read region
            region = slide.read_region((x, y), level, (tile_size, tile_size))
            region = region.convert('RGB')  # Remove alpha channel
            regions.append(region)

            # Display region
            if grid_size[0] == 1 and grid_size[1] == 1:
                ax = axes
            elif grid_size[0] == 1:
                ax = axes[j]
            elif grid_size[1] == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            ax.imshow(region)
            ax.set_title(f"({x}, {y})", fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

    return regions

def detect_tissue_regions(slide, level=None, threshold=220):
    """
    Detect tissue regions in a slide using simple thresholding

    Args:
        slide: OpenSlide object
        level: Magnification level to use for detection (default: lowest resolution)
        threshold: Pixel intensity threshold for tissue detection

    Returns:
        Binary mask of tissue regions, thumbnail image, and scale factors
    """
    # Use lowest resolution level if not specified
    if level is None:
        level = slide.level_count - 1

    # Get slide dimensions at the specified level
    width, height = slide.level_dimensions[level]

    # Get thumbnail at the specified level
    thumbnail = slide.read_region((0, 0), level, (width, height))
    thumbnail = thumbnail.convert('RGB')  # Remove alpha channel

    # Convert to grayscale
    thumbnail_gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)

    # Apply thresholding to identify tissue regions
    _, tissue_mask = cv2.threshold(thumbnail_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Calculate scale factors to map back to level 0
    scale_factor_x = slide.dimensions[0] / width
    scale_factor_y = slide.dimensions[1] / height

    return tissue_mask, thumbnail, (scale_factor_x, scale_factor_y)

def visualize_tissue_detection(slide, tissue_mask, thumbnail, title='Tissue Detection',
                              figsize=(15, 10), save_path=None):
    """
    Visualize tissue detection results

    Args:
        slide: OpenSlide object
        tissue_mask: Binary mask of tissue regions
        thumbnail: Thumbnail image
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Display original thumbnail
    ax1.imshow(thumbnail)
    ax1.set_title('Original Thumbnail', fontsize=14)
    ax1.axis('off')

    # Display tissue mask
    ax2.imshow(tissue_mask, cmap='gray')
    ax2.set_title('Tissue Mask', fontsize=14)
    ax2.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

def extract_tissue_tiles(slide, tissue_mask, scale_factors, level=0, tile_size=1024,
                        overlap=0, max_tiles=10):
    """
    Extract tiles from tissue regions

    Args:
        slide: OpenSlide object
        tissue_mask: Binary mask of tissue regions
        scale_factors: Tuple (scale_x, scale_y) to map from mask to level 0
        level: Magnification level for extracted tiles
        tile_size: Size of tiles to extract
        overlap: Overlap between adjacent tiles
        max_tiles: Maximum number of tiles to extract

    Returns:
        List of (x, y, tile) tuples
    """
    # Find contours in the tissue mask
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Calculate step size
    step = tile_size - overlap

    # Initialize list to store tiles
    tiles = []

    # Extract tiles from each contour
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Scale to level 0 coordinates
        x_orig = int(x * scale_factors[0])
        y_orig = int(y * scale_factors[1])
        w_orig = int(w * scale_factors[0])
        h_orig = int(h * scale_factors[1])

        # Generate grid of coordinates within the bounding box
        for i in range(x_orig, x_orig + w_orig - tile_size + 1, step):
            for j in range(y_orig, y_orig + h_orig - tile_size + 1, step):
                # Extract tile
                tile = slide.read_region((i, j), level, (tile_size, tile_size))
                tile = tile.convert('RGB')  # Remove alpha channel

                # Add to list
                tiles.append((i, j, tile))

                # Check if we've reached the maximum number of tiles
                if len(tiles) >= max_tiles:
                    return tiles

    return tiles

def visualize_extracted_tiles(tiles, grid_size=None, title='Extracted Tiles',
                             figsize=(15, 15), save_path=None):
    """
    Visualize extracted tiles

    Args:
        tiles: List of (x, y, tile) tuples
        grid_size: Tuple (rows, cols) specifying the grid size (default: square grid)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Determine grid size if not specified
    if grid_size is None:
        grid_size = (int(np.ceil(np.sqrt(len(tiles)))), int(np.ceil(np.sqrt(len(tiles)))))

    # Create figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Flatten axes array for easier indexing
    if grid_size[0] == 1 and grid_size[1] == 1:
        axes = np.array([axes])
    elif grid_size[0] == 1 or grid_size[1] == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Display tiles
    for i, (x, y, tile) in enumerate(tiles):
        if i < len(axes):
            axes[i].imshow(tile)
            axes[i].set_title(f"({x}, {y})", fontsize=10)
            axes[i].axis('off')

    # Hide empty subplots
    for i in range(len(tiles), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

def visualize_magnification_levels(slide, x=None, y=None, figsize=(15, 10),
                                 title='Magnification Levels', save_path=None):
    """
    Visualize the same region at different magnification levels

    Args:
        slide: OpenSlide object
        x, y: Coordinates of the center of the region (default: center of slide)
        figsize: Figure size
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Get slide dimensions
    width, height = slide.dimensions

    # Use center of slide if coordinates not specified
    if x is None:
        x = width // 2
    if y is None:
        y = height // 2

    # Create figure
    fig, axes = plt.subplots(1, slide.level_count, figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Ensure axes is an array
    if slide.level_count == 1:
        axes = np.array([axes])

    # Display region at each magnification level
    for level in range(slide.level_count):
        # Calculate size of region to extract (fixed size at level 0)
        size = 1024
        level_size = int(size / slide.level_downsamples[level])

        # Calculate coordinates for the region
        level_x = max(0, x - size // 2)
        level_y = max(0, y - size // 2)

        # Read region
        region = slide.read_region((level_x, level_y), level, (level_size, level_size))
        region = region.convert('RGB')  # Remove alpha channel

        # Display region
        axes[level].imshow(region)
        axes[level].set_title(f"Level {level}\nDownsample: {slide.level_downsamples[level]:.1f}x", fontsize=10)
        axes[level].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

def visualize_slide_with_annotations(slide, annotations, level=None,
                                   title='Slide with Annotations', figsize=(12, 12),
                                   save_path=None, max_size=1000):
    """
    Visualize a slide with annotations

    Args:
        slide: OpenSlide object
        annotations: List of (x, y, w, h, label, color) tuples
        level: Magnification level (default: lowest resolution)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization
        max_size: Maximum size of the thumbnail

    Returns:
        None
    """
    # Use lowest resolution level if not specified
    if level is None:
        level = slide.level_count - 1

    # Get slide dimensions at the specified level
    width, height = slide.level_dimensions[level]

    # Get thumbnail at the specified level
    thumbnail = slide.read_region((0, 0), level, (width, height))
    thumbnail = thumbnail.convert('RGB')  # Remove alpha channel

    # Calculate scale factor to map from level 0 to the current level
    scale_factor_x = width / slide.dimensions[0]
    scale_factor_y = height / slide.dimensions[1]

    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(thumbnail)

    # Add annotations
    for x, y, w, h, label, color in annotations:
        # Scale coordinates to the current level
        x_scaled = x * scale_factor_x
        y_scaled = y * scale_factor_y
        w_scaled = w * scale_factor_x
        h_scaled = h * scale_factor_y

        # Add rectangle
        rect = Rectangle((x_scaled, y_scaled), w_scaled, h_scaled,
                        linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)

        # Add label
        plt.text(x_scaled, y_scaled - 5, label, color=color, fontsize=10,
                backgroundcolor='white', ha='left', va='bottom')

    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

def detect_lung_nodules(slide, level=None, threshold=220, min_size=1000, max_size=50000):
    """
    Detect potential lung nodules in a slide using image processing techniques

    Args:
        slide: OpenSlide object
        level: Magnification level (default: lowest resolution)
        threshold: Pixel intensity threshold for tissue detection
        min_size: Minimum size of nodule in pixels
        max_size: Maximum size of nodule in pixels

    Returns:
        List of (x, y, w, h) tuples representing potential nodules, thumbnail image, and scale factors
    """
    # Use lowest resolution level if not specified
    if level is None:
        level = slide.level_count - 1

    # Get slide dimensions at the specified level
    width, height = slide.level_dimensions[level]

    # Get thumbnail at the specified level
    thumbnail = slide.read_region((0, 0), level, (width, height))
    thumbnail = thumbnail.convert('RGB')  # Remove alpha channel

    # Convert to grayscale
    thumbnail_gray = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2GRAY)

    # Apply thresholding to identify tissue regions
    _, tissue_mask = cv2.threshold(thumbnail_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to enhance nodule detection
    kernel = np.ones((5, 5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size and shape to identify potential nodules
    nodules = []
    for contour in contours:
        # Get area
        area = cv2.contourArea(contour)

        # Filter by size
        if min_size <= area <= max_size:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate circularity (4*pi*area/perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Filter by circularity (nodules tend to be more circular)
            if circularity > 0.6:
                nodules.append((x, y, w, h))

    # Calculate scale factors to map back to level 0
    scale_factor_x = slide.dimensions[0] / width
    scale_factor_y = slide.dimensions[1] / height

    return nodules, thumbnail, (scale_factor_x, scale_factor_y)

def visualize_lung_nodules(slide, nodules, thumbnail, title='Lung Nodule Detection',
                         figsize=(15, 10), save_path=None):
    """
    Visualize detected lung nodules

    Args:
        slide: OpenSlide object
        nodules: List of (x, y, w, h) tuples representing potential nodules
        thumbnail: Thumbnail image
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Create figure
    plt.figure(figsize=figsize)
    plt.imshow(thumbnail)
    plt.title(title, fontsize=16)

    # Add rectangles for nodules
    for i, (x, y, w, h) in enumerate(nodules):
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(x, y - 5, f"Nodule {i+1}", color='red', fontsize=10,
                backgroundcolor='white', ha='left', va='bottom')

    plt.axis('off')
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

def extract_nodule_images(slide, nodules, scale_factors, level=0, padding=50):
    """
    Extract images of detected nodules at high resolution

    Args:
        slide: OpenSlide object
        nodules: List of (x, y, w, h) tuples representing potential nodules
        scale_factors: Tuple (scale_x, scale_y) to map from thumbnail to level 0
        level: Magnification level for extracted nodules
        padding: Padding around nodule in pixels

    Returns:
        List of (x, y, nodule_image) tuples
    """
    # Initialize list to store nodule images
    nodule_images = []

    # Extract each nodule
    for x, y, w, h in nodules:
        # Scale coordinates to level 0
        x_orig = int(x * scale_factors[0])
        y_orig = int(y * scale_factors[1])
        w_orig = int(w * scale_factors[0])
        h_orig = int(h * scale_factors[1])

        # Add padding
        x_pad = max(0, x_orig - padding)
        y_pad = max(0, y_orig - padding)
        w_pad = w_orig + 2 * padding
        h_pad = h_orig + 2 * padding

        # Extract nodule image
        nodule_img = slide.read_region((x_pad, y_pad), level, (w_pad, h_pad))
        nodule_img = nodule_img.convert('RGB')  # Remove alpha channel

        # Add to list
        nodule_images.append((x_orig, y_orig, nodule_img))

    return nodule_images

def visualize_nodule_grid(nodule_images, grid_size=None, title='Lung Nodules',
                        figsize=(15, 15), save_path=None):
    """
    Visualize a grid of extracted nodule images

    Args:
        nodule_images: List of (x, y, nodule_image) tuples
        grid_size: Tuple (rows, cols) specifying the grid size (default: square grid)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Determine grid size if not specified
    if grid_size is None:
        grid_size = (int(np.ceil(np.sqrt(len(nodule_images)))), int(np.ceil(np.sqrt(len(nodule_images)))))

    # Create figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Flatten axes array for easier indexing
    if grid_size[0] == 1 and grid_size[1] == 1:
        axes = np.array([axes])
    elif grid_size[0] == 1 or grid_size[1] == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Display nodules
    for i, (x, y, nodule_img) in enumerate(nodule_images):
        if i < len(axes):
            axes[i].imshow(nodule_img)
            axes[i].set_title(f"Nodule {i+1} at ({x}, {y})", fontsize=10)
            axes[i].axis('off')

    # Hide empty subplots
    for i in range(len(nodule_images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)

    plt.show()

def process_slide_batch(slide_paths, output_dir='slide_visualizations', max_slides=None):
    """
    Process a batch of slides and generate visualizations

    Args:
        slide_paths: List of paths to SVS files
        output_dir: Directory to save visualizations
        max_slides: Maximum number of slides to process

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Limit number of slides if specified
    if max_slides is not None:
        slide_paths = slide_paths[:max_slides]

    # Process each slide
    for i, slide_path in enumerate(tqdm(slide_paths, desc="Processing slides")):
        try:
            # Extract slide name
            slide_name = os.path.splitext(os.path.basename(slide_path))[0]

            # Create slide-specific output directory
            slide_dir = os.path.join(output_dir, slide_name)
            os.makedirs(slide_dir, exist_ok=True)

            # Load slide
            slide = load_svs_slide(slide_path)
            if slide is None:
                continue

            # Get slide info
            info = get_slide_info(slide)

            # Save slide info as text file
            with open(os.path.join(slide_dir, 'slide_info.txt'), 'w') as f:
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")

            # Visualize slide thumbnail
            visualize_slide_thumbnail(slide, title=f"Slide: {slide_name}",
                                    save_path=os.path.join(slide_dir, 'thumbnail.png'))

            # Detect tissue regions
            tissue_mask, thumbnail, scale_factors = detect_tissue_regions(slide)

            # Visualize tissue detection
            visualize_tissue_detection(slide, tissue_mask, thumbnail,
                                     save_path=os.path.join(slide_dir, 'tissue_detection.png'))

            # Extract tissue tiles
            tiles = extract_tissue_tiles(slide, tissue_mask, scale_factors, max_tiles=9)

            # Visualize extracted tiles
            visualize_extracted_tiles(tiles, grid_size=(3, 3),
                                    save_path=os.path.join(slide_dir, 'extracted_tiles.png'))

            # Visualize magnification levels
            visualize_magnification_levels(slide,
                                         save_path=os.path.join(slide_dir, 'magnification_levels.png'))

            # Close slide
            slide.close()

        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")

    print(f"Processed {len(slide_paths)} slides. Visualizations saved to {output_dir}")

def main():
    """
    Main function to demonstrate SVS visualization
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SVS Visualization')
    parser.add_argument('--slide_dir', type=str, required=True, help='Directory containing SVS files')
    parser.add_argument('--output_dir', type=str, default='slide_visualizations', help='Directory to save visualizations')
    parser.add_argument('--max_slides', type=int, default=None, help='Maximum number of slides to process')
    args = parser.parse_args()

    # Get list of SVS files
    slide_paths = [os.path.join(args.slide_dir, f) for f in os.listdir(args.slide_dir) if f.endswith('.svs')]

    # Process slides
    process_slide_batch(slide_paths, output_dir=args.output_dir, max_slides=args.max_slides)

if __name__ == '__main__':
    main()
