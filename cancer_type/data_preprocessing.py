"""
Cancer Type Data Preprocessing

This script preprocesses pathology images for cancer type classification.
It includes functions for loading SVS files, extracting regions of interest,
applying data augmentation, and preparing data for model training.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import openslide
from openslide import OpenSlide
import cv2
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
def create_directories():
    """Create necessary directories for processed data"""
    dirs = [
        'processed_images',
        'processed_images/train',
        'processed_images/val',
        'processed_images/test',
    ]

    # Create category subdirectories based on major cancer categories
    categories = [
        'Adenocarcinoma',
        'Squamous_Cell_Carcinoma',
        'Neuroendocrine_Carcinoma',
        'Large_Cell_Carcinoma',
        'Other'
    ]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

        # Create category subdirectories
        if d != 'processed_images':
            for category in categories:
                category_dir = os.path.join(d, category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                    print(f"Created directory: {category_dir}")

    print("All directories created successfully.")

def load_metadata():
    """
    Load and prepare the metadata for image processing
    """
    try:
        # Try to load the prepared modeling data first
        if os.path.exists('cancer_type_modeling_data.csv'):
            print("Loading prepared modeling data...")
            metadata = pd.read_csv('cancer_type_modeling_data.csv')
        else:
            # If not available, load and merge the raw data
            print("Prepared data not found. Loading and merging raw data...")
            clinical = pd.read_csv("../Data/type/Lung Cancer/lung_cancer.csv")
            pathology = pd.read_csv("../Data/type/Pathology Images/pathology_images.csv")

            # Merge datasets
            metadata = pd.merge(clinical, pathology, on='pid', how='inner')

            # Map cancer types
            cancer_types = {
                "8140": "Adenocarcinoma",
                "8046": "Neuroendocrine_Carcinoma",
                "8070": "Squamous_Cell_Carcinoma",
                "8250": "Bronchioloalveolar_Carcinoma",
                "8041": "Small_Cell_Carcinoma",
                "8071": "Keratinizing_Squamous_Cell_Carcinoma",
                # Add other mappings as needed
            }

            # Clean lc_morph codes and map to cancer types
            metadata['lc_morph'] = metadata['lc_morph'].astype(str).str.strip()
            metadata['cancer_type'] = metadata['lc_morph'].map(cancer_types)

            # Group into major categories
            major_categories = {
                'Adenocarcinoma': ['Adenocarcinoma', 'Bronchioloalveolar_Carcinoma', 'Lepidic_Predominant_Adenocarcinoma',
                                  'Adenocarcinoma_with_Mixed_Subtypes', 'Papillary_Adenocarcinoma', 'Clear_Cell_Adenocarcinoma',
                                  'Mucinous_Adenocarcinoma', 'Mucin_Producing_Adenocarcinoma', 'Acinar_Cell_Carcinoma'],
                'Squamous_Cell_Carcinoma': ['Squamous_Cell_Carcinoma', 'Keratinizing_Squamous_Cell_Carcinoma',
                                           'Non_Keratinizing_Squamous_Cell_Carcinoma'],
                'Neuroendocrine_Carcinoma': ['Neuroendocrine_Carcinoma', 'Small_Cell_Carcinoma', 'Carcinoid_Tumor',
                                            'Atypical_Carcinoid', 'Large_Cell_Neuroendocrine_Carcinoma'],
                'Large_Cell_Carcinoma': ['Large_Cell_Carcinoma'],
                'Other': ['Carcinoma_NOS', 'Neoplasm_Malignant', 'Sarcomatoid_Carcinoma', 'Adenosquamous_Carcinoma']
            }

            # Function to map cancer type to major category
            def map_to_major_category(cancer_type):
                if pd.isna(cancer_type):
                    return 'Unknown'

                for category, types in major_categories.items():
                    if any(t in cancer_type for t in types):
                        return category
                return 'Other'

            # Apply mapping
            metadata['major_category'] = metadata['cancer_type'].apply(map_to_major_category)

        print(f"Loaded metadata with {len(metadata)} entries")
        return metadata

    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

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

def extract_roi(slide, level=0, x=0, y=0, width=1024, height=1024):
    """
    Extract a region of interest from a slide

    Args:
        slide: OpenSlide object
        level: Magnification level (0 is highest resolution)
        x, y: Coordinates of the top-left corner of the ROI
        width, height: Dimensions of the ROI

    Returns:
        PIL Image object containing the ROI
    """
    try:
        roi = slide.read_region((x, y), level, (width, height))
        # Convert to RGB (remove alpha channel)
        roi = roi.convert('RGB')
        return roi
    except Exception as e:
        print(f"Error extracting ROI: {e}")
        return None

def extract_tissue_regions(slide, level=0, tile_size=1024, overlap=0):
    """
    Extract tissue regions from a slide using simple thresholding

    Args:
        slide: OpenSlide object
        level: Magnification level
        tile_size: Size of tiles to extract
        overlap: Overlap between adjacent tiles

    Returns:
        List of (x, y) coordinates for tissue regions
    """
    # Get slide dimensions
    width, height = slide.dimensions

    # Create a thumbnail for tissue detection
    thumbnail = slide.get_thumbnail((width//64, height//64))
    thumbnail_np = np.array(thumbnail)

    # Convert to grayscale
    gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to identify tissue regions
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale factor from thumbnail to original
    scale_x = width / thumbnail.width
    scale_y = height / thumbnail.height

    # Generate coordinates for tissue regions
    coords = []
    step = tile_size - overlap

    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Scale to original dimensions
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)
        w_orig = int(w * scale_x)
        h_orig = int(h * scale_y)

        # Generate grid of coordinates within the bounding box
        for i in range(x_orig, x_orig + w_orig - tile_size + 1, step):
            for j in range(y_orig, y_orig + h_orig - tile_size + 1, step):
                coords.append((i, j))

    # Shuffle coordinates to get a random sample
    random.shuffle(coords)

    # Limit the number of tiles to a reasonable amount
    max_tiles = 10
    if len(coords) > max_tiles:
        coords = coords[:max_tiles]

    return coords

def preprocess_image(image, target_size=(512, 512), normalize=True,
                  color_normalization=False, contrast_enhancement=False):
    """
    Preprocess an image for model input

    Args:
        image: PIL Image object
        target_size: Target dimensions (width, height)
        normalize: Whether to normalize pixel values to [0, 1]
        color_normalization: Whether to apply color normalization
        contrast_enhancement: Whether to apply contrast enhancement

    Returns:
        Preprocessed PIL Image
    """
    # Resize
    image = image.resize(target_size, Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image)

    # Apply color normalization (stain normalization for pathology)
    if color_normalization and len(img_array.shape) == 3:
        # Simple color normalization - normalize each channel separately
        for i in range(3):  # RGB channels
            channel = img_array[:, :, i]
            if np.std(channel) > 0:
                img_array[:, :, i] = (channel - np.mean(channel)) / np.std(channel) * 64 + 128

    # Apply contrast enhancement
    if contrast_enhancement:
        # Convert to LAB color space for luminance enhancement
        if len(img_array.shape) == 3:
            # For RGB images
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l_channel = img_lab[:, :, 0]

            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # Update L-channel and convert back to RGB
            img_lab[:, :, 0] = cl
            img_array = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)

    # Normalize pixel values to [0, 1]
    if normalize:
        img_array = img_array / 255.0

    # Convert back to PIL Image
    if normalize:
        processed_img = Image.fromarray((img_array * 255).astype(np.uint8))
    else:
        processed_img = Image.fromarray(img_array.astype(np.uint8))

    return processed_img

def apply_augmentation(image, augmentation_types=None):
    """
    Apply data augmentation to an image

    Args:
        image: PIL Image object
        augmentation_types: List of augmentation types to apply (default: None, applies all)

    Returns:
        List of augmented PIL Images
    """
    augmented_images = []

    # Define available augmentation types
    all_augmentation_types = [
        'original', 'flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270',
        'color_jitter', 'gaussian_blur', 'random_crop', 'perspective', 'elastic'
    ]

    # Use specified augmentation types or all types
    if augmentation_types is None:
        augmentation_types = all_augmentation_types

    # Original image
    if 'original' in augmentation_types:
        augmented_images.append(image)

    # Basic geometric transformations
    if 'flip_h' in augmentation_types:
        h_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(h_flip)

    if 'flip_v' in augmentation_types:
        v_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
        augmented_images.append(v_flip)

    if 'rotate_90' in augmentation_types:
        rot_90 = image.transpose(Image.ROTATE_90)
        augmented_images.append(rot_90)

    if 'rotate_180' in augmentation_types:
        rot_180 = image.transpose(Image.ROTATE_180)
        augmented_images.append(rot_180)

    if 'rotate_270' in augmentation_types:
        rot_270 = image.transpose(Image.ROTATE_270)
        augmented_images.append(rot_270)

    # Advanced augmentations
    img_array = np.array(image)

    if 'color_jitter' in augmentation_types:
        # Random color jittering
        # Adjust brightness
        brightness_factor = random.uniform(0.8, 1.2)
        color_jitter = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)

        # Adjust contrast
        contrast_factor = random.uniform(0.8, 1.2)
        mean = np.mean(color_jitter, axis=(0, 1), keepdims=True)
        color_jitter = np.clip((color_jitter - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

        # Adjust saturation (convert to HSV, modify S channel, convert back to RGB)
        if len(img_array.shape) == 3:  # Only for color images
            hsv = cv2.cvtColor(color_jitter, cv2.COLOR_RGB2HSV)
            saturation_factor = random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
            color_jitter = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        augmented_images.append(Image.fromarray(color_jitter))

    if 'gaussian_blur' in augmentation_types:
        # Apply Gaussian blur
        kernel_size = random.choice([3, 5, 7])
        gaussian_blur = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
        augmented_images.append(Image.fromarray(gaussian_blur))

    if 'random_crop' in augmentation_types:
        # Random crop and resize
        h, w = img_array.shape[:2]
        crop_ratio = random.uniform(0.8, 0.95)
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)

        # Random crop coordinates
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # Crop image
        cropped = img_array[top:top+crop_h, left:left+crop_w]

        # Resize back to original size
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        augmented_images.append(Image.fromarray(resized))

    if 'perspective' in augmentation_types:
        # Apply perspective transformation
        h, w = img_array.shape[:2]

        # Define source points
        src_pts = np.float32([
            [0, 0],
            [w-1, 0],
            [0, h-1],
            [w-1, h-1]
        ])

        # Define destination points with random perturbation
        max_shift = 0.1  # Maximum shift as a fraction of width/height
        dst_pts = np.float32([
            [random.uniform(0, w*max_shift), random.uniform(0, h*max_shift)],
            [random.uniform(w*(1-max_shift), w-1), random.uniform(0, h*max_shift)],
            [random.uniform(0, w*max_shift), random.uniform(h*(1-max_shift), h-1)],
            [random.uniform(w*(1-max_shift), w-1), random.uniform(h*(1-max_shift), h-1)]
        ])

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Apply perspective transformation
        perspective = cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(Image.fromarray(perspective))

    if 'elastic' in augmentation_types:
        # Apply elastic transformation
        h, w = img_array.shape[:2]

        # Create displacement fields
        alpha = random.uniform(40, 60)  # Displacement scale
        sigma = random.uniform(4, 6)    # Smoothing factor

        # Create random displacement fields
        dx = np.random.rand(h, w) * 2 - 1
        dy = np.random.rand(h, w) * 2 - 1

        # Smooth displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

        # Create meshgrid
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Displace meshgrid
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        # Apply elastic transformation
        elastic = cv2.remap(img_array, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        augmented_images.append(Image.fromarray(elastic))

    return augmented_images

def process_slides(metadata, image_dir, output_dir='processed_images', max_slides=None,
                 target_size=(512, 512), color_normalization=True, contrast_enhancement=True,
                 augmentation_types=None, basic_augmentation_only=False):
    """
    Process slides and extract ROIs

    Args:
        metadata: DataFrame containing slide metadata
        image_dir: Directory containing SVS files
        output_dir: Directory to save processed images
        max_slides: Maximum number of slides to process (for testing)
        target_size: Target size for processed images (default: (512, 512))
        color_normalization: Whether to apply color normalization (default: True)
        contrast_enhancement: Whether to apply contrast enhancement (default: True)
        augmentation_types: List of augmentation types to apply (default: None, applies all)
        basic_augmentation_only: Whether to use only basic augmentations (default: False)

    Returns:
        DataFrame with processed image information
    """
    # Create a new DataFrame to store processed image information
    processed_data = []

    # Get list of slides to process
    slides_to_process = metadata[['pid', 'image_filename', 'major_category']].drop_duplicates()

    if max_slides is not None:
        slides_to_process = slides_to_process.head(max_slides)

    print(f"Processing {len(slides_to_process)} slides...")

    # Define basic augmentation types
    basic_augmentation = ['original', 'flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270']

    # Use basic augmentation if specified
    if basic_augmentation_only:
        augmentation_types = basic_augmentation

    # Process each slide
    for _, row in tqdm(slides_to_process.iterrows(), total=len(slides_to_process)):
        pid = row['pid']
        filename = row['image_filename']
        category = row['major_category']

        # Skip if category is unknown
        if pd.isna(category) or category == 'Unknown':
            continue

        # Replace spaces with underscores in category name
        category = category.replace(' ', '_')

        # Construct slide path
        slide_path = os.path.join(image_dir, filename)

        # Check if file exists
        if not os.path.exists(slide_path):
            print(f"Slide not found: {slide_path}")
            continue

        # Load slide
        slide = load_svs_slide(slide_path)
        if slide is None:
            continue

        # Extract tissue regions
        coords = extract_tissue_regions(slide, tile_size=target_size[0])

        # Process each region
        for i, (x, y) in enumerate(coords):
            # Extract ROI
            roi = extract_roi(slide, level=0, x=x, y=y, width=target_size[0], height=target_size[1])
            if roi is None:
                continue

            # Preprocess image
            processed_roi = preprocess_image(
                roi,
                target_size=target_size,
                normalize=True,
                color_normalization=color_normalization,
                contrast_enhancement=contrast_enhancement
            )

            # Apply augmentation
            augmented_rois = apply_augmentation(processed_roi, augmentation_types=augmentation_types)

            # Split into train/val/test (70/15/15 split)
            split_rand = random.random()
            if split_rand < 0.7:
                split = 'train'
            elif split_rand < 0.85:
                split = 'val'
            else:
                split = 'test'

            # Save augmented images
            for j, aug_roi in enumerate(augmented_rois):
                # Create output filename
                output_filename = f"{pid}_{i}_{j}.png"
                output_path = os.path.join(output_dir, split, category, output_filename)

                # Save image
                aug_roi.save(output_path)

                # Determine augmentation type
                if augmentation_types is not None and j < len(augmentation_types):
                    aug_type = augmentation_types[j]
                else:
                    aug_type = f"augmentation_{j}"

                # Add to processed data
                processed_data.append({
                    'pid': pid,
                    'original_filename': filename,
                    'processed_filename': output_filename,
                    'category': category,
                    'split': split,
                    'x': x,
                    'y': y,
                    'augmentation_type': aug_type,
                    'color_normalization': color_normalization,
                    'contrast_enhancement': contrast_enhancement,
                    'target_size': f"{target_size[0]}x{target_size[1]}"
                })

    # Create DataFrame from processed data
    processed_df = pd.DataFrame(processed_data)

    # Save processed data
    processed_df.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)

    print(f"Processed {len(processed_df)} images from {len(slides_to_process)} slides")

    # Save processing parameters
    processing_params = {
        'target_size': target_size,
        'color_normalization': color_normalization,
        'contrast_enhancement': contrast_enhancement,
        'augmentation_types': augmentation_types,
        'basic_augmentation_only': basic_augmentation_only,
        'num_slides_processed': len(slides_to_process),
        'num_images_generated': len(processed_df)
    }

    # Save parameters as JSON
    import json
    with open(os.path.join(output_dir, 'processing_params.json'), 'w') as f:
        json.dump(processing_params, f, indent=4)

    return processed_df

def analyze_processed_data(processed_df):
    """
    Analyze the processed data and generate statistics

    Args:
        processed_df: DataFrame containing processed image information

    Returns:
        None
    """
    print("\nAnalyzing processed data:")

    # Count images by category
    category_counts = processed_df['category'].value_counts()
    print("\nImages per category:")
    print(category_counts)

    # Count images by split
    split_counts = processed_df['split'].value_counts()
    print("\nImages per split:")
    print(split_counts)

    # Count images by category and split
    category_split_counts = pd.crosstab(processed_df['category'], processed_df['split'])
    print("\nImages per category and split:")
    print(category_split_counts)

    # Visualize category distribution
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar')
    plt.title('Number of Images per Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('processed_category_distribution.png')

    # Visualize split distribution
    plt.figure(figsize=(10, 6))
    split_counts.plot(kind='bar')
    plt.title('Number of Images per Split')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('processed_split_distribution.png')

    # Visualize category and split distribution
    plt.figure(figsize=(14, 8))
    category_split_counts.plot(kind='bar', stacked=True)
    plt.title('Number of Images per Category and Split')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('processed_category_split_distribution.png')

def main():
    """
    Main function to run the preprocessing pipeline
    """
    print("=" * 80)
    print("CANCER TYPE DATA PREPROCESSING")
    print("=" * 80)

    # Create directories
    create_directories()

    # Load metadata
    metadata = load_metadata()
    if metadata is None:
        print("Error: Could not load metadata. Exiting.")
        return

    # Set image directory
    image_dir = "../Data/type/Pathology Images/images"
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return

    # Process slides (limit to 5 slides for testing)
    processed_df = process_slides(metadata, image_dir, max_slides=5)

    # Analyze processed data
    analyze_processed_data(processed_df)

    print("\n" + "=" * 80)
    print("DATA PREPROCESSING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
