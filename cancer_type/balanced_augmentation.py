"""
Balanced Data Augmentation for Cancer Type Classification

This script implements balanced data augmentation to address class imbalance
in the cancer type classification dataset. It applies more augmentations to
underrepresented classes to create a balanced training dataset.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
import shutil
import cv2
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """
    Load the processed data CSV file
    
    Returns:
        DataFrame containing processed image information
    """
    processed_data_path = 'processed_images/processed_data.csv'
    
    if not os.path.exists(processed_data_path):
        print(f"Error: Processed data file not found: {processed_data_path}")
        return None
    
    try:
        processed_df = pd.read_csv(processed_data_path)
        print(f"Loaded processed data with {len(processed_df)} entries")
        return processed_df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def analyze_class_distribution(processed_df):
    """
    Analyze class distribution and determine augmentation factors
    
    Args:
        processed_df: DataFrame containing processed image information
        
    Returns:
        Dictionary mapping categories to augmentation factors
    """
    # Count images by category and split
    train_counts = processed_df[processed_df['split'] == 'train']['category'].value_counts()
    
    print("\nTraining images per category:")
    print(train_counts)
    
    # Calculate augmentation factors
    max_count = train_counts.max()
    augmentation_factors = {}
    
    for category, count in train_counts.items():
        # Calculate how many more augmentations we need
        factor = max(1, int(np.ceil(max_count / count)))
        augmentation_factors[category] = factor
    
    print("\nAugmentation factors per category:")
    for category, factor in augmentation_factors.items():
        print(f"  {category}: {factor}x")
    
    return augmentation_factors

def apply_advanced_augmentation(image):
    """
    Apply advanced data augmentation techniques to an image
    
    Args:
        image: PIL Image object
        
    Returns:
        Augmented PIL Image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Randomly choose augmentation technique
    augmentation_type = random.choice([
        'color_jitter',
        'gaussian_blur',
        'random_crop',
        'perspective',
        'elastic'
    ])
    
    if augmentation_type == 'color_jitter':
        # Random color jittering
        # Adjust brightness
        brightness_factor = random.uniform(0.8, 1.2)
        img_array = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)
        
        # Adjust contrast
        contrast_factor = random.uniform(0.8, 1.2)
        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
        img_array = np.clip((img_array - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Adjust saturation (convert to HSV, modify S channel, convert back to RGB)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation_factor = random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    elif augmentation_type == 'gaussian_blur':
        # Apply Gaussian blur
        kernel_size = random.choice([3, 5, 7])
        img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    
    elif augmentation_type == 'random_crop':
        # Random crop and resize
        h, w = img_array.shape[:2]
        crop_ratio = random.uniform(0.8, 0.95)
        crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
        
        # Random crop coordinates
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # Crop image
        img_array = img_array[top:top+crop_h, left:left+crop_w]
        
        # Resize back to original size
        img_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    elif augmentation_type == 'perspective':
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
        img_array = cv2.warpPerspective(img_array, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif augmentation_type == 'elastic':
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
        img_array = cv2.remap(img_array, map_x, map_y, 
                             interpolation=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT)
    
    # Convert back to PIL Image
    augmented_img = Image.fromarray(img_array)
    
    return augmented_img

def create_balanced_dataset(processed_df, augmentation_factors, output_dir='balanced_images'):
    """
    Create a balanced dataset by applying additional augmentations
    
    Args:
        processed_df: DataFrame containing processed image information
        augmentation_factors: Dictionary mapping categories to augmentation factors
        output_dir: Directory to save balanced dataset
        
    Returns:
        DataFrame with balanced dataset information
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create split and category subdirectories
    splits = ['train', 'val', 'test']
    categories = processed_df['category'].unique()
    
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        for category in categories:
            category_dir = os.path.join(split_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
    
    # Initialize list to store balanced dataset information
    balanced_data = []
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Get images for this split
        split_df = processed_df[processed_df['split'] == split]
        
        # Process each category
        for category in categories:
            # Get images for this category
            category_df = split_df[split_df['category'] == category]
            
            # Get augmentation factor for this category
            factor = augmentation_factors.get(category, 1)
            
            # Process each image
            for _, row in tqdm(category_df.iterrows(), total=len(category_df), 
                              desc=f"Processing {category}"):
                # Get image path
                image_path = os.path.join('processed_images', split, category, row['processed_filename'])
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Load image
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
                
                # Copy original image to balanced dataset
                output_path = os.path.join(output_dir, split, category, row['processed_filename'])
                image.save(output_path)
                
                # Add to balanced data
                balanced_data.append({
                    'pid': row['pid'],
                    'original_filename': row['original_filename'],
                    'processed_filename': row['processed_filename'],
                    'category': category,
                    'split': split,
                    'augmentation': 'original'
                })
                
                # Apply additional augmentations for training set
                if split == 'train':
                    # Apply factor-1 additional augmentations (we already have the original)
                    for i in range(factor - 1):
                        # Apply advanced augmentation
                        augmented_img = apply_advanced_augmentation(image)
                        
                        # Create output filename
                        output_filename = f"{row['pid']}_{i}_adv.png"
                        output_path = os.path.join(output_dir, split, category, output_filename)
                        
                        # Save augmented image
                        augmented_img.save(output_path)
                        
                        # Add to balanced data
                        balanced_data.append({
                            'pid': row['pid'],
                            'original_filename': row['original_filename'],
                            'processed_filename': output_filename,
                            'category': category,
                            'split': split,
                            'augmentation': f'advanced_{i}'
                        })
    
    # Create DataFrame from balanced data
    balanced_df = pd.DataFrame(balanced_data)
    
    # Save balanced data
    balanced_df.to_csv(os.path.join(output_dir, 'balanced_data.csv'), index=False)
    
    print(f"\nCreated balanced dataset with {len(balanced_df)} images")
    
    return balanced_df

def analyze_balanced_dataset(balanced_df):
    """
    Analyze the balanced dataset and generate statistics
    
    Args:
        balanced_df: DataFrame containing balanced dataset information
        
    Returns:
        None
    """
    print("\nAnalyzing balanced dataset:")
    
    # Count images by category
    category_counts = balanced_df['category'].value_counts()
    print("\nImages per category:")
    print(category_counts)
    
    # Count images by split
    split_counts = balanced_df['split'].value_counts()
    print("\nImages per split:")
    print(split_counts)
    
    # Count images by category and split
    category_split_counts = pd.crosstab(balanced_df['category'], balanced_df['split'])
    print("\nImages per category and split:")
    print(category_split_counts)
    
    # Visualize category distribution
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar')
    plt.title('Number of Images per Category (Balanced)')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('balanced_category_distribution.png')
    
    # Visualize split distribution
    plt.figure(figsize=(10, 6))
    split_counts.plot(kind='bar')
    plt.title('Number of Images per Split (Balanced)')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('balanced_split_distribution.png')
    
    # Visualize category and split distribution
    plt.figure(figsize=(14, 8))
    category_split_counts.plot(kind='bar', stacked=True)
    plt.title('Number of Images per Category and Split (Balanced)')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('balanced_category_split_distribution.png')
    
    # Compare original and balanced distributions for training set
    train_df = balanced_df[balanced_df['split'] == 'train']
    train_category_counts = train_df['category'].value_counts()
    
    # Load original processed data
    processed_df = pd.read_csv('processed_images/processed_data.csv')
    orig_train_df = processed_df[processed_df['split'] == 'train']
    orig_train_category_counts = orig_train_df['category'].value_counts()
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Original': orig_train_category_counts,
        'Balanced': train_category_counts
    })
    
    print("\nComparison of original and balanced training sets:")
    print(comparison_df)
    
    # Visualize comparison
    plt.figure(figsize=(14, 8))
    comparison_df.plot(kind='bar')
    plt.title('Comparison of Original and Balanced Training Sets')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('original_vs_balanced_comparison.png')

def main():
    """
    Main function to run the balanced augmentation pipeline
    """
    print("=" * 80)
    print("BALANCED DATA AUGMENTATION FOR CANCER TYPE CLASSIFICATION")
    print("=" * 80)
    
    # Load processed data
    processed_df = load_processed_data()
    if processed_df is None:
        print("Error: Could not load processed data. Exiting.")
        return
    
    # Analyze class distribution and determine augmentation factors
    augmentation_factors = analyze_class_distribution(processed_df)
    
    # Create balanced dataset
    balanced_df = create_balanced_dataset(processed_df, augmentation_factors)
    
    # Analyze balanced dataset
    analyze_balanced_dataset(balanced_df)
    
    print("\n" + "=" * 80)
    print("BALANCED DATA AUGMENTATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
