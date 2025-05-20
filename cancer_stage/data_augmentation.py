"""
Data Augmentation for CT Scans and Segmentation Masks

This module provides functions for augmenting 3D CT scans and their corresponding
segmentation masks to balance datasets and improve model generalization.

Augmentations include:
- Random rotations
- Random flips
- Random zooms
- Elastic deformations
- Gaussian noise
- Intensity shifts
- Contrast adjustments
- Gamma corrections

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import scipy.ndimage as ndimage
from skimage import exposure, transform
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import pandas as pd
import cv2

def random_rotation_3d(image, mask=None, max_angle=10, axes=(1, 2)):
    """
    Randomly rotate a 3D image and its mask.
    
    Args:
        image: 3D numpy array of the CT scan
        mask: 3D numpy array of the segmentation mask (optional)
        max_angle: Maximum rotation angle in degrees
        axes: Tuple of axes to rotate around (default: (1, 2) for axial rotation)
        
    Returns:
        Rotated image and mask (if provided)
    """
    # Generate random angle
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Rotate image
    rotated_image = ndimage.rotate(image, angle, axes=axes, reshape=False, order=1, mode='nearest')
    
    # Rotate mask if provided
    if mask is not None:
        rotated_mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0, mode='nearest')
        return rotated_image, rotated_mask
    
    return rotated_image

def random_flip_3d(image, mask=None, axis=None):
    """
    Randomly flip a 3D image and its mask along a specified axis.
    
    Args:
        image: 3D numpy array of the CT scan
        mask: 3D numpy array of the segmentation mask (optional)
        axis: Axis to flip along (if None, a random axis is chosen)
        
    Returns:
        Flipped image and mask (if provided)
    """
    # Choose random axis if not specified
    if axis is None:
        axis = np.random.choice([0, 1, 2])
    
    # Flip image
    flipped_image = np.flip(image, axis=axis).copy()
    
    # Flip mask if provided
    if mask is not None:
        flipped_mask = np.flip(mask, axis=axis).copy()
        return flipped_image, flipped_mask
    
    return flipped_image

def random_zoom_3d(image, mask=None, zoom_range=(0.9, 1.1)):
    """
    Randomly zoom a 3D image and its mask.
    
    Args:
        image: 3D numpy array of the CT scan
        mask: 3D numpy array of the segmentation mask (optional)
        zoom_range: Range of zoom factors (min, max)
        
    Returns:
        Zoomed image and mask (if provided)
    """
    # Generate random zoom factors for each dimension
    zoom_factors = np.random.uniform(zoom_range[0], zoom_range[1], 3)
    
    # Zoom image
    zoomed_image = ndimage.zoom(image, zoom_factors, order=1, mode='nearest')
    
    # Resize back to original shape
    original_shape = image.shape
    if zoomed_image.shape != original_shape:
        zoomed_image = transform.resize(zoomed_image, original_shape, order=1, mode='constant', anti_aliasing=True, preserve_range=True)
    
    # Zoom mask if provided
    if mask is not None:
        zoomed_mask = ndimage.zoom(mask, zoom_factors, order=0, mode='nearest')
        
        # Resize back to original shape
        if zoomed_mask.shape != original_shape:
            zoomed_mask = transform.resize(zoomed_mask, original_shape, order=0, mode='constant', anti_aliasing=False, preserve_range=True)
            zoomed_mask = (zoomed_mask > 0.5).astype(mask.dtype)  # Threshold to ensure binary mask
        
        return zoomed_image, zoomed_mask
    
    return zoomed_image

def elastic_transform_3d(image, mask=None, alpha=15, sigma=3, random_state=None):
    """
    Apply elastic deformation to a 3D image and its mask.
    
    Args:
        image: 3D numpy array of the CT scan
        mask: 3D numpy array of the segmentation mask (optional)
        alpha: Scaling factor for deformations
        sigma: Smoothing factor for deformations
        random_state: Random state for reproducibility
        
    Returns:
        Deformed image and mask (if provided)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    
    # Generate displacement fields
    d, h, w = shape
    dz = random_state.rand(d, h, w) * 2 - 1
    dy = random_state.rand(d, h, w) * 2 - 1
    dx = random_state.rand(d, h, w) * 2 - 1
    
    # Smooth displacement fields
    dz = ndimage.gaussian_filter(dz, sigma, mode='constant', cval=0) * alpha
    dy = ndimage.gaussian_filter(dy, sigma, mode='constant', cval=0) * alpha
    dx = ndimage.gaussian_filter(dx, sigma, mode='constant', cval=0) * alpha
    
    # Create meshgrid
    z, y, x = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing='ij')
    
    # Displace meshgrid
    indices = [z + dz, y + dy, x + dx]
    
    # Interpolate image
    deformed_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
    
    # Interpolate mask if provided
    if mask is not None:
        deformed_mask = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')
        return deformed_image, deformed_mask
    
    return deformed_image

def add_gaussian_noise(image, mean=0, std=0.01):
    """
    Add Gaussian noise to a 3D image.
    
    Args:
        image: 3D numpy array of the CT scan
        mean: Mean of the Gaussian noise
        std: Standard deviation of the Gaussian noise
        
    Returns:
        Noisy image
    """
    # Generate noise
    noise = np.random.normal(mean, std, image.shape)
    
    # Add noise to image
    noisy_image = image + noise
    
    # Clip to original range
    min_val, max_val = image.min(), image.max()
    noisy_image = np.clip(noisy_image, min_val, max_val)
    
    return noisy_image

def random_intensity_shift(image, shift_range=(-0.1, 0.1)):
    """
    Apply random intensity shift to a 3D image.
    
    Args:
        image: 3D numpy array of the CT scan
        shift_range: Range of intensity shifts (min, max)
        
    Returns:
        Shifted image
    """
    # Generate random shift
    shift = np.random.uniform(shift_range[0], shift_range[1])
    
    # Apply shift
    shifted_image = image + shift
    
    # Clip to original range
    min_val, max_val = image.min(), image.max()
    shifted_image = np.clip(shifted_image, min_val, max_val)
    
    return shifted_image

def random_contrast_adjustment(image, gain_range=(0.8, 1.2)):
    """
    Apply random contrast adjustment to a 3D image.
    
    Args:
        image: 3D numpy array of the CT scan
        gain_range: Range of contrast gains (min, max)
        
    Returns:
        Contrast-adjusted image
    """
    # Generate random gain
    gain = np.random.uniform(gain_range[0], gain_range[1])
    
    # Compute mean intensity
    mean = np.mean(image)
    
    # Apply contrast adjustment
    adjusted_image = mean + gain * (image - mean)
    
    # Clip to original range
    min_val, max_val = image.min(), image.max()
    adjusted_image = np.clip(adjusted_image, min_val, max_val)
    
    return adjusted_image

def random_gamma_correction(image, gamma_range=(0.8, 1.2)):
    """
    Apply random gamma correction to a 3D image.
    
    Args:
        image: 3D numpy array of the CT scan
        gamma_range: Range of gamma values (min, max)
        
    Returns:
        Gamma-corrected image
    """
    # Generate random gamma
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    
    # Normalize image to [0, 1] for gamma correction
    min_val, max_val = image.min(), image.max()
    normalized = (image - min_val) / (max_val - min_val)
    
    # Apply gamma correction
    corrected = np.power(normalized, gamma)
    
    # Scale back to original range
    corrected_image = corrected * (max_val - min_val) + min_val
    
    return corrected_image

def augment_ct_scan(ct_scan, mask=None, augmentation_types=None, p=0.5):
    """
    Apply multiple random augmentations to a CT scan and its mask.
    
    Args:
        ct_scan: 3D numpy array of the CT scan
        mask: 3D numpy array of the segmentation mask (optional)
        augmentation_types: List of augmentation types to apply (if None, all are considered)
        p: Probability of applying each augmentation
        
    Returns:
        Augmented CT scan and mask (if provided)
    """
    # Define all available augmentations
    all_augmentations = [
        'rotation',
        'flip',
        'zoom',
        'elastic',
        'noise',
        'intensity',
        'contrast',
        'gamma'
    ]
    
    # Use specified augmentations or all
    augmentation_types = augmentation_types or all_augmentations
    
    # Make copies to avoid modifying originals
    augmented_ct = ct_scan.copy()
    augmented_mask = mask.copy() if mask is not None else None
    
    # Apply random augmentations
    for aug_type in augmentation_types:
        if np.random.random() < p:
            if aug_type == 'rotation':
                if mask is not None:
                    augmented_ct, augmented_mask = random_rotation_3d(augmented_ct, augmented_mask)
                else:
                    augmented_ct = random_rotation_3d(augmented_ct)
            
            elif aug_type == 'flip':
                if mask is not None:
                    augmented_ct, augmented_mask = random_flip_3d(augmented_ct, augmented_mask)
                else:
                    augmented_ct = random_flip_3d(augmented_ct)
            
            elif aug_type == 'zoom':
                if mask is not None:
                    augmented_ct, augmented_mask = random_zoom_3d(augmented_ct, augmented_mask)
                else:
                    augmented_ct = random_zoom_3d(augmented_ct)
            
            elif aug_type == 'elastic':
                if mask is not None:
                    augmented_ct, augmented_mask = elastic_transform_3d(augmented_ct, augmented_mask)
                else:
                    augmented_ct = elastic_transform_3d(augmented_ct)
            
            elif aug_type == 'noise':
                augmented_ct = add_gaussian_noise(augmented_ct)
            
            elif aug_type == 'intensity':
                augmented_ct = random_intensity_shift(augmented_ct)
            
            elif aug_type == 'contrast':
                augmented_ct = random_contrast_adjustment(augmented_ct)
            
            elif aug_type == 'gamma':
                augmented_ct = random_gamma_correction(augmented_ct)
    
    if mask is not None:
        return augmented_ct, augmented_mask
    
    return augmented_ct

def create_balanced_dataset(data_dir, output_dir, labels_csv=None, target_counts=None, augmentation_types=None):
    """
    Create a balanced dataset by augmenting underrepresented classes.
    
    Args:
        data_dir: Directory containing preprocessed CT scans and masks
        output_dir: Directory to save augmented data
        labels_csv: Path to CSV file with class labels
        target_counts: Dictionary mapping class labels to target counts
        augmentation_types: List of augmentation types to apply
        
    Returns:
        DataFrame with information about the balanced dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load labels if provided
    if labels_csv is not None:
        labels_df = pd.read_csv(labels_csv)
    else:
        # Try to infer labels from directory structure
        patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        labels_df = pd.DataFrame({
            'patient_id': patient_dirs,
            'class': ['unknown'] * len(patient_dirs)
        })
    
    # Count samples per class
    class_counts = labels_df['class'].value_counts().to_dict()
    print(f"Original class distribution: {class_counts}")
    
    # Determine target counts if not provided
    if target_counts is None:
        max_count = max(class_counts.values())
        target_counts = {cls: max_count for cls in class_counts.keys()}
    
    # Calculate how many augmented samples to generate for each class
    augmentation_counts = {
        cls: max(0, target_counts[cls] - count) 
        for cls, count in class_counts.items()
    }
    
    print(f"Augmentation counts: {augmentation_counts}")
    
    # Create a DataFrame to track augmented samples
    augmented_df = pd.DataFrame(columns=['patient_id', 'original_patient_id', 'class', 'augmentation_types'])
    
    # Process each class
    for cls, count in augmentation_counts.items():
        if count <= 0:
            continue
        
        # Get patients of this class
        class_patients = labels_df[labels_df['class'] == cls]['patient_id'].tolist()
        
        if not class_patients:
            print(f"Warning: No patients found for class {cls}")
            continue
        
        # Generate augmented samples
        augmented_count = 0
        while augmented_count < count:
            # Select a random patient
            patient_id = np.random.choice(class_patients)
            
            # Load CT scan and mask
            ct_path = os.path.join(data_dir, patient_id, 'ct_scan_preprocessed.npy')
            mask_path = os.path.join(data_dir, patient_id, 'mask_preprocessed.npy')
            
            if not os.path.exists(ct_path) or not os.path.exists(mask_path):
                print(f"Warning: Missing data for patient {patient_id}")
                continue
            
            ct_scan = np.load(ct_path)
            mask = np.load(mask_path)
            
            # Select random augmentation types
            if augmentation_types is None:
                num_augs = np.random.randint(1, 4)  # Apply 1-3 random augmentations
                selected_augs = np.random.choice([
                    'rotation', 'flip', 'zoom', 'elastic', 
                    'noise', 'intensity', 'contrast', 'gamma'
                ], size=num_augs, replace=False).tolist()
            else:
                selected_augs = augmentation_types
            
            # Apply augmentations
            augmented_ct, augmented_mask = augment_ct_scan(
                ct_scan, mask, augmentation_types=selected_augs, p=1.0
            )
            
            # Generate new patient ID
            new_patient_id = f"{patient_id}_aug_{augmented_count}"
            
            # Save augmented data
            patient_output_dir = os.path.join(output_dir, new_patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)
            
            np.save(os.path.join(patient_output_dir, 'ct_scan_preprocessed.npy'), augmented_ct)
            np.save(os.path.join(patient_output_dir, 'mask_preprocessed.npy'), augmented_mask)
            
            # Add to tracking DataFrame
            augmented_df = pd.concat([
                augmented_df,
                pd.DataFrame({
                    'patient_id': [new_patient_id],
                    'original_patient_id': [patient_id],
                    'class': [cls],
                    'augmentation_types': [','.join(selected_augs)]
                })
            ], ignore_index=True)
            
            augmented_count += 1
            
            if augmented_count % 10 == 0:
                print(f"Generated {augmented_count}/{count} augmented samples for class {cls}")
    
    # Save augmentation info
    augmented_df.to_csv(os.path.join(output_dir, 'augmentation_info.csv'), index=False)
    
    # Create combined DataFrame with original and augmented samples
    combined_df = pd.concat([
        labels_df,
        augmented_df[['patient_id', 'class']]
    ], ignore_index=True)
    
    # Save combined info
    combined_df.to_csv(os.path.join(output_dir, 'dataset_info.csv'), index=False)
    
    # Print final class distribution
    final_class_counts = combined_df['class'].value_counts().to_dict()
    print(f"Final class distribution: {final_class_counts}")
    
    return combined_df

def visualize_augmentations(ct_scan, mask=None, slice_idx=None):
    """
    Visualize different augmentations applied to a CT scan slice.
    
    Args:
        ct_scan: 3D numpy array of the CT scan
        mask: 3D numpy array of the segmentation mask (optional)
        slice_idx: Slice index to visualize (default: middle slice)
    """
    if slice_idx is None:
        slice_idx = ct_scan.shape[0] // 2
    
    # Define augmentations to visualize
    augmentations = [
        ('Original', lambda x, m: (x, m)),
        ('Rotation', lambda x, m: random_rotation_3d(x, m, max_angle=15)),
        ('Flip', lambda x, m: random_flip_3d(x, m, axis=1)),
        ('Zoom', lambda x, m: random_zoom_3d(x, m, zoom_range=(0.9, 1.1))),
        ('Elastic', lambda x, m: elastic_transform_3d(x, m, alpha=15, sigma=3)),
        ('Noise', lambda x, m: (add_gaussian_noise(x, std=0.05), m)),
        ('Intensity', lambda x, m: (random_intensity_shift(x, shift_range=(-0.1, 0.1)), m)),
        ('Contrast', lambda x, m: (random_contrast_adjustment(x, gain_range=(0.8, 1.2)), m)),
        ('Gamma', lambda x, m: (random_gamma_correction(x, gamma_range=(0.8, 1.2)), m))
    ]
    
    # Set up the figure
    n_rows = 3
    n_cols = 3
    plt.figure(figsize=(15, 15))
    
    # Apply and visualize each augmentation
    for i, (title, aug_func) in enumerate(augmentations):
        plt.subplot(n_rows, n_cols, i+1)
        
        if mask is not None:
            aug_ct, aug_mask = aug_func(ct_scan.copy(), mask.copy())
            
            # Show CT with mask overlay
            plt.imshow(aug_ct[slice_idx], cmap='gray')
            plt.imshow(aug_mask[slice_idx], alpha=0.3, cmap='hot')
        else:
            aug_ct = aug_func(ct_scan.copy(), None)[0]
            plt.imshow(aug_ct[slice_idx], cmap='gray')
        
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(data_dir, labels_csv=None):
    """
    Analyze the class distribution in a dataset.
    
    Args:
        data_dir: Directory containing CT scans and masks
        labels_csv: Path to CSV file with class labels
        
    Returns:
        DataFrame with class distribution information
    """
    # Load labels if provided
    if labels_csv is not None:
        labels_df = pd.read_csv(labels_csv)
    else:
        # Try to infer labels from directory structure
        patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        labels_df = pd.DataFrame({
            'patient_id': patient_dirs,
            'class': ['unknown'] * len(patient_dirs)
        })
    
    # Count samples per class
    class_counts = labels_df['class'].value_counts().reset_index()
    class_counts.columns = ['class', 'count']
    
    # Calculate percentages
    total = class_counts['count'].sum()
    class_counts['percentage'] = class_counts['count'] / total * 100
    
    # Print summary
    print(f"Total samples: {total}")
    print("\nClass distribution:")
    for _, row in class_counts.iterrows():
        print(f"  {row['class']}: {row['count']} samples ({row['percentage']:.1f}%)")
    
    # Visualize distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts['class'], class_counts['count'])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return class_counts

def main():
    """
    Main function to demonstrate data augmentation capabilities.
    """
    # Example usage
    data_dir = "preprocessed_data"
    output_dir = "balanced_data"
    labels_csv = "patient_labels.csv"
    
    # Analyze original class distribution
    print("Analyzing original class distribution...")
    class_counts = analyze_class_distribution(data_dir, labels_csv)
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    balanced_df = create_balanced_dataset(data_dir, output_dir, labels_csv)
    
    # Analyze balanced class distribution
    print("\nAnalyzing balanced class distribution...")
    balanced_counts = analyze_class_distribution(output_dir, os.path.join(output_dir, 'dataset_info.csv'))
    
    print("\nData augmentation complete!")

if __name__ == "__main__":
    main()
