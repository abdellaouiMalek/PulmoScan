"""
CT Scan Data Augmentation

This module provides functions for augmenting 3D CT scan data to improve model training.
It includes various augmentation techniques specifically designed for medical imaging.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
import random

def random_rotation_3d(volume, max_angle=10):
    """
    Apply random rotation to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        max_angle: Maximum rotation angle in degrees

    Returns:
        Rotated volume
    """
    # Generate random angles for each axis
    angles = [random.uniform(-max_angle, max_angle) for _ in range(3)]

    # Apply rotation
    rotated = ndimage.rotate(volume, angles[0], axes=(1, 2), reshape=False, order=1, mode='nearest')
    rotated = ndimage.rotate(rotated, angles[1], axes=(0, 2), reshape=False, order=1, mode='nearest')
    rotated = ndimage.rotate(rotated, angles[2], axes=(0, 1), reshape=False, order=1, mode='nearest')

    return rotated

def random_shift_3d(volume, max_shift=10):
    """
    Apply random shift to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        max_shift: Maximum shift in pixels

    Returns:
        Shifted volume
    """
    # Generate random shifts for each axis
    shifts = [random.randint(-max_shift, max_shift) for _ in range(3)]

    # Apply shift
    shifted = ndimage.shift(volume, shifts, order=1, mode='nearest')

    return shifted

def random_flip_3d(volume):
    """
    Apply random flipping to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan

    Returns:
        Flipped volume
    """
    # Randomly decide which axes to flip
    flip_axes = [random.choice([True, False]) for _ in range(3)]

    # Apply flips
    flipped = volume.copy()
    if flip_axes[0]:
        flipped = flipped[::-1, :, :]
    if flip_axes[1]:
        flipped = flipped[:, ::-1, :]
    if flip_axes[2]:
        flipped = flipped[:, :, ::-1]

    return flipped

def random_zoom_3d(volume, zoom_range=(0.9, 1.1)):
    """
    Apply random zoom to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        zoom_range: Range of zoom factors

    Returns:
        Zoomed volume
    """
    # Generate random zoom factors for each axis
    zoom_factors = [random.uniform(zoom_range[0], zoom_range[1]) for _ in range(3)]

    # Apply zoom
    zoomed = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')

    # Resize back to original shape if needed
    if zoomed.shape != volume.shape:
        # Create interpolator
        orig_shape = volume.shape
        x = np.linspace(0, zoomed.shape[0] - 1, zoomed.shape[0])
        y = np.linspace(0, zoomed.shape[1] - 1, zoomed.shape[1])
        z = np.linspace(0, zoomed.shape[2] - 1, zoomed.shape[2])
        interpolator = RegularGridInterpolator((x, y, z), zoomed, method='linear', bounds_error=False, fill_value=0)

        # Create new coordinates
        x_new = np.linspace(0, zoomed.shape[0] - 1, orig_shape[0])
        y_new = np.linspace(0, zoomed.shape[1] - 1, orig_shape[1])
        z_new = np.linspace(0, zoomed.shape[2] - 1, orig_shape[2])

        # Create meshgrid
        x_new_mesh, y_new_mesh, z_new_mesh = np.meshgrid(x_new, y_new, z_new, indexing='ij')
        points = np.vstack([x_new_mesh.ravel(), y_new_mesh.ravel(), z_new_mesh.ravel()]).T

        # Interpolate
        zoomed = interpolator(points).reshape(orig_shape)

    return zoomed

def random_noise_3d(volume, noise_factor=0.05):
    """
    Add random noise to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        noise_factor: Factor controlling the amount of noise

    Returns:
        Volume with added noise
    """
    # Generate random noise
    noise = np.random.normal(0, noise_factor, volume.shape)

    # Add noise to volume
    noisy = volume + noise

    # Clip to original range
    min_val, max_val = volume.min(), volume.max()
    noisy = np.clip(noisy, min_val, max_val)

    return noisy

def random_gamma_3d(volume, gamma_range=(0.8, 1.2)):
    """
    Apply random gamma correction to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        gamma_range: Range of gamma values

    Returns:
        Gamma-corrected volume
    """
    # Ensure volume is positive
    min_val = volume.min()
    shifted = volume - min_val

    # Normalize to [0, 1] for gamma correction
    max_val = shifted.max()
    if max_val > 0:
        normalized = shifted / max_val
    else:
        return volume

    # Apply gamma correction
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    gamma_corrected = np.power(normalized, gamma)

    # Scale back to original range
    gamma_corrected = gamma_corrected * max_val + min_val

    return gamma_corrected

def random_contrast_3d(volume, contrast_range=(0.8, 1.2), brightness_range=(-0.1, 0.1)):
    """
    Apply random contrast and brightness adjustment to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        contrast_range: Range of contrast factors
        brightness_range: Range of brightness adjustments

    Returns:
        Contrast-adjusted volume
    """
    # Generate random contrast and brightness factors
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    brightness = random.uniform(brightness_range[0], brightness_range[1])

    # Apply contrast and brightness adjustment
    mean = volume.mean()
    adjusted = (volume - mean) * contrast + mean + brightness

    # Clip to original range
    min_val, max_val = volume.min(), volume.max()
    adjusted = np.clip(adjusted, min_val, max_val)

    return adjusted

def elastic_deformation_3d(volume, alpha=15, sigma=3):
    """
    Apply elastic deformation to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        alpha: Scaling factor for deformation
        sigma: Smoothing factor for deformation

    Returns:
        Deformed volume
    """
    shape = volume.shape

    # Generate random displacement fields
    dx = np.random.rand(*shape) * 2 - 1
    dy = np.random.rand(*shape) * 2 - 1
    dz = np.random.rand(*shape) * 2 - 1

    # Smooth displacement fields
    dx = ndimage.gaussian_filter(dx, sigma) * alpha
    dy = ndimage.gaussian_filter(dy, sigma) * alpha
    dz = ndimage.gaussian_filter(dz, sigma) * alpha

    # Create meshgrid with correct dimensions
    z, y, x = np.meshgrid(np.arange(shape[2]), np.arange(shape[1]), np.arange(shape[0]), indexing='ij')

    # Transpose to match volume dimensions (depth, height, width)
    x = x.transpose(2, 1, 0)
    y = y.transpose(2, 1, 0)
    z = z.transpose(2, 1, 0)

    # Create coordinates for each point in the volume
    coords = np.zeros(shape + (3,))
    coords[..., 0] = x + dx
    coords[..., 1] = y + dy
    coords[..., 2] = z + dz

    # Clip coordinates to valid range
    for i in range(3):
        coords[..., i] = np.clip(coords[..., i], 0, shape[i] - 1)

    # Apply deformation using map_coordinates
    # We need to reshape the coordinates for map_coordinates
    coords_reshape = coords.reshape(-1, 3)
    indices = coords_reshape[:, 0], coords_reshape[:, 1], coords_reshape[:, 2]

    # Apply deformation
    deformed = ndimage.map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)

    return deformed

def random_crop_3d(volume, crop_size):
    """
    Apply random cropping to a 3D volume.

    Args:
        volume: 3D numpy array of the CT scan
        crop_size: Size of the crop (depth, height, width)

    Returns:
        Cropped volume
    """
    # Ensure crop_size is valid
    if any(c > s for c, s in zip(crop_size, volume.shape)):
        raise ValueError("Crop size must be smaller than or equal to volume size")

    # Calculate maximum valid starting positions
    max_starts = [s - c for s, c in zip(volume.shape, crop_size)]

    # Generate random starting positions
    starts = [random.randint(0, m) for m in max_starts]

    # Extract crop
    cropped = volume[
        starts[0]:starts[0] + crop_size[0],
        starts[1]:starts[1] + crop_size[1],
        starts[2]:starts[2] + crop_size[2]
    ]

    return cropped

def augment_ct_scan(volume, augmentation_types=None, p=0.5):
    """
    Apply a series of random augmentations to a 3D CT scan.

    Args:
        volume: 3D numpy array of the CT scan
        augmentation_types: List of augmentation types to apply (default: all except elastic)
        p: Probability of applying each augmentation

    Returns:
        Augmented volume
    """
    # Define available augmentation types
    all_augmentation_types = [
        'rotation', 'shift', 'flip', 'zoom', 'noise',
        'gamma', 'contrast', 'crop'  # 'elastic' removed temporarily due to potential issues
    ]

    # Use specified augmentation types or all types
    if augmentation_types is None:
        augmentation_types = all_augmentation_types

    # Make a copy of the volume
    augmented = volume.copy()

    # Apply augmentations with probability p
    for aug_type in augmentation_types:
        if random.random() < p:
            if aug_type == 'rotation':
                augmented = random_rotation_3d(augmented)
            elif aug_type == 'shift':
                augmented = random_shift_3d(augmented)
            elif aug_type == 'flip':
                augmented = random_flip_3d(augmented)
            elif aug_type == 'zoom':
                augmented = random_zoom_3d(augmented)
            elif aug_type == 'noise':
                augmented = random_noise_3d(augmented)
            elif aug_type == 'gamma':
                augmented = random_gamma_3d(augmented)
            elif aug_type == 'contrast':
                augmented = random_contrast_3d(augmented)
            elif aug_type == 'elastic':
                augmented = elastic_deformation_3d(augmented)
            elif aug_type == 'crop' and hasattr(volume, 'shape'):
                # Only apply crop if we have a valid crop size
                crop_size = [max(1, int(s * 0.8)) for s in volume.shape]
                if all(c > 0 for c in crop_size):
                    augmented = random_crop_3d(augmented, crop_size)
                    # Resize back to original shape
                    augmented = ndimage.zoom(augmented,
                                           [o/c for o, c in zip(volume.shape, augmented.shape)],
                                           order=1, mode='nearest')

    return augmented

def create_augmented_batch(volume, batch_size=8, augmentation_types=None):
    """
    Create a batch of augmented volumes from a single volume.

    Args:
        volume: 3D numpy array of the CT scan
        batch_size: Number of augmented volumes to create
        augmentation_types: List of augmentation types to apply (default: all)

    Returns:
        Batch of augmented volumes [batch_size, depth, height, width]
    """
    # Create batch
    batch = np.zeros((batch_size,) + volume.shape, dtype=volume.dtype)

    # First sample is the original
    batch[0] = volume

    # Create augmented samples
    for i in range(1, batch_size):
        batch[i] = augment_ct_scan(volume, augmentation_types)

    return batch

def visualize_augmentations(volume, n_augmentations=5, slice_idx=None):
    """
    Visualize original volume and multiple augmentations.

    Args:
        volume: 3D numpy array of the CT scan
        n_augmentations: Number of augmentations to visualize
        slice_idx: Slice index to visualize (default: middle slice)

    Returns:
        None (displays the visualization)
    """
    import matplotlib.pyplot as plt

    # Select slice to visualize
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    # Create augmentations
    augmentations = [volume]
    for _ in range(n_augmentations):
        augmentations.append(augment_ct_scan(volume))

    # Create figure
    fig, axes = plt.subplots(1, n_augmentations + 1, figsize=(3 * (n_augmentations + 1), 4))

    # Display original
    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Display augmentations
    for i in range(n_augmentations):
        axes[i + 1].imshow(augmentations[i + 1][slice_idx], cmap='gray')
        axes[i + 1].set_title(f'Augmentation {i + 1}')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
    
def visualize_augmentation_effects(ct_volume, mask=None, n_augmentations=3, slice_idx=None):
    """
    Visualize the effects of different augmentations on a CT scan and optionally its mask.
    
    Args:
        ct_volume: 3D numpy array of the CT scan
        mask: Optional 3D numpy array of the segmentation mask
        n_augmentations: Number of different augmentations to show
        slice_idx: Slice index to visualize (default: middle slice)
    """
    import matplotlib.pyplot as plt
    
    # Select slice to visualize
    if slice_idx is None:
        slice_idx = ct_volume.shape[0] // 2
    
    # Define augmentation types to demonstrate
    aug_types = [
        ['rotation'],
        ['flip'],
        ['zoom'],
        ['noise'],
        ['contrast'],
        ['gamma'],
        ['rotation', 'flip', 'zoom'],  # Combined spatial augmentations
        ['noise', 'contrast', 'gamma']  # Combined intensity augmentations
    ]
    
    # Select a subset of augmentation types
    selected_aug_types = aug_types[:n_augmentations]
    
    # Create figure
    n_cols = n_augmentations + 1  # Original + augmentations
    fig_width = 4 * n_cols
    
    if mask is not None:
        n_rows = 2  # CT and mask
        fig_height = 8
    else:
        n_rows = 1  # CT only
        fig_height = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # If only one row, wrap axes in a list to make indexing consistent
    if n_rows == 1:
        axes = [axes]
    
    # Display original CT
    axes[0][0].imshow(ct_volume[slice_idx], cmap='gray')
    axes[0][0].set_title('Original CT')
    axes[0][0].axis('off')
    
    # Display original mask if provided
    if mask is not None:
        axes[1][0].imshow(mask[slice_idx], cmap='hot')
        axes[1][0].set_title('Original Mask')
        axes[1][0].axis('off')
    
    # Generate and display augmentations
    for i, aug_type in enumerate(selected_aug_types):
        # Set random seed for reproducibility
        np.random.seed(42 + i)
        
        # Apply augmentation to CT
        aug_ct = augment_ct_scan(ct_volume, aug_type)
        
        # Display augmented CT
        axes[0][i+1].imshow(aug_ct[slice_idx], cmap='gray')
        axes[0][i+1].set_title(f'{" + ".join(aug_type)}\nCT')
        axes[0][i+1].axis('off')
        
        # Apply same augmentation to mask if provided (only for spatial augmentations)
        if mask is not None:
            # For masks, only use spatial transformations
            mask_aug_types = [t for t in aug_type if t in ['rotation', 'shift', 'flip', 'zoom']]
            
            # Reset random seed to get identical spatial transformations
            np.random.seed(42 + i)
            
            # Apply augmentation to mask
            if mask_aug_types:
                aug_mask = augment_ct_scan(mask.astype(float), mask_aug_types)
                # Ensure mask remains binary
                aug_mask = (aug_mask > 0.5).astype(np.uint8)
            else:
                # If no spatial augmentations, use original mask
                aug_mask = mask
            
            # Display augmented mask
            axes[1][i+1].imshow(aug_mask[slice_idx], cmap='hot')
            axes[1][i+1].set_title(f'{" + ".join(mask_aug_types) if mask_aug_types else "No spatial aug"}\nMask')
            axes[1][i+1].axis('off')
    
    # Reset random seed
    np.random.seed(None)
    
    plt.tight_layout()
    plt.show()