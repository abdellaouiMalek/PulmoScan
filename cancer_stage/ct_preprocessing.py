import numpy as np
import scipy.ndimage as ndimage
from skimage import exposure
import matplotlib.pyplot as plt
import pydicom
import os
from tqdm import tqdm

def load_dicom_series_safely(directory):
    """
    Load a series of DICOM files from a directory, checking for PixelData existence.

    Args:
        directory: Path to directory containing DICOM files

    Returns:
        3D NumPy array of pixel data, list of valid DICOM datasets, and spacing
    """
    print(f"Loading DICOM series from {directory}...")

    # Find all DICOM files in the directory
    dicom_files = [os.path.join(directory, f) for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f)) and f.endswith('.dcm')]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    print(f"Found {len(dicom_files)} DICOM files")

    # Try to get slice position information for sorting
    first_slice = pydicom.dcmread(dicom_files[0])
    position_key = None

    for key in ['ImagePositionPatient', 'SliceLocation', 'InstanceNumber']:
        if hasattr(first_slice, key):
            position_key = key
            break

    # Sort files by slice position
    if position_key:
        print(f"Sorting slices by {position_key}...")
        if position_key == 'ImagePositionPatient':
            # For ImagePositionPatient, sort by the z-coordinate (usually the third value)
            dicom_files.sort(key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).ImagePositionPatient[2])
        else:
            # For other position keys, sort by the value directly
            dicom_files.sort(key=lambda x: getattr(pydicom.dcmread(x, stop_before_pixels=True), position_key))
    else:
        print("Warning: No slice position information found. Using file order.")
        # Sort files by name if no position information is available
        dicom_files.sort()

    # Read all DICOM files
    print("Reading DICOM slices...")
    dicom_datasets = []
    valid_dicom_datasets = []

    for f in tqdm(dicom_files):
        try:
            ds = pydicom.dcmread(f)
            dicom_datasets.append(ds)

            # Check if PixelData exists
            if hasattr(ds, 'PixelData') and ds.PixelData is not None:
                valid_dicom_datasets.append(ds)
            else:
                print(f"Warning: DICOM file {f} has no pixel data, skipping for stacking")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not valid_dicom_datasets:
        raise ValueError("No valid DICOM files with pixel data found")

    # Extract pixel data
    print("Stacking slices...")
    pixel_data = np.stack([ds.pixel_array for ds in valid_dicom_datasets])

    # Apply rescale slope and intercept if available
    if hasattr(valid_dicom_datasets[0], 'RescaleSlope') and hasattr(valid_dicom_datasets[0], 'RescaleIntercept'):
        slope = valid_dicom_datasets[0].RescaleSlope
        intercept = valid_dicom_datasets[0].RescaleIntercept
        pixel_data = pixel_data * slope + intercept
        print(f"Applied rescale: slope={slope}, intercept={intercept}")

    # Get spacing information
    spacing = None
    if hasattr(valid_dicom_datasets[0], 'PixelSpacing') and hasattr(valid_dicom_datasets[0], 'SliceThickness'):
        pixel_spacing = valid_dicom_datasets[0].PixelSpacing
        slice_thickness = valid_dicom_datasets[0].SliceThickness
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        print(f"Spacing (z, y, x): {spacing} mm")

    print(f"CT scan loaded with shape {pixel_data.shape}")

    return pixel_data, valid_dicom_datasets, spacing

def resample_volume(volume, original_spacing, target_spacing=[1.0, 1.0, 1.0]):
    """
    Resample a 3D volume to a new voxel spacing.

    Args:
        volume: 3D numpy array of the CT scan
        original_spacing: Original voxel spacing (z, y, x) in mm
        target_spacing: Target voxel spacing (z, y, x) in mm

    Returns:
        Resampled volume and new spacing
    """
    # Convert to numpy arrays
    original_spacing = np.array(original_spacing)
    target_spacing = np.array(target_spacing)

    # Calculate resize factor
    resize_factor = original_spacing / target_spacing

    # Calculate new shape
    new_shape = np.round(volume.shape * resize_factor).astype(int)

    # Calculate real resize factor
    real_resize = new_shape / volume.shape

    # Resample the volume using scipy's zoom
    resampled_volume = ndimage.zoom(volume, real_resize, order=3, mode='nearest')

    return resampled_volume, target_spacing

def normalize_hu_values(volume, min_hu=-1000, max_hu=400, output_range=(0, 1)):
    """
    Normalize Hounsfield Units (HU) to a specified range.

    Args:
        volume: 3D numpy array of the CT scan
        min_hu: Minimum HU value to consider
        max_hu: Maximum HU value to consider
        output_range: Output range for normalization (default: 0 to 1)

    Returns:
        Normalized volume
    """
    # Clip HU values
    clipped = np.clip(volume, min_hu, max_hu)

    # Normalize to [0, 1]
    normalized = (clipped - min_hu) / (max_hu - min_hu)

    # Scale to output range if different from [0, 1]
    if output_range != (0, 1):
        normalized = normalized * (output_range[1] - output_range[0]) + output_range[0]

    return normalized

def resize_volume(volume, target_shape):
    """
    Resize a 3D volume to a target shape.

    Args:
        volume: 3D numpy array of the CT scan
        target_shape: Target shape (depth, height, width)

    Returns:
        Resized volume
    """
    # Calculate resize factors
    factors = np.array(target_shape) / np.array(volume.shape)

    # Resize using scipy's zoom
    resized = ndimage.zoom(volume, factors, order=3, mode='nearest')

    return resized

def center_crop_or_pad(volume, target_shape, pad_value=0):
    """
    Center crop or pad a 3D volume to the target shape.

    Args:
        volume: 3D numpy array of the CT scan
        target_shape: Target shape (depth, height, width)
        pad_value: Value to use for padding

    Returns:
        Cropped/padded volume
    """
    # Check if target_shape is a boolean or other invalid type
    if isinstance(target_shape, bool):
        print("Warning: target_shape is a boolean. Using volume.shape as target_shape.")
        target_shape = volume.shape
    elif not isinstance(target_shape, (tuple, list, np.ndarray)):
        print(f"Warning: target_shape is not a tuple, list, or array. Using volume.shape as target_shape.")
        target_shape = volume.shape

    # Convert to numpy arrays
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)

    # Initialize output array
    result = np.ones(target_shape, dtype=volume.dtype) * pad_value

    # Calculate slices for cropping/padding
    starts = np.maximum(0, (current_shape - target_shape) // 2)
    ends = starts + np.minimum(current_shape, target_shape)

    target_starts = np.maximum(0, (target_shape - current_shape) // 2)
    target_ends = target_starts + np.minimum(current_shape, target_shape)

    # Perform cropping/padding
    result[
        target_starts[0]:target_ends[0],
        target_starts[1]:target_ends[1],
        target_starts[2]:target_ends[2]
    ] = volume[
        starts[0]:ends[0],
        starts[1]:ends[1],
        starts[2]:ends[2]
    ]

    return result

def enhance_contrast(volume, clip_limit=0.01):
    """
    Enhance contrast in a 3D volume using CLAHE.

    Args:
        volume: 3D numpy array of the CT scan
        clip_limit: Clip limit for CLAHE

    Returns:
        Contrast-enhanced volume
    """
    # Apply CLAHE to each slice
    enhanced = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        enhanced[i] = exposure.equalize_adapthist(volume[i], clip_limit=clip_limit)

    return enhanced

def apply_lung_window(volume, window_center=-600, window_width=1500):
    """
    Apply lung window to the CT scan.

    Args:
        volume: 3D numpy array of the CT scan
        window_center: Center of the window in HU
        window_width: Width of the window in HU

    Returns:
        Windowed volume
    """
    # Calculate window boundaries
    min_value = window_center - window_width/2
    max_value = window_center + window_width/2

    # Apply window
    windowed = np.clip(volume, min_value, max_value)

    # Normalize to [0, 1]
    windowed = (windowed - min_value) / (max_value - min_value)

    return windowed

def preprocess_ct_scan(ct_scan, spacing=None, target_spacing=[1.0, 1.0, 1.0],
                       target_shape=(128, 256, 256), normalize=True, enhance=False,
                       apply_window=False, output_range=(0, 1)):
    """
    Comprehensive preprocessing for a CT scan.

    Args:
        ct_scan: 3D numpy array of the CT scan
        spacing: Original voxel spacing (z, y, x) in mm
        target_spacing: Target voxel spacing (z, y, x) in mm
        target_shape: Target shape (depth, height, width)
        normalize: Whether to normalize HU values
        enhance: Whether to enhance contrast
        apply_window: Whether to apply lung window
        output_range: Output range for normalization

    Returns:
        Preprocessed CT scan
    """
    # Make a copy to avoid modifying the original
    processed = ct_scan.copy()

    # Step 1: Resample to uniform voxel spacing if spacing is provided
    if spacing is not None:
        processed, _ = resample_volume(processed, spacing, target_spacing)
        print(f"After resampling: {processed.shape}")

    # Step 2: Normalize HU values
    if normalize:
        processed = normalize_hu_values(processed, output_range=output_range)
        print(f"After normalization: value range [{processed.min():.2f}, {processed.max():.2f}]")

    # Step 3: Apply lung window if requested
    if apply_window:
        processed = apply_lung_window(processed)
        print(f"After windowing: value range [{processed.min():.2f}, {processed.max():.2f}]")

    # Step 4: Enhance contrast if requested
    if enhance:
        processed = enhance_contrast(processed)
        print(f"After contrast enhancement: value range [{processed.min():.2f}, {processed.max():.2f}]")

    # Step 5: Resize to target shape
    if target_shape is not None:
        # Ensure target_shape is a tuple, not a boolean
        if isinstance(target_shape, bool):
            print("Warning: target_shape is a boolean. Using processed.shape as target_shape.")
            target_shape = tuple(processed.shape)
        elif not isinstance(target_shape, (tuple, list)):
            print(f"Warning: target_shape is not a tuple or list. Using processed.shape as target_shape.")
            target_shape = tuple(processed.shape)

        processed = center_crop_or_pad(processed, target_shape)
        print(f"After resizing: {processed.shape}")

    return processed

def preprocess_dicom_directory(dicom_dir, target_spacing=[1.0, 1.0, 1.0],
                              target_shape=(128, 256, 256), normalize=True,
                              enhance=False, apply_window=False, output_range=(0, 1)):
    """
    Load and preprocess a CT scan from a directory of DICOM files.

    Args:
        dicom_dir: Directory containing DICOM files
        target_spacing: Target voxel spacing (z, y, x) in mm
        target_shape: Target shape (depth, height, width)
        normalize: Whether to normalize HU values
        enhance: Whether to enhance contrast
        apply_window: Whether to apply lung window
        output_range: Output range for normalization

    Returns:
        Preprocessed CT scan, original CT scan, and spacing
    """
    # Load DICOM series safely (checking for PixelData)
    ct_scan, dicom_datasets, spacing = load_dicom_series_safely(dicom_dir)

    # Preprocess the CT scan
    preprocessed_ct = preprocess_ct_scan(
        ct_scan=ct_scan,
        spacing=spacing,
        target_spacing=target_spacing,
        target_shape=target_shape,
        normalize=normalize,
        enhance=enhance,
        apply_window=apply_window,
        output_range=output_range
    )

    return preprocessed_ct, ct_scan, spacing

def visualize_preprocessing(original, preprocessed, slice_idx=None):
    """
    Visualize original and preprocessed CT scans.

    Args:
        original: Original 3D CT scan
        preprocessed: Preprocessed 3D CT scan
        slice_idx: Slice index to visualize (default: middle slice)
    """
    if slice_idx is None:
        slice_idx = original.shape[0] // 2

    # Ensure slice_idx is within bounds for both volumes
    slice_idx_orig = min(slice_idx, original.shape[0] - 1)
    slice_idx_proc = min(slice_idx, preprocessed.shape[0] - 1)

    plt.figure(figsize=(12, 6))

    # Original slice
    plt.subplot(1, 2, 1)
    plt.imshow(original[slice_idx_orig], cmap='gray')
    plt.title(f"Original (Slice {slice_idx_orig})")
    plt.axis('off')

    # Preprocessed slice
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed[slice_idx_proc], cmap='gray')
    plt.title(f"Preprocessed (Slice {slice_idx_proc})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
