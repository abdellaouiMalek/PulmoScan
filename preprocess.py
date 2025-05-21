import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
from scipy import ndimage
from scipy.stats import gaussian_kde
from skimage import morphology
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def read_luna_dataset(root_dir):
    Dataset={}
    subsets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(subsets)

    for subset in subsets:
        mhd_files = [f for f in os.listdir(os.path.join(root_dir, subset)) if f.endswith(".mhd")]
        if subset not in Dataset:
            Dataset[subset] = {}

        for index, file in enumerate(mhd_files):
            Dataset[subset][f"CT_{index}"] = file

    return Dataset

def load_csv(filename):
    return pd.read_csv(f"Data/Luna/{filename}")

def convert_world_to_voxel_coord(seriesuid, numpyOrigin, numpySpacing, annotations):
    """
    Convert world coordinates to voxel coordinates for a specific seriesuid.
    
    Parameters:
        seriesuid (str): The seriesuid to filter annotations.
        numpyOrigin (np.array): Origin of the CT scan in world coordinates.
        numpySpacing (np.array): Spacing of the CT scan voxels.
        annotations (pd.DataFrame or pd.Series): DataFrame or Series containing annotations with world coordinates.
    
    Returns:
        pd.DataFrame: Filtered annotations with voxel coordinates.
    """
    # Check if annotations is a Series and convert it to a DataFrame if necessary
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T  # Convert Series to DataFrame

    # Ensure annotations is a DataFrame
    if not isinstance(annotations, pd.DataFrame):
        raise ValueError("annotations must be a pandas DataFrame or Series.")
    # Filter the annotations for the specific seriesuid
    mask = annotations["seriesuid"] == seriesuid
    filtered_annotations = annotations[mask].copy()  # Create a copy to avoid modifying the original
        
    # Convert world coordinates to voxel coordinates
    filtered_annotations.loc[:, "coordZ"] = np.round((filtered_annotations.loc[:, "coordZ"] - numpyOrigin[0]) / numpySpacing[0]).astype(int)
    filtered_annotations.loc[:, "coordY"] = np.round((filtered_annotations.loc[:, "coordY"] - numpyOrigin[1]) / numpySpacing[1]).astype(int)
    filtered_annotations.loc[:, "coordX"] = np.round((filtered_annotations.loc[:, "coordX"] - numpyOrigin[2]) / numpySpacing[2]).astype(int)
    
    return filtered_annotations

def convert_world_to_voxel_coord_bbox(seriesuid, numpyOrigin, numpySpacing, annotations):
    """
    Convert world coordinates (in mm) to voxel coordinates.

    Parameters:
        seriesuid (str): Series UID for accessing annotations, can be None now.
        numpyOrigin (np.ndarray): The origin (z, y, x) of the scan in world space.
        numpySpacing (np.ndarray): Voxel spacing (z, y, x) in world space.
        annotations (pd.DataFrame or pd.Series): The annotations to process.

    Returns:
        pd.DataFrame or pd.Series: Updated annotations with voxel coordinates.
    """
    if seriesuid is not None:
        # If seriesuid is provided, you might fetch or filter annotations based on seriesuid
        annotations = annotations[annotations['seriesuid'] == seriesuid]

    # Ensure DataFrame format
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T

    # Convert each annotation's world coordinates to voxel coordinates
    for index, row in annotations.iterrows():
        # Convert the center coordinates of the nodule
        coordZ_voxel = (row['coordZ'] - numpyOrigin[0]) / numpySpacing[0]
        coordY_voxel = (row['coordY'] - numpyOrigin[1]) / numpySpacing[1]
        coordX_voxel = (row['coordX'] - numpyOrigin[2]) / numpySpacing[2]

        # Convert bounding box coordinates
        bbox_z_voxel = (row['bbox_z'] - numpyOrigin[0]) / numpySpacing[0]
        bbox_y_voxel = (row['bbox_y'] - numpyOrigin[1]) / numpySpacing[1]
        bbox_x_voxel = (row['bbox_x'] - numpyOrigin[2]) / numpySpacing[2]

        # Convert diameter to voxel space (separately for z, y, x)
        diameter_mm = row['diameter_mm']
        diameter_voxel_z = diameter_mm / numpySpacing[0]
        diameter_voxel_y = diameter_mm / numpySpacing[1]
        diameter_voxel_x = diameter_mm / numpySpacing[2]

        # Update the annotation with voxel coordinates
        annotations.at[index, 'coordZ'] = coordZ_voxel
        annotations.at[index, 'coordY'] = coordY_voxel
        annotations.at[index, 'coordX'] = coordX_voxel
        annotations.at[index, 'bbox_z'] = bbox_z_voxel
        annotations.at[index, 'bbox_y'] = bbox_y_voxel
        annotations.at[index, 'bbox_x'] = bbox_x_voxel

        # Also add diameter in voxel space
        annotations.at[index, 'diameter_voxel_z'] = diameter_voxel_z
        annotations.at[index, 'diameter_voxel_y'] = diameter_voxel_y
        annotations.at[index, 'diameter_voxel_x'] = diameter_voxel_x

    return annotations


def resize_image_with_annotation(seriesuid, numpyImage, numpyOrigin, numpySpacing, annotations):
    # Step 1: Filter annotations (but do not convert to voxel yet!)
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T
    mask = annotations["seriesuid"] == seriesuid
    filtered_annotations = annotations[mask].copy()

    # Step 2: Store world coordinates separately
    world_coords = filtered_annotations[["coordX", "coordY", "coordZ"]].copy()

    # Step 3: Resample the image
    RESIZE_SPACING = np.array([1, 1, 1])
    resize_factor = numpySpacing / RESIZE_SPACING
    new_real_shape = numpyImage.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / numpyImage.shape
    new_volume = ndimage.zoom(numpyImage, zoom=real_resize)

    # Step 4: Compute newOrigin
    newOrigin = numpyOrigin  # in this case, origin stays same in world space when resampling

    # Step 5: Recalculate voxel coordinates from world coordinates
    filtered_annotations.loc[:, "coordX"] = ((world_coords["coordX"] - newOrigin[2]) / RESIZE_SPACING[2]).round().astype(int)
    filtered_annotations.loc[:, "coordY"] = ((world_coords["coordY"] - newOrigin[1]) / RESIZE_SPACING[1]).round().astype(int)
    filtered_annotations.loc[:, "coordZ"] = ((world_coords["coordZ"] - newOrigin[0]) / RESIZE_SPACING[0]).round().astype(int)

    return new_volume, newOrigin, RESIZE_SPACING, filtered_annotations

def resize_image_with_annotation_bbox(seriesuid, numpyImage, numpyOrigin, numpySpacing, annotations):
    """
    Resize a 3D CT scan to 1mm voxel spacing and update corresponding annotations.

    Parameters:
        seriesuid (str): The seriesuid of the scan.
        numpyImage (np.ndarray): 3D volume (z, y, x).
        numpyOrigin (np.ndarray): Original origin (z, y, x).
        numpySpacing (np.ndarray): Original voxel spacing (z, y, x).
        annotations (pd.DataFrame or pd.Series): Annotations containing nodule center and bounding box.

    Returns:
        tuple: (resized volume, new origin, new spacing, updated annotations)
    """
    # Step 1: Convert world coordinates to voxel coordinates first
    newAnnotations = convert_world_to_voxel_coord_bbox(seriesuid, numpyOrigin, numpySpacing, annotations)

    # Step 2: Define target spacing
    RESIZE_SPACING = np.array([1, 1, 1])

    # Step 3: Compute resize factors
    resize_factor = numpySpacing / RESIZE_SPACING
    new_real_shape = numpyImage.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    real_resize = new_shape / numpyImage.shape

    # Step 4: Resize the image
    new_volume = ndimage.zoom(numpyImage, zoom=real_resize, order=1)  # linear interpolation (order=1)

    # Step 5: Adjust the origin properly (origin stays the same in world space)
    newOrigin = numpyOrigin  # No change needed to the origin after resampling spacing!

    # Step 6: Update the annotations (rescale voxel coordinates)
    scale = np.array([resize_factor[0], resize_factor[1], resize_factor[2]])

    for coord in ["coordZ", "bbox_z"]:
        newAnnotations[coord] = (newAnnotations[coord] * scale[0]).round().astype(int)

    for coord in ["coordY", "bbox_y"]:
        newAnnotations[coord] = (newAnnotations[coord] * scale[1]).round().astype(int)

    for coord in ["coordX", "bbox_x"]:
        newAnnotations[coord] = (newAnnotations[coord] * scale[2]).round().astype(int)

    return new_volume, newOrigin, RESIZE_SPACING, newAnnotations


def resize_image_with_given_annotations(numpyImage, numpyOrigin, numpySpacing, annotations):
    """
    Resize a 3D CT scan to 1mm voxel spacing and update corresponding annotations.
    The annotations are provided directly as input.

    Parameters:
        numpyImage (np.ndarray): 3D volume (z, y, x).
        numpyOrigin (np.ndarray): Original origin (z, y, x).
        numpySpacing (np.ndarray): Original voxel spacing (z, y, x).
        annotations (pd.DataFrame or pd.Series): Annotations containing nodule center and bounding box.

    Returns:
        tuple: (resized volume, new origin, new spacing, updated annotations)
    """
    # Check if annotations is a DataFrame or Series
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T  # Convert Series to DataFrame for consistency
    
    # Step 1: Convert world coordinates to voxel coordinates first
    newAnnotations = convert_world_to_voxel_coord_bbox(None, numpyOrigin, numpySpacing, annotations)

    # Step 2: Define target spacing
    RESIZE_SPACING = np.array([1, 1, 1])

    # Step 3: Compute resize factors
    resize_factor = numpySpacing / RESIZE_SPACING
    new_real_shape = numpyImage.shape * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    real_resize = new_shape / numpyImage.shape

    # Step 4: Resize the image
    new_volume = ndimage.zoom(numpyImage, zoom=real_resize, order=1)  # linear interpolation (order=1)

    # Step 5: Adjust the origin properly (origin stays the same in world space)
    newOrigin = numpyOrigin  # No change needed to the origin after resampling spacing!

    # Step 6: Update the annotations (rescale voxel coordinates)
    scale = np.array([resize_factor[0], resize_factor[1], resize_factor[2]])

    for coord in ["coordZ", "bbox_z"]:
        newAnnotations[coord] = (newAnnotations[coord] * scale[0]).round().astype(int)

    for coord in ["coordY", "bbox_y"]:
        newAnnotations[coord] = (newAnnotations[coord] * scale[1]).round().astype(int)

    for coord in ["coordX", "bbox_x"]:
        newAnnotations[coord] = (newAnnotations[coord] * scale[2]).round().astype(int)

    return new_volume, newOrigin, RESIZE_SPACING, newAnnotations


def threshHold_segmentation(numpyImage, lower=-1000, upper=-600):
    """
    Segment the lung region using thresholding.
    Args:
        numpyImage: Input image in Hounsfield Units (HU).
        lower: Lower Hounsfield value
        upp: Upper Hounsfield value
    Returns:
        Binary mask of the slected region.
    """
    mask = np.logical_and(numpyImage >= lower, numpyImage <= upper)

    return mask

def clip_CT_scan(numpyImage):
    return np.clip(numpyImage, -1200, 600)

def get_lung_mask(numpyImage):
    # Get the lung region
    lung_mask = threshHold_segmentation(numpyImage)

    # Post process the lung mask (remove holes/smooth boundaries)
    lung_mask = morphology.binary_closing(lung_mask, morphology.ball(5))
    lung_mask = morphology.binary_opening(lung_mask, morphology.ball(5))
    lung_mask = morphology.binary_dilation(lung_mask, morphology.ball(2))

    return lung_mask

def refine_nodule_masks(
        numpyImage, 
        lung_mask, 
        nodule_ranges= [
            (-1200, -600),  # Lung tissue
            (-600, -300),  # Subsolid nodules (Ground-Glass Opacities)
            (-100, 100),   # Solid nodules
            (300, 600)     # Calcified nodules
        ]
    ):
    refined_masks = []

    for lower, upper in nodule_ranges:
        nodule_mask = threshHold_segmentation(numpyImage, lower=lower, upper=upper)
        # Restrict to lung region
        refined_mask = np.logical_and(nodule_mask, lung_mask)
        refined_masks.append(refined_mask.astype(np.uint8))
    
    return refined_masks

def isolate_lung(numpyImage):
    lung_mask = get_lung_mask(numpyImage)
    refined_masks = refine_nodule_masks(numpyImage, lung_mask)

    combined_mask = np.zeros_like(refined_masks[0])
    for mask in refined_masks:
        combined_mask = np.logical_or(combined_mask, mask)

    return combined_mask


def Min_Max_scaling(numpyImage):
    normalized_image = (numpyImage - (-1000)) / (400 - (-1000))

    return normalized_image

def crop_image(volume, crop_shape):
    """
    Crop a 3D volume to the desired crop_shape from the center.
    
    Parameters:
        volume (np.ndarray): 3D numpy array to crop.
        crop_shape (tuple): Desired output shape (depth, height, width).
        
    Returns:
        np.ndarray: Cropped volume.
    """
    z, y, x = volume.shape
    cz, cy, cx = crop_shape

    start_z = (z - cz) // 2
    start_y = (y - cy) // 2
    start_x = (x - cx) // 2

    end_z = start_z + cz
    end_y = start_y + cy
    end_x = start_x + cx

    return volume[start_z:end_z, start_y:end_y, start_x:end_x]

def get_metadata(file_path):
    """
    Get shape, spacing, and origin of a medical image (e.g., .mhd, .nii, .nrrd).
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        tuple: (shape, spacing, origin)
            - shape: (Depth, Height, Width) in voxels.
            - spacing: (Depth_spacing, Height_spacing, Width_spacing) in mm.
            - origin: (Depth_origin, Height_origin, Width_origin) in mm (world coordinates).
    """
    img = sitk.ReadImage(file_path)  # Metadata-only (fast)
    
    # Get metadata (reversed to match ZYX order)
    shape = list(reversed(img.GetSize()))      # (Depth, Height, Width)
    spacing = list(reversed(img.GetSpacing())) # (Depth_spacing, Height_spacing, Width_spacing)
    origin = list(reversed(img.GetOrigin()))   # (Depth_origin, Height_origin, Width_origin)
    
    return shape, spacing, origin

def calculate_new_shape(numpyImage_shape, numpySpacing, target_spacing=[1.0, 1.0, 1.0]):
    """
    Args:
        numpyImage_shape: Shape of the input image (Z, Y, X).
        numpySpacing: Current spacing (Z_sp, Y_sp, X_sp).
        target_spacing: Desired spacing (default: [1.0, 1.0, 1.0] mm).
    
    Returns:
        new_shape: Resampled shape (Z, Y, X) after accounting for spacing change.
        resize_factor: Scaling factor for each axis (Z, Y, X).
    """
    # Ensure inputs are numpy arrays
    numpySpacing = np.array(numpySpacing)
    target_spacing = np.array(target_spacing)
    
    # Calculate resize factor (per-axis scaling)
    resize_factor = numpySpacing / target_spacing  # (Z_scale, Y_scale, X_scale)
    
    # Compute new shape (rounded to nearest integer)
    new_real_shape = np.array(numpyImage_shape) * resize_factor
    new_shape = np.round(new_real_shape).astype(int)
    
    return new_shape

def load_paths(filename):
    data = pd.read_csv(filename)

    paths = {}

    for index, row in data.iterrows():
        paths[row["Filename"]] = row["File Path"]

    return paths
    
    
def add_voxel_coord(annotations_subset, origin, spacing):
    """Convert real-world coordinates to voxel coordinates for a subset of annotations."""
    subset = annotations_subset.copy()
    subset["voxel_Z"] = np.round((subset["coordZ"] - origin[0]) / spacing[0]).astype(int)
    subset["voxel_Y"] = np.round((subset["coordY"] - origin[1]) / spacing[1]).astype(int)
    subset["voxel_X"] = np.round((subset["coordX"] - origin[2]) / spacing[2]).astype(int)
    subset["voxel_bbox_z"] = np.round((subset["bbox_z"] - origin[0]) / spacing[0]).astype(int)
    subset["voxel_bbox_y"] = np.round((subset["bbox_y"] - origin[1]) / spacing[1]).astype(int)
    subset["voxel_bbox_x"] = np.round((subset["bbox_x"] - origin[2]) / spacing[2]).astype(int)
    return subset

def process_all_annotations(annotations_df, seriesuid_to_path):
    """
    Process all annotations in the DataFrame to add voxel coordinates.
    
    Args:
        annotations_df (pd.DataFrame): DataFrame with columns: seriesuid, coordX, coordY, coordZ.
        seriesuid_to_path (dict): Maps seriesuid to CT scan file paths.
        
    Returns:
        pd.DataFrame: Annotations with added voxel coordinates (voxel_X, voxel_Y, voxel_Z).
    """
    results = []
    for seriesuid, group in annotations_df.groupby("seriesuid"):
        if seriesuid in seriesuid_to_path:
            file_path = seriesuid_to_path[seriesuid]
            shape, spacing, origin = get_metadata(file_path)
            subset_with_voxel = add_voxel_coord(group, origin, spacing)
            results.append(subset_with_voxel)
        else:
            print(f"Warning: seriesuid {seriesuid} not found in path dictionary. Skipping.")
    
    return pd.concat(results, ignore_index=True)

def extract_patch(ct_scan, center, patch_size):
        """Extracts a 3D patch centered around `center` with handling for out-of-bounds."""
        x, y, z = center
        ph, pw, pd = np.array(patch_size) // 2

        # Calculate slice indices with boundary checks
        x_start = max(0, x - ph)
        x_end = min(ct_scan.shape[0], x + ph + (1 if patch_size[0] % 2 else 0))
        y_start = max(0, y - pw)
        y_end = min(ct_scan.shape[1], y + pw + (1 if patch_size[1] % 2 else 0))
        z_start = max(0, z - pd)
        z_end = min(ct_scan.shape[2], z + pd + (1 if patch_size[2] % 2 else 0))

        # Extract the patch
        patch = ct_scan[z_start:z_end, x_start:x_end, y_start:y_end]

        # Pad the patch if it's smaller than the target size
        return patch
    
def sliding_window_crop(ct_scan, center, patch_size):
    offsets = [
        (-10, -10, -10),
        (+10, +10, +10),
        (-10, 0, 0),
        (+10, 0, 0),
        (0, -10, 0),
        (0, +10, 0),
        (0, 0, -10),
        (0, 0, +10),
    ]
    
    patches = []
    new_relative_centers = []

    for offset in offsets:
        patch_center = (
            center[0] + offset[0],
            center[1] + offset[1],
            center[2] + offset[2]
        )
        patch = extract_patch(ct_scan, patch_center, patch_size)
        patches.append(patch)
        
        # Calculate new relative coord inside the patch
        patch_origin = (
            patch_center[0] - patch_size[0] // 2,
            patch_center[1] - patch_size[1] // 2,
            patch_center[2] - patch_size[2] // 2
        )
        
        relative_coord = (
            center[0] - patch_origin[0],
            center[1] - patch_origin[1],
            center[2] - patch_origin[2]
        )
        
        new_relative_centers.append(relative_coord)

    return patches, new_relative_centers


def sliding_window_crop_bbox(ct_scan, bbox, patch_size):
    z, y, x, height, width, depth = bbox
    bbox_center = (z + height // 2, y + width // 2, x + depth // 2)
    
    offsets = [
        (-10, -10, -10),
        (+10, +10, +10),
        (-10, 0, 0),
        (+10, 0, 0),
        (0, -10, 0),
        (0, +10, 0),
        (0, 0, -10),
        (0, 0, +10),
    ]
    
    patches = []
    new_relative_bboxes = []

    for offset in offsets:
        patch_center = (
            bbox_center[0] + offset[0],
            bbox_center[1] + offset[1],
            bbox_center[2] + offset[2]
        )
        patch = extract_patch(ct_scan, patch_center, patch_size)
        patches.append(patch)
        
        # Patch origin
        patch_origin = (
            patch_center[0] - patch_size[0] // 2,
            patch_center[1] - patch_size[1] // 2,
            patch_center[2] - patch_size[2] // 2
        )
        
        # New bbox relative to patch
        relative_bbox = (
            z - patch_origin[0],    # New z
            y - patch_origin[1],    # New y
            x - patch_origin[2],    # New x
            height,
            width,
            depth
        )
        
        new_relative_bboxes.append(relative_bbox)

    return patches, new_relative_bboxes

def flip_patch_3d(patch, relative_center):
    """
    Flips a 3D patch around the x, y, and z axes and updates the relative center coordinates.

    Args:
        patch (numpy.ndarray): A 3D patch of shape (depth, height, width).
        relative_center (tuple): The (z, y, x) coordinates relative to the patch.

    Returns:
        list: A list of tuples [(flipped_patch, new_relative_center), ...] for each flip.
    """
    d, h, w = patch.shape
    z, y, x = relative_center

    flips = []

    # Flip along the x-axis (depth)
    flip_x = np.flip(patch, axis=0)
    new_z = d - z - 1
    flips.append((flip_x, (new_z, y, x)))

    # Flip along the y-axis (height)
    flip_y = np.flip(patch, axis=1)
    new_y = h - y - 1
    flips.append((flip_y, (z, new_y, x)))

    # Flip along the z-axis (width)
    flip_z = np.flip(patch, axis=2)
    new_x = w - x - 1
    flips.append((flip_z, (z, y, new_x)))

    return flips


def flip_patch_3d_bbox(patch, bbox):
    """
    Flips a 3D patch around the x, y, and z axes and updates the bounding box coordinates.

    Args:
        patch (numpy.ndarray): A 3D patch of shape (depth, height, width).
        bbox (tuple): The bbox (z, y, x, height, width, depth) relative to the patch.

    Returns:
        list: A list of tuples [(flipped_patch, new_bbox), ...] for each flip.
    """
    d, h, w = patch.shape
    z, y, x, height, width, depth = bbox

    flips = []

    # Flip along the x-axis (depth)
    flip_x = np.flip(patch, axis=0)
    new_z = d - z - height
    flips.append((flip_x, (new_z, y, x, height, width, depth)))

    # Flip along the y-axis (height)
    flip_y = np.flip(patch, axis=1)
    new_y = h - y - width
    flips.append((flip_y, (z, new_y, x, height, width, depth)))

    # Flip along the z-axis (width)
    flip_z = np.flip(patch, axis=2)
    new_x = w - x - depth
    flips.append((flip_z, (z, y, new_x, height, width, depth)))

    return flips


def rotate_patch_3d(patch, relative_center):
    """
    Rotates a 3D patch by 90°, 180°, and 270° and updates the relative center coordinates.

    Args:
        patch (numpy.ndarray): A 3D patch of shape (depth, height, width).
        relative_center (tuple): The (z, y, x) coordinates relative to the patch.

    Returns:
        list: A list of tuples [(rotated_patch, new_relative_center), ...] for each rotation.
    """
    d, h, w = patch.shape
    z, y, x = relative_center

    rotations = []

    # Rotate 90° counterclockwise
    rotate_90 = np.rot90(patch, k=1, axes=(1, 2))
    new_center_90 = (z, w - x - 1, y)
    rotations.append((rotate_90, new_center_90))

    # Rotate 180°
    rotate_180 = np.rot90(patch, k=2, axes=(1, 2))
    new_center_180 = (z, h - y - 1, w - x - 1)
    rotations.append((rotate_180, new_center_180))

    # Rotate 270° counterclockwise (or 90° clockwise)
    rotate_270 = np.rot90(patch, k=3, axes=(1, 2))
    new_center_270 = (z, x, h - y - 1)
    rotations.append((rotate_270, new_center_270))

    return rotations


def rotate_patch_3d_bbox(patch, bbox):
    """
    Rotates a 3D patch by 90°, 180°, and 270° and updates the bounding box coordinates.

    Args:
        patch (numpy.ndarray): A 3D patch of shape (depth, height, width).
        bbox (tuple): The bbox (z, y, x, height, width, depth) relative to the patch.

    Returns:
        list: A list of tuples [(rotated_patch, new_bbox), ...] for each rotation.
    """
    d, h, w = patch.shape
    z, y, x, height, width, depth = bbox

    rotations = []

    # Rotate 90° counterclockwise
    rotate_90 = np.rot90(patch, k=1, axes=(1, 2))
    new_y_90 = x
    new_x_90 = w - (y + height)
    rotations.append((rotate_90, (z, new_y_90, new_x_90, width, height, depth)))

    # Rotate 180°
    rotate_180 = np.rot90(patch, k=2, axes=(1, 2))
    new_y_180 = h - (y + height)
    new_x_180 = w - (x + width)
    rotations.append((rotate_180, (z, new_y_180, new_x_180, height, width, depth)))

    # Rotate 270° counterclockwise
    rotate_270 = np.rot90(patch, k=3, axes=(1, 2))
    new_y_270 = h - (x + width)
    new_x_270 = y
    rotations.append((rotate_270, (z, new_y_270, new_x_270, width, height, depth)))

    return rotations
