import os
import logging
import numpy as np
import pydicom
import SimpleITK as sitk
from django.conf import settings
import random
import pandas as pd
from scipy import ndimage
from skimage import morphology

# Configure logging
logger = logging.getLogger(__name__)

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def convert_world_to_voxel_coord_bbox(seriesuid, numpyOrigin, numpySpacing, annotations):
    """
    Convert world coordinates (in mm) to voxel coordinates.

    Parameters:
        seriesuid (str): Series UID for accessing annotations, can be None now.
        numpyOrigin (np.ndarray): The origin (z, y, x) of the scan in world space.
        numpySpacing (np.ndarray): Voxel spacing (z, y, x) in world space.
        annotations (pd.DataFrame or pd.Series): The annotations to process.

    Returns:
        pd.DataFrame: Updated annotations with voxel coordinates.
    """
    # Create a copy to avoid SettingWithCopyWarning
    annotations = annotations.copy()
    
    if seriesuid is not None:
        # Use .loc to ensure we're working with a proper DataFrame
        annotations = annotations.loc[annotations['seriesuid'] == seriesuid].copy()

    # Ensure DataFrame format
    if isinstance(annotations, pd.Series):
        annotations = annotations.to_frame().T

    # Convert coordinates
    annotations['coordZ'] = (annotations['coordZ'] - numpyOrigin[0]) / numpySpacing[0]
    annotations['coordY'] = (annotations['coordY'] - numpyOrigin[1]) / numpySpacing[1]
    annotations['coordX'] = (annotations['coordX'] - numpyOrigin[2]) / numpySpacing[2]

    # Convert bounding box coordinates
    annotations['bbox_z'] = (annotations['bbox_z'] - numpyOrigin[0]) / numpySpacing[0]
    annotations['bbox_y'] = (annotations['bbox_y'] - numpyOrigin[1]) / numpySpacing[1]
    annotations['bbox_x'] = (annotations['bbox_x'] - numpyOrigin[2]) / numpySpacing[2]

    # Convert diameter to voxel space
    annotations['diameter_voxel_z'] = annotations['diameter_mm'] / numpySpacing[0]
    annotations['diameter_voxel_y'] = annotations['diameter_mm'] / numpySpacing[1]
    annotations['diameter_voxel_x'] = annotations['diameter_mm'] / numpySpacing[2]

    return annotations

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
    newAnnotations = convert_world_to_voxel_coord_bbox(seriesuid, numpyOrigin, numpySpacing, annotations).copy()

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

    # Create a copy of the columns we'll modify
    for coord in ["coordZ", "bbox_z"]:
        newAnnotations.loc[:, coord] = (newAnnotations[coord] * scale[0]).round().astype(int)

    for coord in ["coordY", "bbox_y"]:
        newAnnotations.loc[:, coord] = (newAnnotations[coord] * scale[1]).round().astype(int)

    for coord in ["coordX", "bbox_x"]:
        newAnnotations.loc[:, coord] = (newAnnotations[coord] * scale[2]).round().astype(int)

    return new_volume, newOrigin, RESIZE_SPACING, newAnnotations

def clip_CT_scan(numpyImage):
    return np.clip(numpyImage, -1200, 600)

def threshHold_segmentation(numpyImage, lower=-1200, upper=-600):
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
    normalized_image = (numpyImage - (-1200)) / (600 - (-1000))

    return normalized_image

def segment_nodules(ct_scan):
    # Combine masks for all nodule types
    mask = np.zeros_like(ct_scan, dtype=bool)

    ranges = [
        (-600, -300),  # subsolid
        (-100, 100),   # solid
        (300, 600)     # calcified
    ]

    for low, high in ranges:
        mask |= (ct_scan >= low) & (ct_scan <= high)

    return mask.astype(np.uint8)
