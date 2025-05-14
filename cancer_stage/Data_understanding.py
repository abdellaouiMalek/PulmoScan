import os
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from collections import defaultdict
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
import seaborn as sns
from skimage.color import label2rgb
from tqdm import tqdm

def find_patient_data_paths(base_dir, patient_id):
    """
    Find paths to CT scan directory and segmentation mask file for a patient

    Args:
        base_dir: Base directory containing patient data
        patient_id: Patient ID

    Returns:
        Tuple of (ct_dir, mask_path) or (None, None) if not found
    """
    # Check if patient directory exists
    patient_dir = os.path.join(base_dir, patient_id)
    if not os.path.exists(patient_dir):
        print(f"Patient directory not found: {patient_dir}")
        return None, None

    # Find study directory
    study_dirs = [d for d in os.listdir(patient_dir)
                 if os.path.isdir(os.path.join(patient_dir, d)) and 'StudyID' in d]

    if not study_dirs:
        print(f"No study directory found for patient {patient_id}")
        return None, None

    study_dir = study_dirs[0]
    study_path = os.path.join(patient_dir, study_dir)

    # Find CT scan directory
    ct_subdirs = [d for d in os.listdir(study_path)
                 if os.path.isdir(os.path.join(study_path, d)) and d.startswith('1.000000')]

    if not ct_subdirs:
        print(f"No CT scan directory found for patient {patient_id}")
        return None, None

    ct_subdir = ct_subdirs[0]
    ct_dir = os.path.join(study_path, ct_subdir)

    # Find mask directory and file
    mask_subdirs = [d for d in os.listdir(study_path)
                   if os.path.isdir(os.path.join(study_path, d)) and d.startswith('300.000000')]

    if not mask_subdirs:
        print(f"No mask directory found for patient {patient_id}")
        return ct_dir, None

    mask_subdir = mask_subdirs[0]
    mask_dir = os.path.join(study_path, mask_subdir)

    # Find the actual mask file
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.dcm')]
    if not mask_files:
        print(f"No mask file found for patient {patient_id}")
        return ct_dir, None

    mask_path = os.path.join(mask_dir, mask_files[0])

    return ct_dir, mask_path

# Load csv files
def load_csv_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_ct_scan_from_slices(ct_dir):
    """
    Load a CT scan from a folder of 2D DICOM slices

    Args:
        ct_dir: Path to directory containing DICOM slices

    Returns:
        3D NumPy array of CT scan, list of DICOM datasets, and metadata
    """
    print(f"Loading CT scan from {ct_dir}...")

    # Find all DICOM files in the directory
    dicom_files = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir)
                  if os.path.isfile(os.path.join(ct_dir, f)) and f.endswith('.dcm')]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {ct_dir}")

    print(f"Found {len(dicom_files)} DICOM slices")

    # Read the first slice to get metadata
    first_slice = pydicom.dcmread(dicom_files[0])

    # Extract metadata
    metadata = {
        'PatientID': first_slice.PatientID if hasattr(first_slice, 'PatientID') else 'Unknown',
        'PatientName': str(first_slice.PatientName) if hasattr(first_slice, 'PatientName') else 'Unknown',
        'StudyDate': first_slice.StudyDate if hasattr(first_slice, 'StudyDate') else 'Unknown',
        'Modality': first_slice.Modality if hasattr(first_slice, 'Modality') else 'Unknown',
    }

    # Try to get slice position information
    position_key = None
    for key in ['ImagePositionPatient', 'SliceLocation', 'InstanceNumber']:
        if hasattr(first_slice, key):
            position_key = key
            break

    if position_key is None:
        print("Warning: No slice position information found. Using file order.")
        # Sort files by name if no position information is available
        dicom_files.sort()
    else:
        # Sort files by slice position
        if position_key == 'ImagePositionPatient':
            # For ImagePositionPatient, sort by the z-coordinate (usually the third value)
            dicom_files.sort(key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).ImagePositionPatient[2])
        else:
            # For other position keys, sort by the value directly
            dicom_files.sort(key=lambda x: getattr(pydicom.dcmread(x, stop_before_pixels=True), position_key))

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
    if hasattr(dicom_datasets[0], 'RescaleSlope') and hasattr(dicom_datasets[0], 'RescaleIntercept'):
        slope = dicom_datasets[0].RescaleSlope
        intercept = dicom_datasets[0].RescaleIntercept
        pixel_data = pixel_data * slope + intercept
        print(f"Applied rescale: slope={slope}, intercept={intercept}")

    # Get spacing information
    metadata['Spacing'] = None
    if hasattr(dicom_datasets[0], 'PixelSpacing') and hasattr(dicom_datasets[0], 'SliceThickness'):
        pixel_spacing = dicom_datasets[0].PixelSpacing
        slice_thickness = dicom_datasets[0].SliceThickness
        metadata['Spacing'] = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        print(f"Spacing (z, y, x): {metadata['Spacing']} mm")

    print(f"CT scan loaded with shape {pixel_data.shape}")

    return pixel_data, dicom_datasets, metadata

def load_and_stack_ct_slices(ct_dir):
    """
    Load DICOM CT scan slices from a directory and stack them into a 3D volume

    Args:
        ct_dir: Directory containing DICOM slices

    Returns:
        3D NumPy array of stacked slices, list of DICOM datasets
    """
    print(f"Loading CT scan slices from {ct_dir}...")

    # Find all DICOM files in the directory
    dicom_files = [os.path.join(ct_dir, f) for f in os.listdir(ct_dir)
                  if os.path.isfile(os.path.join(ct_dir, f)) and f.endswith('.dcm')]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {ct_dir}")

    print(f"Found {len(dicom_files)} DICOM slices")

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
    if hasattr(dicom_datasets[0], 'RescaleSlope') and hasattr(dicom_datasets[0], 'RescaleIntercept'):
        slope = dicom_datasets[0].RescaleSlope
        intercept = dicom_datasets[0].RescaleIntercept
        pixel_data = pixel_data * slope + intercept
        print(f"Applied rescale: slope={slope}, intercept={intercept}")

    # Get spacing information
    spacing = None
    if hasattr(dicom_datasets[0], 'PixelSpacing') and hasattr(dicom_datasets[0], 'SliceThickness'):
        pixel_spacing = dicom_datasets[0].PixelSpacing
        slice_thickness = dicom_datasets[0].SliceThickness
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        print(f"Spacing (z, y, x): {spacing} mm")

    print(f"CT scan loaded with shape {pixel_data.shape}")

    return pixel_data, dicom_datasets, spacing