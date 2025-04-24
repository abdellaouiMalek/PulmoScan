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

# Load dataset as a dictionary
def load_dataset_structure(root_dir):

    dataset_structure = defaultdict(lambda: defaultdict(list))

    for patient in sorted(os.listdir(root_dir)):
        patient_path = os.path.join(root_dir, patient)

        if os.path.isdir(patient_path):
            for study in sorted(os.listdir(patient_path)):
                study_path = os.path.join(patient_path, study)

                if os.path.isdir(study_path):
                    scan_folders = [f for f in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, f))]
                    dataset_structure[patient][study] = scan_folders

    return dataset_structure

# Load csv files
def load_csv_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Extract the slice properties
def extract_image_properties(dicom_path):

    dicom_data = pydicom.dcmread(dicom_path)

    properties = {
        "Image Shape": dicom_data.pixel_array.shape,
        "Min Intensity": dicom_data.pixel_array.min(),
        "Max Intensity": dicom_data.pixel_array.max(),
        "Intensity Mean": dicom_data.pixel_array.mean(),
        "Pixel Spacing": dicom_data.PixelSpacing if hasattr(dicom_data, "PixelSpacing") else None,
        "Slice Thickness": dicom_data.SliceThickness if hasattr(dicom_data, "SliceThickness") else None
    }
    return properties

# Visualize one slice
def visualize_dicom_image(dicom_path):

    dicom_data = pydicom.dcmread(dicom_path)
    plt.imshow(dicom_data.pixel_array, cmap='gray')
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()

# Visualize the class distribution of lung cancer stages
def plot_class_balance(class_balance):
    class_balance.plot(kind='bar', title="Cancer Stage Distribution")
    plt.xlabel("Cancer Stage")
    plt.ylabel("Frequency")
    plt.show()

def analyze_correlation(clinical_data):

    stage_mapping = {
        'I': 1, 'Ia': 1, 'Ib': 1.5,
        'II': 2, 'IIa': 2, 'IIb': 2.5,
        'III': 3, 'IIIa': 3, 'IIIb': 3.5,
        'IV': 4
    }

    clinical_data['Overall.Stage'] = clinical_data['Overall.Stage'].map(stage_mapping)

    relevant_features = ['Overall.Stage', 'Survival.time', 'age']

    correlation_matrix = clinical_data[relevant_features].corr(method='spearman')

    # Create a figure with a 2x2 grid layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cancer Stage Analysis", fontsize=16, fontweight='bold')

    # Boxplot of survival time across cancer stages
    sns.boxplot(ax=axes[0, 0], x=clinical_data['Overall.Stage'], y=clinical_data['Survival.time'])
    axes[0, 0].set_title("Survival Time Across Cancer Stages")
    axes[0, 0].set_xlabel("Cancer Stage")
    axes[0, 0].set_ylabel("Survival Time")

    # Scatter plot of cancer stage vs survival time
    sns.scatterplot(ax=axes[0, 1], x=clinical_data['Overall.Stage'], y=clinical_data['Survival.time'], alpha=0.6)
    axes[0, 1].set_title("Scatter Plot: Cancer Stage vs. Survival Time")
    axes[0, 1].set_xlabel("Cancer Stage")
    axes[0, 1].set_ylabel("Survival Time")
    axes[0, 1].grid(True)

    # Heatmap of correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1, 0])
    axes[1, 0].set_title("Correlation Matrix")

    # Pair Plot
    sns.kdeplot(ax=axes[1, 1], x=clinical_data['Overall.Stage'], y=clinical_data['Survival.time'], cmap="Blues", fill=True)
    axes[1, 1].set_title("Density Plot: Cancer Stage vs. Survival Time")
    axes[1, 1].set_xlabel("Cancer Stage")
    axes[1, 1].set_ylabel("Survival Time")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def load_ct_scan_3d(scan_dir):
    slices = [pydicom.dcmread(os.path.join(scan_dir, f)) for f in sorted(os.listdir(scan_dir))]

    # Sort slices by slice location (for correct order)
    slices = sorted(slices, key=lambda x: float(x.SliceLocation))

    # Stack the slices into a 3D volume (scan_3d)
    scan_3d = np.stack([s.pixel_array for s in slices], axis=0)

    # Get voxel spacing from the metadata
    voxel_spacing = (slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[0].SliceThickness)

    # Get the origin (from the first DICOM file)
    origin = np.array(slices[0].ImagePositionPatient) if 'ImagePositionPatient' in slices[0] else np.zeros(3)

    return scan_3d, voxel_spacing, origin

# Function to Display CT Scan Slice
def display_ct_scan(scan_3d):
    # Get the middle slice index
    middle_slice_idx = scan_3d.shape[0] // 2

    # Display the middle slice
    plt.figure(figsize=(8, 8))
    plt.imshow(scan_3d[middle_slice_idx], cmap="gray")
    plt.title(f"Middle Slice of CT Scan ({middle_slice_idx})")
    plt.axis("off")  # Turn off axis
    plt.show()

def load_segmentation_mask(seg_dcm_path):
    """Loads a 3D segmentation mask from a DICOM SEG file."""
    seg_dcm = pydicom.dcmread(seg_dcm_path)

    if "PixelData" not in seg_dcm:
        raise ValueError("Segmentation DICOM does not contain pixel data.")

    # Extract segmentation mask (already 3D)
    mask_array = seg_dcm.pixel_array

    return mask_array

def visualize_image(volumeImage):
    # Create figure and adjust layout to make room for slider
    fig, (ax, ax_xray) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(right=0.8)

    # Initialize slice (middle slice)
    initial_slice = volumeImage.shape[0] // 2
    current_slice = initial_slice

    # Display initial slice (CT scan) on the left side
    im = ax.imshow(volumeImage[initial_slice], cmap="gray")
    ax.set_title(f"CT Scan Slice {initial_slice}")
    ax.axis("off")

    # Load your chest X-ray image
    try:
        xray_image_path = "../Assets/chest.jpg"  # Replace with your actual chest X-ray image path
        xray_image = mpimg.imread(xray_image_path)
        height, width = xray_image.shape[:2]
        slices_nbr = volumeImage.shape[0] - 1
        line_slices = np.linspace(0, height, slices_nbr + 1)

        # Set up X-ray image on the right side
        xray_im = ax_xray.imshow(xray_image, cmap="gray")
        ax_xray.set_title("Chest X-Ray Reference")
        ax_xray.axis("off")

        # Add a horizontal line on the X-ray image indicating the current slice
        line = ax_xray.axhline(y=line_slices[current_slice], color="red", linestyle="--", linewidth=2)
    except Exception as e:
        print(f"Warning: Could not load X-ray reference image: {e}")
        # If X-ray image can't be loaded, display CT info instead
        ax_xray.text(0.5, 0.5, f"CT Scan Info:\nTotal Slices: {volumeImage.shape[0]}\nSlice Size: {volumeImage.shape[1]}x{volumeImage.shape[2]}",
                    ha='center', va='center', fontsize=12)
        ax_xray.set_title("CT Scan Information")
        ax_xray.axis("off")
        line = None

    # Create vertical slider on the right side
    ax_slider = plt.axes([0.85, 0.26, 0.03, 0.47])  # [left, bottom, width, height]
    slice_slider = Slider(
        ax=ax_slider,
        label="Slice Number",
        valmin=0,  # Start from 0 (first slice)
        valmax=volumeImage.shape[0] - 1,
        valinit=initial_slice,
        valstep=1,
        orientation="vertical"
    )

    # Add slice number indicator
    ax_text = plt.figtext(0.85, 0.15, f"Slice: {initial_slice}/{volumeImage.shape[0]-1}",
                         ha="center", fontsize=10)

    # Update function for slider
    def update(val):
        current_slice = int(slice_slider.val)
        im.set_data(volumeImage[current_slice])  # Update CT slice
        ax.set_title(f"CT Scan Slice {current_slice}")
        ax_text.set_text(f"Slice: {current_slice}/{volumeImage.shape[0]-1}")

        # Update the line on the X-ray image if it exists
        if line is not None:
            line.set_ydata(line_slices[current_slice])

        fig.canvas.draw_idle()  # Refresh plot

    # Register update function with slider
    slice_slider.on_changed(update)

    # Show the plot
    plt.show()


def visualize_image_interactive(volumeImage, window_center=50, window_width=350):
    """Interactive visualization of CT scan with windowing controls"""
    # Create figure and adjust layout to make room for sliders
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9)

    # Initialize slice (middle slice)
    initial_slice = volumeImage.shape[0] // 2

    # Apply initial windowing
    def apply_window(image, center, width):
        img_min = center - width // 2
        img_max = center + width // 2
        windowed_image = np.clip(image, img_min, img_max)
        windowed_image = ((windowed_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return windowed_image

    # Display initial slice with windowing
    windowed_slice = apply_window(volumeImage[initial_slice], window_center, window_width)
    im = ax.imshow(windowed_slice, cmap="gray")
    ax.set_title(f"CT Scan Slice {initial_slice}")
    ax.axis("off")

    # Create slice slider
    ax_slice = plt.axes([0.1, 0.1, 0.8, 0.03])
    slice_slider = Slider(
        ax=ax_slice,
        label="Slice",
        valmin=0,
        valmax=volumeImage.shape[0] - 1,
        valinit=initial_slice,
        valstep=1
    )

    # Create window center slider
    ax_center = plt.axes([0.1, 0.05, 0.8, 0.03])
    center_slider = Slider(
        ax=ax_center,
        label="Window Center",
        valmin=-1000,
        valmax=1000,
        valinit=window_center
    )

    # Create window width slider
    ax_width = plt.axes([0.1, 0.15, 0.8, 0.03])
    width_slider = Slider(
        ax=ax_width,
        label="Window Width",
        valmin=1,
        valmax=2000,
        valinit=window_width
    )

    # Update function for sliders
    def update(val):
        current_slice = int(slice_slider.val)
        current_center = center_slider.val
        current_width = width_slider.val

        # Apply windowing to the current slice
        windowed_slice = apply_window(volumeImage[current_slice], current_center, current_width)

        # Update the image
        im.set_data(windowed_slice)
        ax.set_title(f"CT Scan Slice {current_slice}")
        fig.canvas.draw_idle()

    # Register update function with sliders
    slice_slider.on_changed(update)
    center_slider.on_changed(update)
    width_slider.on_changed(update)

    # Show the plot
    plt.show()