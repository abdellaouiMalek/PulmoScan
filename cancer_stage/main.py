"""
PulmoScan - Cancer Stage Classification Main Module

This is the main entry point for the cancer stage classification part of the PulmoScan project.
It integrates the 3D ResNet model training and evaluation functionality.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import random
import pydicom
from torchvision import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import modules from the train_3d_resnet.py file
from train_3d_resnet import (
    set_seed, BasicBlock3D, ResNet3D, ResNet18_3D, CTScanDataset,
    train_model, evaluate_model, plot_training_history, plot_confusion_matrix
)

# Import direct dataset
from direct_dataset import DirectCTScanDataset

# Import preprocessing modules
from ct_preprocessing import (
    preprocess_ct_scan, visualize_preprocessing,
    resample_volume, normalize_hu_values, resize_volume,
    center_crop_or_pad, enhance_contrast, apply_lung_window,
    load_dicom_series_safely, preprocess_dicom_directory
)

# Import data augmentation modules
from ct_augmentation import (
    augment_ct_scan, visualize_augmentation_effects, create_augmented_batch, visualize_augmentations,
    random_rotation_3d, random_shift_3d, random_flip_3d, random_zoom_3d,
    random_noise_3d, random_gamma_3d, random_contrast_3d, elastic_deformation_3d
)

# Import data understanding module
from Data_understanding import (
    load_csv_labels, load_ct_scan_from_slices, load_and_stack_ct_slices, find_patient_data_paths
)

def preprocess_and_save_data(base_dir, output_dir, patient_ids=None,
                      target_spacing=[1.0, 1.0, 1.0], target_shape=(128, 256, 256),
                      augmentation_count=5, augmentation_types=None):
    """
    Preprocess CT scans and masks and save them to disk

    Args:
        base_dir: Base directory containing patient data
        output_dir: Directory to save preprocessed data
        patient_ids: List of patient IDs to process (if None, process all)
        target_spacing: Target voxel spacing in mm
        target_shape: Target volume shape
        augmentation_count: Number of augmented copies to create per patient
        augmentation_types: List of augmentation types to apply
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # If no patient IDs provided, process all patients in the base directory
    if patient_ids is None:
        patient_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('LUNG')]

    print(f"Found {len(patient_ids)} patients to process")

    # Process each patient
    for patient_id in patient_ids:
        try:
            print(f"\nProcessing patient {patient_id}...")

            # Find paths
            ct_dir, mask_path = find_patient_data_paths(base_dir, patient_id)

            if ct_dir is None or mask_path is None:
                print(f"Skipping patient {patient_id}: Missing CT or mask")
                continue

            # Load CT scan using our safe loading function
            try:
                # Use the safe loading function that checks for PixelData
                ct_scan, valid_dicom_datasets, spacing = load_dicom_series_safely(ct_dir)
                original_spacing = spacing
            except Exception as e:
                print(f"Error loading CT scan for patient {patient_id}: {e}")
                continue

            # Load and align mask
            try:
                mask = pydicom.dcmread(mask_path)
                # Check if mask has pixel data
                if not hasattr(mask, 'PixelData') or mask.PixelData is None:
                    print(f"Skipping patient {patient_id}: Mask has no pixel data")
                    continue

                mask_data = mask.pixel_array

                # Get Z positions from valid DICOM datasets
                ct_z_positions = [float(ds.ImagePositionPatient[2]) for ds in valid_dicom_datasets if hasattr(ds, 'ImagePositionPatient')]

                mask_z_positions = []
                for item in mask.PerFrameFunctionalGroupsSequence:
                    if hasattr(item, 'PlanePositionSequence') and hasattr(item.PlanePositionSequence[0], 'ImagePositionPatient'):
                        z = float(item.PlanePositionSequence[0].ImagePositionPatient[2])
                        mask_z_positions.append(z)

                if not mask_z_positions or not ct_z_positions:
                    print(f"Skipping patient {patient_id}: Missing position information")
                    continue

                # Align mask to CT
                aligned_mask_slices = []
                for ct_z in ct_z_positions:
                    mask_idx = np.argmin(np.abs(np.array(mask_z_positions) - ct_z))
                    aligned_mask_slices.append(mask_data[mask_idx])

                aligned_mask = np.stack(aligned_mask_slices)
            except Exception as e:
                print(f"Error processing mask for patient {patient_id}: {e}")
                continue

            # Preprocess CT scan
            preprocessed_ct = preprocess_ct_scan(
                ct_scan=ct_scan,
                spacing=original_spacing,
                target_spacing=target_spacing,
                target_shape=target_shape,
                normalize=True,
                enhance=True
            )

            # Preprocess mask
            resampled_mask, _ = resample_volume(aligned_mask, original_spacing, target_spacing)
            preprocessed_mask = center_crop_or_pad(resampled_mask, target_shape)
            preprocessed_mask = (preprocessed_mask > 0.5).astype(np.uint8)

            # Create patient output directory
            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)

            # Save original preprocessed data
            np.save(os.path.join(patient_output_dir, 'ct_scan.npy'), preprocessed_ct)
            np.save(os.path.join(patient_output_dir, 'mask.npy'), preprocessed_mask)

            # Apply data augmentation
            if augmentation_count > 0:
                print(f"Generating {augmentation_count} augmented samples for patient {patient_id}...")

                # Create augmentation directory
                aug_dir = os.path.join(patient_output_dir, 'augmentations')
                os.makedirs(aug_dir, exist_ok=True)

                # Generate augmented samples
                for i in range(augmentation_count):
                    # Set random seed for reproducibility and to ensure CT and mask get the same transformations
                    random_seed = i + hash(patient_id) % 10000
                    np.random.seed(random_seed)

                    # Augment CT scan
                    augmented_ct = augment_ct_scan(preprocessed_ct, augmentation_types)

                    # For masks, we use only spatial transformations (not intensity-based ones)
                    mask_aug_types = ['rotation', 'shift', 'flip', 'zoom'] if augmentation_types is None else [
                        t for t in augmentation_types if t in ['rotation', 'shift', 'flip', 'zoom']
                    ]

                    # Apply same transformations to mask
                    np.random.seed(random_seed)  # Use same seed for mask to get identical transformations
                    augmented_mask = augment_ct_scan(preprocessed_mask.astype(float), mask_aug_types)

                    # Reset random seed
                    np.random.seed(None)

                    # Ensure mask remains binary
                    augmented_mask = (augmented_mask > 0.5).astype(np.uint8)

                    # Save augmented data
                    np.save(os.path.join(aug_dir, f'ct_scan_aug_{i}.npy'), augmented_ct)
                    np.save(os.path.join(aug_dir, f'mask_aug_{i}.npy'), augmented_mask)

                print(f"Successfully generated {augmentation_count} augmented samples")

            print(f"Successfully processed patient {patient_id}")

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")

    print("\nBatch preprocessing with augmentation complete!")

def main(csv_path="E:/cancer stage/lung_csv.csv",
         patch_size=(64, 64, 64),
         batch_size=8,
         num_epochs=50,
         patience=10,
         learning_rate=0.001,
         weight_decay=1e-4,
         model_save_path="resnet18_3d_cancer_stage.pth",
         base_dir="E:/cancer stage/NSCLC-Radiomics",
         target_spacing=[1.0, 1.0, 1.0],
         target_shape=(128, 256, 256),
         use_augmentation=True,
         augmentation_types=None,
         use_direct_processing=True):
    """
    Main function to run the cancer stage classification pipeline

    Args:
        csv_path: Path to CSV with labels
        patch_size: Size of patches to extract
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        patience: Patience for early stopping
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        model_save_path: Path to save the trained model
        base_dir: Base directory containing patient data
        target_spacing: Target voxel spacing in mm
        target_shape: Target shape for preprocessing
        use_augmentation: Whether to use augmentation for training
        augmentation_types: List of augmentation types to apply
        use_direct_processing: Whether to use direct processing without saving to disk
    """
    print("=" * 80)
    print("CANCER STAGE CLASSIFICATION USING 3D RESNET")
    print("=" * 80)

    # Set random seeds for reproducibility
    set_seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    if use_direct_processing:
        print("\nUsing direct processing without saving to disk...")
        train_dataset = DirectCTScanDataset(
            base_dir=base_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            target_spacing=target_spacing,
            target_shape=target_shape,
            mode='train',
            use_augmentation=use_augmentation,
            augmentation_types=augmentation_types
        )

        val_dataset = DirectCTScanDataset(
            base_dir=base_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            target_spacing=target_spacing,
            target_shape=target_shape,
            mode='val',
            use_augmentation=False
        )

        test_dataset = DirectCTScanDataset(
            base_dir=base_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            target_spacing=target_spacing,
            target_shape=target_shape,
            mode='test',
            use_augmentation=False
        )
    else:
        # Use the original dataset that loads from preprocessed files
        data_dir = "preprocessed_data_integrated"
        print(f"\nUsing preprocessed data from {data_dir}...")
        train_dataset = CTScanDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            mode='train'
        )

        val_dataset = CTScanDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            mode='val'
        )

        test_dataset = CTScanDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            mode='test'
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = ResNet18_3D(num_classes=4)
    model = model.to(device)

    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to(device))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train model
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience
    )

    # Save trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'")

    # Plot training history
    plot_training_history(history)

    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def evaluate_pretrained_model(model_path, csv_path, patch_size=(64, 64, 64), batch_size=8,
                       base_dir="E:/cancer stage/NSCLC-Radiomics", target_spacing=[1.0, 1.0, 1.0],
                       target_shape=(128, 256, 256), use_direct_processing=True, data_dir=None):
    """
    Load a pretrained model and evaluate it on the test set

    Args:
        model_path: Path to the pretrained model
        csv_path: Path to CSV with labels
        patch_size: Size of patches to extract
        batch_size: Batch size
        base_dir: Base directory containing patient data
        target_spacing: Target voxel spacing in mm
        target_shape: Target shape for preprocessing
        use_direct_processing: Whether to use direct processing without saving to disk
        data_dir: Directory with preprocessed data (only used if use_direct_processing=False)
    """
    print("=" * 80)
    print(f"EVALUATING PRETRAINED MODEL: {model_path}")
    print("=" * 80)

    # Set random seeds for reproducibility
    set_seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test dataset and dataloader
    if use_direct_processing:
        print("\nUsing direct processing without saving to disk...")
        test_dataset = DirectCTScanDataset(
            base_dir=base_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            target_spacing=target_spacing,
            target_shape=target_shape,
            mode='test',
            use_augmentation=False
        )
    else:
        if data_dir is None:
            data_dir = "preprocessed_data_integrated"
        print(f"\nUsing preprocessed data from {data_dir}...")
        test_dataset = CTScanDataset(
            data_dir=data_dir,
            csv_path=csv_path,
            patch_size=patch_size,
            mode='test'
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = ResNet18_3D(num_classes=4)

    # Load pretrained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model on test set
    print("Evaluating model on test set...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return test_acc

if __name__ == "__main__":
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description='Cancer Stage Classification using 3D ResNet')

    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    # Train mode parser
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--csv_path', type=str, default='E:/cancer stage/lung_csv.csv',
                        help='Path to CSV with labels')
    train_parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Size of patches to extract (depth, height, width)')
    train_parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    train_parser.add_argument('--num_epochs', type=int, default=50,
                        help='Maximum number of epochs')
    train_parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    train_parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    train_parser.add_argument('--model_save_path', type=str, default='resnet18_3d_cancer_stage.pth',
                        help='Path to save the trained model')
    train_parser.add_argument('--base_dir', type=str, default='E:/cancer stage/NSCLC-Radiomics',
                        help='Base directory containing patient data')
    train_parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Target voxel spacing in mm (z, y, x)')
    train_parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 256, 256],
                        help='Target shape for preprocessing (depth, height, width)')
    train_parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Whether to use augmentation for training')
    train_parser.add_argument('--use_direct_processing', action='store_true', default=True,
                        help='Whether to use direct processing without saving to disk')

    # Evaluate mode parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a pretrained model')
    eval_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model')
    eval_parser.add_argument('--csv_path', type=str, default='E:/cancer stage/lung_csv.csv',
                        help='Path to CSV with labels')
    eval_parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Size of patches to extract (depth, height, width)')
    eval_parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    eval_parser.add_argument('--base_dir', type=str, default='E:/cancer stage/NSCLC-Radiomics',
                        help='Base directory containing patient data')
    eval_parser.add_argument('--target_spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help='Target voxel spacing in mm (z, y, x)')
    eval_parser.add_argument('--target_shape', type=int, nargs=3, default=[128, 256, 256],
                        help='Target shape for preprocessing (depth, height, width)')
    eval_parser.add_argument('--use_direct_processing', action='store_true', default=True,
                        help='Whether to use direct processing without saving to disk')
    eval_parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory with preprocessed data (only used if use_direct_processing=False)')

    # Parse arguments
    args = parser.parse_args()

    # If no mode is specified, show help and exit
    if args.mode is None:
        parser.print_help()
        exit(1)

    # Convert patch_size from list to tuple
    patch_size = tuple(args.patch_size)

    # Call appropriate function based on mode
    if args.mode == 'train':
        # Convert target_spacing and target_shape from lists to tuples
        target_spacing = tuple(args.target_spacing)
        target_shape = tuple(args.target_shape)

        main(
            csv_path=args.csv_path,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_save_path=args.model_save_path,
            base_dir=args.base_dir,
            target_spacing=target_spacing,
            target_shape=target_shape,
            use_augmentation=args.use_augmentation,
            use_direct_processing=args.use_direct_processing
        )
    elif args.mode == 'evaluate':
        # Convert target_spacing and target_shape from lists to tuples
        target_spacing = tuple(args.target_spacing)
        target_shape = tuple(args.target_shape)

        evaluate_pretrained_model(
            model_path=args.model_path,
            csv_path=args.csv_path,
            patch_size=patch_size,
            batch_size=args.batch_size,
            base_dir=args.base_dir,
            target_spacing=target_spacing,
            target_shape=target_shape,
            use_direct_processing=args.use_direct_processing,
            data_dir=args.data_dir
        )
