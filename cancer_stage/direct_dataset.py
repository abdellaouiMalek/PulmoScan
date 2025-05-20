"""
Direct Dataset for CT Scans

This module provides a PyTorch Dataset that directly processes CT scans without saving them to disk.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import pydicom
from sklearn.model_selection import train_test_split

from ct_preprocessing import (
    preprocess_ct_scan, resample_volume, center_crop_or_pad, load_dicom_series_safely
)
from ct_augmentation import augment_ct_scan
from Data_understanding import find_patient_data_paths

class DirectCTScanDataset(Dataset):
    def __init__(self, base_dir, csv_path, patch_size=(64, 64, 64),
                 target_spacing=[1.0, 1.0, 1.0], target_shape=(128, 256, 256),
                 transform=None, mode='train', test_size=0.2, val_size=0.1,
                 use_augmentation=True, augmentation_types=None):
        """
        Dataset for directly processing CT scans and their cancer stage labels
        """
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.transform = transform
        self.mode = mode
        self.use_augmentation = use_augmentation and mode == 'train'
        self.augmentation_types = augmentation_types

        # Load CSV with patient IDs and cancer stage labels
        self.df = pd.read_csv(csv_path)

        # Map cancer stage to numerical labels (0-3)
        # Merge IIIa and IIIb into a single III stage
        stage_mapping = {'I': 0, 'Ia': 0, 'Ib': 0,
                         'II': 1, 'IIa': 1, 'IIb': 1,
                         'III': 2, 'IIIa': 2, 'IIIb': 2,
                         'IV': 3, 'IVa': 3, 'IVb': 3}

        # Check which column name is used in the CSV file
        if 'Overall.Stage' in self.df.columns:
            stage_column = 'Overall.Stage'
        elif 'cancer_stage' in self.df.columns:
            stage_column = 'cancer_stage'
        else:
            # Try to find a column that might contain stage information
            potential_columns = [col for col in self.df.columns if 'stage' in col.lower()]
            if potential_columns:
                stage_column = potential_columns[0]
                print(f"Using column '{stage_column}' for cancer stage information")
            else:
                raise ValueError("Could not find cancer stage column in CSV file")

        # Print unique values in the stage column to help with debugging
        print(f"Unique values in {stage_column}: {self.df[stage_column].unique()}")

        # Drop rows with NaN values in the stage column
        nan_count = self.df[stage_column].isna().sum()
        if nan_count > 0:
            print(f"Dropping {nan_count} rows with NaN values in {stage_column}")
            self.df = self.df.dropna(subset=[stage_column])

        # Merge IIIa and IIIb into a single III stage
        self.df[stage_column] = self.df[stage_column].replace({'IIIa': 'III', 'IIIb': 'III'})
        print(f"After merging IIIa and IIIb: Unique values in {stage_column}: {self.df[stage_column].unique()}")

        # Map stages to numerical labels
        self.df['stage_label'] = self.df[stage_column].map(stage_mapping)

        # Check if any stages couldn't be mapped
        unmapped_count = self.df['stage_label'].isna().sum()
        if unmapped_count > 0:
            print(f"Warning: {unmapped_count} rows have stage values that couldn't be mapped")
            print(f"Unmapped stage values: {self.df[self.df['stage_label'].isna()][stage_column].unique()}")
            # Drop rows with unmapped stage values
            self.df = self.df.dropna(subset=['stage_label'])

        # Get list of all available patient IDs in the base directory
        available_patients = []
        for patient_id in os.listdir(base_dir):
            patient_dir = os.path.join(base_dir, patient_id)
            if os.path.isdir(patient_dir):
                # Check if patient has CT and mask data
                ct_dir, mask_path = find_patient_data_paths(base_dir, patient_id)
                if ct_dir is not None and mask_path is not None:
                    available_patients.append(patient_id)

        # Check which column name is used for patient IDs
        if 'PatientID' in self.df.columns:
            patient_id_column = 'PatientID'
        elif 'patient_id' in self.df.columns:
            patient_id_column = 'patient_id'
        else:
            # Try to find a column that might contain patient ID information
            potential_columns = [col for col in self.df.columns if 'patient' in col.lower() or 'id' in col.lower()]
            if potential_columns:
                patient_id_column = potential_columns[0]
                print(f"Using column '{patient_id_column}' for patient ID information")
            else:
                raise ValueError("Could not find patient ID column in CSV file")

        # Filter dataframe to only include available patients
        self.df = self.df[self.df[patient_id_column].isin(available_patients)]

        # Split data into train, validation, and test sets
        # Check if we have enough samples for stratification
        min_samples_per_class = self.df['stage_label'].value_counts().min()
        if min_samples_per_class >= 2 and len(self.df) >= 5:
            # We have enough samples for stratified split
            train_val_df, test_df = train_test_split(
                self.df, test_size=test_size, random_state=42, stratify=self.df['stage_label']
            )

            # Check if we have enough samples for stratified validation split
            min_samples_per_class_train_val = train_val_df['stage_label'].value_counts().min()
            if min_samples_per_class_train_val >= 2:
                train_df, val_df = train_test_split(
                    train_val_df, test_size=val_size/(1-test_size), random_state=42,
                    stratify=train_val_df['stage_label']
                )
            else:
                # Not enough samples for stratified validation split
                print("Warning: Not enough samples per class for stratified validation split")
                train_df, val_df = train_test_split(
                    train_val_df, test_size=val_size/(1-test_size), random_state=42
                )
        else:
            # Not enough samples for stratified split
            print("Warning: Not enough samples per class for stratified split")
            train_val_df, test_df = train_test_split(
                self.df, test_size=test_size, random_state=42
            )

            train_df, val_df = train_test_split(
                train_val_df, test_size=val_size/(1-test_size), random_state=42
            )

        # Select the appropriate dataframe based on mode
        if mode == 'train':
            self.df = train_df
        elif mode == 'val':
            self.df = val_df
        elif mode == 'test':
            self.df = test_df

        # Store patient IDs and labels
        self.patients = []
        for _, row in self.df.iterrows():
            patient_id = row[patient_id_column]
            label = row['stage_label']
            self.patients.append((patient_id, label))

        print(f"Created {mode} dataset with {len(self.patients)} patients")

        # Calculate class weights for weighted loss
        if mode == 'train':
            # Get the number of classes from the stage mapping (should be 4 for stages I, II, III, IV)
            num_classes = 4  # Hardcoded to match the model's expected number of classes

            # Calculate weights for classes that are present in the data
            class_counts = self.df['stage_label'].value_counts().sort_index()
            print(f"Class distribution: {class_counts}")

            # Calculate weights inversely proportional to class frequencies
            # This gives higher weights to underrepresented classes
            total_samples = len(self.df)
            weights = total_samples / (class_counts * num_classes)

            # Create a tensor with zeros for all classes
            self.class_weights = torch.zeros(num_classes, dtype=torch.float32)

            # Fill in weights for classes that are present in the data
            for class_idx, weight in zip(class_counts.index.astype(int), weights.values):
                if class_idx < num_classes:
                    self.class_weights[class_idx] = weight

            # If any class has zero weight (not present in data), set it to the mean of other weights
            zero_weight_indices = (self.class_weights == 0).nonzero(as_tuple=True)[0]
            if len(zero_weight_indices) > 0 and len(zero_weight_indices) < num_classes:
                mean_weight = self.class_weights[self.class_weights > 0].mean().item()
                self.class_weights[zero_weight_indices] = mean_weight

            print(f"Class weights: {self.class_weights}")
            print(f"These weights will be used to handle class imbalance during training.")

    def __len__(self):
        return len(self.patients)

    def get_sample_weights(self):
        """
        Calculate sample weights for WeightedRandomSampler
        This is used to balance the dataset during training by oversampling
        underrepresented classes.

        Returns:
            List of weights for each sample in the dataset
        """
        # Get all labels
        labels = [label for _, label in self.patients]

        # Count occurrences of each class
        class_counts = {}
        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Get total number of samples
        num_samples = len(labels)
        num_classes = len(class_counts)

        # Calculate weights as (total_samples) / (num_classes * class_count)
        # This gives higher weights to samples from underrepresented classes
        class_weights = {}
        for label, count in class_counts.items():
            class_weights[label] = num_samples / (num_classes * count)

        # Assign weight to each sample based on its class
        weights = []
        for _, label in self.patients:
            if label in class_weights:
                weights.append(class_weights[label])
            else:
                # Fallback for any class not in class_weights
                weights.append(1.0)

        return weights

    def __getitem__(self, idx):
        patient_id, label = self.patients[idx]

        # Find paths for this patient
        ct_dir, mask_path = find_patient_data_paths(self.base_dir, patient_id)

        # Process CT scan and mask
        try:
            # Load CT scan
            ct_scan, _, spacing = load_dicom_series_safely(ct_dir)  # We don't need valid_dicom_datasets

            # Load mask (we don't actually use the mask for training, but we check if it exists)
            # This is just to verify that the patient has both CT and mask data
            mask = pydicom.dcmread(mask_path)
            if not hasattr(mask, 'pixel_array'):
                raise ValueError(f"Mask for patient {patient_id} has no pixel data")

            # Preprocess CT scan
            preprocessed_ct = preprocess_ct_scan(
                ct_scan=ct_scan,
                spacing=spacing,
                target_spacing=self.target_spacing,
                target_shape=self.target_shape,
                normalize=True,
                enhance=True
            )

            # Apply augmentation if in training mode
            if self.use_augmentation and random.random() < 0.5:  # 50% chance of augmentation
                # Set random seed for reproducibility
                random_seed = random.randint(0, 10000)
                np.random.seed(random_seed)

                # Augment CT scan
                preprocessed_ct = augment_ct_scan(preprocessed_ct, self.augmentation_types)

                # Reset random seed
                np.random.seed(None)

            # Extract a random patch if in training mode, or center patch otherwise
            if self.mode == 'train':
                patch = self._extract_random_patch(preprocessed_ct)
            else:
                patch = self._extract_center_patch(preprocessed_ct)

            # Apply transforms if specified
            if self.transform:
                patch = self.transform(patch)

            # Convert to tensor and add channel dimension
            patch = torch.from_numpy(patch).float().unsqueeze(0)  # Add channel dimension

            return patch, label

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            # Return a zero tensor and the label in case of error
            patch = np.zeros(self.patch_size, dtype=np.float32)
            patch = torch.from_numpy(patch).float().unsqueeze(0)
            return patch, label

    def _extract_random_patch(self, volume):
        """Extract a random patch from the volume"""
        d, h, w = volume.shape
        pd, ph, pw = self.patch_size

        # Ensure patch size is not larger than volume
        pd = min(pd, d)
        ph = min(ph, h)
        pw = min(pw, w)

        # Random starting point
        d_start = random.randint(0, d - pd) if d > pd else 0
        h_start = random.randint(0, h - ph) if h > ph else 0
        w_start = random.randint(0, w - pw) if w > pw else 0

        # Extract patch
        patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        # Pad if necessary
        if patch.shape != self.patch_size:
            temp_patch = np.zeros(self.patch_size, dtype=patch.dtype)
            temp_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = temp_patch

        return patch

    def _extract_center_patch(self, volume):
        """Extract the center patch from the volume"""
        d, h, w = volume.shape
        pd, ph, pw = self.patch_size

        # Ensure patch size is not larger than volume
        pd = min(pd, d)
        ph = min(ph, h)
        pw = min(pw, w)

        # Center starting point
        d_start = (d - pd) // 2 if d > pd else 0
        h_start = (h - ph) // 2 if h > ph else 0
        w_start = (w - pw) // 2 if w > pw else 0

        # Extract patch
        patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        # Pad if necessary
        if patch.shape != self.patch_size:
            temp_patch = np.zeros(self.patch_size, dtype=patch.dtype)
            temp_patch[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = temp_patch

        return patch
