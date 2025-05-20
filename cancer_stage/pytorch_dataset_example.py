"""
Example PyTorch Dataset for CT Scans with Augmentation

This module provides an example of how to create a PyTorch dataset that loads
preprocessed and augmented CT scans for training deep learning models.

Author: [Your Name]
Date: [Current Date]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob

from ct_augmentation import augment_ct_scan

class AugmentedCTDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None, include_augmentations=True, transform=None):
        """
        Dataset for loading preprocessed and augmented CT scans
        
        Args:
            data_dir: Directory containing preprocessed patient data
            patient_ids: List of patient IDs to include (if None, include all)
            include_augmentations: Whether to include augmented samples
            transform: Additional transformations to apply (can be None)
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get all patient directories
        if patient_ids is None:
            patient_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # Load patient data paths and labels
        self.samples = []
        
        for patient_id in patient_ids:
            patient_dir = os.path.join(data_dir, patient_id)
            
            # Get label for this patient (this is just an example - replace with your actual label logic)
            # For example, you might load labels from a CSV file
            label = 0  # Placeholder - replace with actual label
            
            # Add original sample
            ct_path = os.path.join(patient_dir, "ct_scan.npy")
            mask_path = os.path.join(patient_dir, "mask.npy")
            
            if os.path.exists(ct_path) and os.path.exists(mask_path):
                self.samples.append({
                    "patient_id": patient_id,
                    "ct_path": ct_path,
                    "mask_path": mask_path,
                    "label": label,
                    "is_augmented": False
                })
            
            # Add augmented samples if requested
            if include_augmentations:
                aug_dir = os.path.join(patient_dir, "augmentations")
                if os.path.exists(aug_dir):
                    # Find all augmented CT scans
                    aug_ct_paths = glob.glob(os.path.join(aug_dir, "ct_scan_aug_*.npy"))
                    
                    for aug_ct_path in aug_ct_paths:
                        # Get corresponding mask path
                        aug_id = os.path.basename(aug_ct_path).replace("ct_scan_aug_", "").replace(".npy", "")
                        aug_mask_path = os.path.join(aug_dir, f"mask_aug_{aug_id}.npy")
                        
                        if os.path.exists(aug_mask_path):
                            self.samples.append({
                                "patient_id": patient_id,
                                "ct_path": aug_ct_path,
                                "mask_path": aug_mask_path,
                                "label": label,
                                "is_augmented": True
                            })
        
        print(f"Loaded {len(self.samples)} samples ({len(patient_ids)} patients)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load CT scan and mask
        ct_scan = np.load(sample["ct_path"])
        mask = np.load(sample["mask_path"])
        
        # Apply additional transformations if specified
        if self.transform and not sample["is_augmented"]:  # Only apply to non-augmented samples
            # Set random seed for reproducibility
            random_seed = idx
            np.random.seed(random_seed)
            
            # Apply same transformation to both CT and mask
            ct_scan = augment_ct_scan(ct_scan)
            
            # For masks, use only spatial transformations
            np.random.seed(random_seed)
            mask = augment_ct_scan(mask.astype(float), ["rotation", "shift", "flip", "zoom"])
            mask = (mask > 0.5).astype(np.uint8)
            
            # Reset random seed
            np.random.seed(None)
        
        # Convert to PyTorch tensors
        ct_tensor = torch.from_numpy(ct_scan).float().unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)   # Add channel dimension
        label_tensor = torch.tensor(sample["label"]).long()
        
        return {
            "ct": ct_tensor,
            "mask": mask_tensor,
            "label": label_tensor,
            "patient_id": sample["patient_id"],
            "is_augmented": sample["is_augmented"]
        }

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = AugmentedCTDataset(
        data_dir="preprocessed_data_augmented",
        include_augmentations=True,
        transform=True  # Apply additional random augmentations during training
    )
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Example of iterating through the data loader
    for batch_idx, batch in enumerate(train_loader):
        ct = batch["ct"]
        mask = batch["mask"]
        label = batch["label"]
        
        print(f"Batch {batch_idx}:")
        print(f"  CT shape: {ct.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Label: {label}")
        
        # In a real training loop, you would pass these to your model
        # outputs = model(ct)
        # loss = criterion(outputs, label)
        # loss.backward()
        # optimizer.step()
        
        # Only process a few batches for this example
        if batch_idx >= 2:
            break
