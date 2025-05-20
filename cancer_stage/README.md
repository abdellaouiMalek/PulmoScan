# Cancer Stage Classification

This module implements a 3D ResNet model for lung cancer stage classification using CT scans.

## Overview

The cancer stage classification module uses a 3D ResNet-18 architecture to classify lung cancer stages (I, II, III, IV) from CT scans. The module includes:

- Data preprocessing for CT scans and segmentation masks
- Data augmentation for 3D medical images
- 3D ResNet model implementation
- Training and evaluation pipeline
- Command-line interface for training and evaluation

## Usage

### Training a New Model

To train a new model, use the following command:

```bash
python main.py train [options]
```

Options:
- `--csv_path`: Path to CSV with labels (default: 'E:/cancer stage/lung_csv.csv')
- `--patch_size`: Size of patches to extract (depth, height, width) (default: 64 64 64)
- `--batch_size`: Batch size (default: 8)
- `--num_epochs`: Maximum number of epochs (default: 50)
- `--patience`: Patience for early stopping (default: 10)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--model_save_path`: Path to save the trained model (default: 'resnet18_3d_cancer_stage.pth')
- `--base_dir`: Base directory containing patient data (default: 'E:/cancer stage/NSCLC-Radiomics')
- `--target_spacing`: Target voxel spacing in mm (z, y, x) (default: 1.0 1.0 1.0)
- `--target_shape`: Target shape for preprocessing (depth, height, width) (default: 128 256 256)
- `--use_augmentation`: Whether to use augmentation for training (default: True)
- `--use_direct_processing`: Whether to use direct processing without saving to disk (default: True)

Example:
```bash
python main.py train --batch_size 16 --num_epochs 100 --use_direct_processing
```

### Evaluating a Pretrained Model

To evaluate a pretrained model, use the following command:

```bash
python main.py evaluate --model_path MODEL_PATH [options]
```

Options:
- `--model_path`: Path to the pretrained model (required)
- `--csv_path`: Path to CSV with labels (default: 'E:/cancer stage/lung_csv.csv')
- `--patch_size`: Size of patches to extract (depth, height, width) (default: 64 64 64)
- `--batch_size`: Batch size (default: 8)
- `--base_dir`: Base directory containing patient data (default: 'E:/cancer stage/NSCLC-Radiomics')
- `--target_spacing`: Target voxel spacing in mm (z, y, x) (default: 1.0 1.0 1.0)
- `--target_shape`: Target shape for preprocessing (depth, height, width) (default: 128 256 256)
- `--use_direct_processing`: Whether to use direct processing without saving to disk (default: True)
- `--data_dir`: Directory with preprocessed data (only used if use_direct_processing=False)

Example:
```bash
python main.py evaluate --model_path resnet18_3d_cancer_stage.pth --batch_size 16 --use_direct_processing
```

## Data Format

### Direct Processing Mode (Default)

When using direct processing mode (`--use_direct_processing`), the module works with raw DICOM data:

- The base directory should contain patient folders (e.g., LUNG1-001, LUNG1-002, etc.)
- Each patient folder should contain:
  - A directory with CT scan DICOM files
  - A DICOM segmentation mask file

The module will automatically:
1. Load the raw DICOM files
2. Preprocess them on-the-fly (resampling, normalization, etc.)
3. Extract patches for training/evaluation
4. Apply augmentation during training if enabled

### Preprocessed Data Mode

When not using direct processing mode, the module expects preprocessed CT scans and segmentation masks in the following format:

- Each patient's data should be in a separate directory named with the patient ID
- Each patient directory should contain:
  - `ct_scan.npy`: Preprocessed CT scan as a NumPy array
  - `mask.npy`: Preprocessed segmentation mask as a NumPy array
  - `augmentations/`: Directory containing augmented versions of the CT scan (for training)

### CSV Format

The CSV file should contain the following columns:
- `patient_id`: Patient ID matching the directory names
- `cancer_stage`: Cancer stage (I, II, III, IV)

## Model Architecture

The model is a 3D adaptation of the ResNet-18 architecture with the following modifications:
- 3D convolutions instead of 2D convolutions
- Dropout layers for regularization
- Batch normalization
- Adaptive average pooling

## Performance Metrics

The model is evaluated using the following metrics:
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

## Dependencies

- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm
- pydicom (for preprocessing)
