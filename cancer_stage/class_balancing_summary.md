# Class Balancing Techniques for Cancer Stage Classification

This document summarizes the techniques implemented to address class imbalance in the cancer stage classification project.

## Problem

The cancer stage dataset is imbalanced, with some stages having significantly more samples than others. This leads to:
- Model bias towards majority classes
- Poor performance on minority classes
- Validation accuracy dropping during training due to overfitting to majority classes

## Solutions Implemented

### 1. Weighted Random Sampling

- Added `get_sample_weights()` method to `DirectCTScanDataset` class
- Implemented `WeightedRandomSampler` in the training DataLoader
- This ensures that each batch contains a balanced representation of all classes

### 2. Combined Loss Function

- Implemented `FocalLoss` which focuses more on hard examples (typically from minority classes)
- Created `CombinedLoss` that combines weighted cross-entropy and focal loss
- This approach gives more weight to minority classes during training

### 3. Model Architecture Improvements

- Increased dropout rate from 0.2 to 0.3 in dense layers
- Added additional dropout layers after each dense block
- Added an intermediate fully connected layer with dropout in the classifier
- These changes help prevent overfitting to the majority classes

### 4. Training Optimizations

- Added gradient clipping to prevent exploding gradients
- Reduced learning rate for more stable training
- Increased weight decay for better regularization
- Enabled progressive unfreezing for better transfer learning
- Increased patience for early stopping to allow more exploration

## Expected Outcomes

These changes should result in:
- More balanced performance across all cancer stages
- Improved validation accuracy that doesn't drop during training
- Better generalization to unseen data
- Higher F1 scores for minority classes

## Monitoring

To verify the effectiveness of these changes, monitor:
- Per-class accuracy and F1 scores
- Confusion matrix to see if predictions are balanced across classes
- Training and validation loss curves for signs of overfitting
