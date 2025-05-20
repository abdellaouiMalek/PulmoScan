"""
Training script for 3D DenseNet model for cancer stage classification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Import the DenseNet model
from densenet_3d import DenseNet121_3D, DenseNet169_3D, DenseNet201_3D

# Import the dataset
from direct_dataset import DirectCTScanDataset

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Training function with checkpointing, early stopping, and learning rate scheduling
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs=50, patience=10, checkpoint_dir='densenet_checkpoints', save_freq=5,
                resume_from=None, progressive_unfreezing=False):
    """
    Train the model with early stopping, checkpointing, and learning rate scheduling

    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Number of epochs to wait for improvement before early stopping
        checkpoint_dir: Directory to save checkpoints
        save_freq: Frequency (in epochs) to save regular checkpoints
        resume_from: Path to checkpoint to resume training from
        progressive_unfreezing: Whether to use progressive unfreezing for transfer learning

    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    start_epoch = 0

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Resume from checkpoint if specified
    if resume_from and os.path.isfile(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        # Handle model architecture changes
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Warning: Could not load model state dict: {e}")
            print("Attempting to load compatible weights...")

            # Get state dict from checkpoint and current model
            checkpoint_state_dict = checkpoint['model_state_dict']
            model_state_dict = model.state_dict()

            # Create a new state dict with only compatible keys
            compatible_state_dict = {}
            for k, v in checkpoint_state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    compatible_state_dict[k] = v
                    print(f"Loaded: {k}")
                else:
                    print(f"Skipped: {k} - Shape mismatch or not in model")

            # Load the compatible weights
            model.load_state_dict(compatible_state_dict, strict=False)
            print("Loaded compatible weights. New parameters will be initialized randomly.")

        # Load optimizer and scheduler states
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print("Could not load optimizer or scheduler state. Using fresh states.")

        # Load training history
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accs = checkpoint['train_accs']
        val_accs = checkpoint['val_accs']
        print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")

    # Progressive unfreezing setup
    if progressive_unfreezing and hasattr(model, 'features'):
        # Initially freeze all layers except the classifier
        for param in model.features.parameters():
            param.requires_grad = False

        # Get the number of blocks in the model
        num_blocks = sum(1 for name, _ in model.features.named_children() if 'denseblock' in name)

        # Calculate when to unfreeze each block
        unfreeze_epochs = [int(start_epoch + (num_epochs - start_epoch) * (i / num_blocks)) for i in range(num_blocks)]
        print(f"Progressive unfreezing schedule: {unfreeze_epochs}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Progressive unfreezing
        if progressive_unfreezing and hasattr(model, 'features'):
            for i, unfreeze_epoch in enumerate(unfreeze_epochs):
                if epoch == unfreeze_epoch:
                    block_name = f'denseblock{num_blocks - i}'
                    print(f"Unfreezing {block_name}")
                    for name, param in model.features.named_parameters():
                        if block_name in name:
                            param.requires_grad = True

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item(), "acc": train_correct/train_total})

        # Calculate average training loss and accuracy
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                # Update progress bar
                val_pbar.set_postfix({"loss": loss.item(), "acc": val_correct/val_total})

        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Calculate class-wise accuracy for monitoring class imbalance
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # Check every 5 epochs or at the end
            print("\nClass-wise validation performance:")
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(predicted.cpu().numpy())

            # Calculate per-class metrics
            from sklearn.metrics import classification_report

            # Get unique classes present in the data
            unique_classes = sorted(np.unique(all_labels))

            # Full list of class names
            all_class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']

            # Create class names dynamically based on actual classes present
            present_class_names = [all_class_names[i] for i in unique_classes]

            print(f"Classes present in validation set: {present_class_names}")
            print(classification_report(all_labels, all_preds,
                                       labels=unique_classes,
                                       target_names=present_class_names,
                                       zero_division=0))

        # Update learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint at regular intervals
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Validation loss improved to {best_val_loss:.4f}")

            # Save best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")

            # Save final model
            final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, final_model_path)
            print(f"Saved final model to {final_model_path}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Return trained model and history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

    return model, history

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Test loss, accuracy, predictions, and true labels
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update metrics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average test loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total

    return test_loss, test_acc, all_preds, all_labels

# Visualization functions
def plot_training_history(history):
    """Plot training and validation loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('densenet_training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, all_class_names):
    """Plot confusion matrix"""
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    # Create class names dynamically based on actual classes present
    present_class_names = [all_class_names[i] for i in unique_classes]

    # Compute confusion matrix for only the classes that are present
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Use the class names for present classes
    tick_marks = np.arange(len(present_class_names))
    plt.xticks(tick_marks, present_class_names, rotation=45)
    plt.yticks(tick_marks, present_class_names)

    # Add text annotations
    thresh = cm.max() / 2. if cm.size > 0 and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('densenet_confusion_matrix.png')
    plt.show()

# Main function
def main():
    # Configuration parameters
    csv_path = "E:/cancer stage/lung_csv.csv"  # CSV file with patient IDs and cancer stage labels
    patch_size = (64, 64, 64)  # Size of patches to extract from CT volumes
    batch_size = 8  # Batch size for training
    num_epochs = 50  # Maximum number of epochs
    patience = 10  # Patience for early stopping
    learning_rate = 0.001  # Initial learning rate
    weight_decay = 1e-4  # Weight decay for regularization
    model_save_path = "densenet121_3d_cancer_stage.pth"  # Path to save the trained model
    base_dir = "E:/cancer stage/NSCLC-Radiomics"  # Base directory containing patient data
    target_spacing = (1.0, 1.0, 1.0)  # Target voxel spacing in mm
    target_shape = (128, 256, 256)  # Target shape for preprocessing
    use_augmentation = True  # Whether to use augmentation for training
    model_type = "densenet121"  # Model type: densenet121, densenet169, densenet201
    scheduler_type = "cosine"  # Scheduler type: plateau, cosine
    progressive_unfreezing = False  # Whether to use progressive unfreezing

    # Set random seeds for reproducibility
    set_seed(42)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    print("\nCreating datasets with direct processing...")
    train_dataset = DirectCTScanDataset(
        base_dir=base_dir,
        csv_path=csv_path,
        patch_size=patch_size,
        target_spacing=target_spacing,
        target_shape=target_shape,
        mode='train',
        use_augmentation=use_augmentation
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

    # Create model based on specified type
    print(f"\nCreating {model_type} model...")
    if model_type == "densenet121":
        model = DenseNet121_3D(num_classes=4)
    elif model_type == "densenet169":
        model = DenseNet169_3D(num_classes=4)
    elif model_type == "densenet201":
        model = DenseNet201_3D(num_classes=4)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights.to(device))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define learning rate scheduler
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Create checkpoint directory
    checkpoint_dir = f"{model_type}_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Check if there's a checkpoint to resume from
    resume_from = None
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        resume_from = best_model_path
        print(f"Found best model checkpoint: {resume_from}")

    # Train model
    print("\nStarting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        save_freq=5,  # Save checkpoint every 5 epochs
        resume_from=resume_from,
        progressive_unfreezing=progressive_unfreezing
    )

    # Save trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to '{model_save_path}'")

    # Plot training history
    plot_training_history(history)

    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot confusion matrix
    all_class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    plot_confusion_matrix(all_labels, all_preds, all_class_names)

    # Print classification report
    print("\nClassification Report:")
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(all_labels))
    # Full list of class names
    all_class_names = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    # Create class names dynamically based on actual classes present
    present_class_names = [all_class_names[i] for i in unique_classes]
    print(f"Classes present in test set: {present_class_names}")
    print(classification_report(all_labels, all_preds,
                               labels=unique_classes,
                               target_names=present_class_names,
                               zero_division=0))

if __name__ == "__main__":
    main()
