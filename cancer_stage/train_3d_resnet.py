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
from torchvision import models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Define the 3D ResNet model
class BasicBlock3D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.dropout = nn.Dropout3d(0.2)  # Add dropout for regularization

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after activation
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4, pretrained=False):  # 4 stages: I, II, III, IV
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)  # Add dropout before final layer
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool3d(out, kernel_size=3, stride=2, padding=1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)  # Apply dropout before final layer
        out = self.fc(out)
        return out

def ResNet18_3D(num_classes=4, pretrained=False):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes, pretrained)

# Custom Dataset for CT scans with patches
class CTScanDataset(Dataset):
    def __init__(self, data_dir, csv_path, patch_size=(64, 64, 64), transform=None, mode='train', test_size=0.2, val_size=0.1):
        """
        Dataset for loading CT scans and their cancer stage labels

        Args:
            data_dir: Directory containing preprocessed patient data
            csv_path: Path to CSV file with patient IDs and cancer stage labels
            patch_size: Size of patches to extract from CT volumes
            transform: Optional transforms to apply
            mode: 'train', 'val', or 'test'
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode

        # Load CSV with patient IDs and cancer stage labels
        self.df = pd.read_csv(csv_path)

        # Map cancer stage to numerical labels (0-3)
        stage_mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        self.df['stage_label'] = self.df['cancer_stage'].map(stage_mapping)

        # Get list of all available patient IDs in the data directory
        available_patients = []
        for patient_id in os.listdir(data_dir):
            patient_dir = os.path.join(data_dir, patient_id)
            if os.path.isdir(patient_dir):
                # Check if CT scan file exists
                ct_path = os.path.join(patient_dir, 'ct_scan.npy')
                if os.path.exists(ct_path):
                    available_patients.append(patient_id)

        # Filter dataframe to only include available patients
        self.df = self.df[self.df['patient_id'].isin(available_patients)]

        # Split data into train, validation, and test sets
        train_val_df, test_df = train_test_split(
            self.df, test_size=test_size, random_state=42, stratify=self.df['stage_label']
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_df['stage_label']
        )

        # Select the appropriate dataframe based on mode
        if mode == 'train':
            self.df = train_df
        elif mode == 'val':
            self.df = val_df
        elif mode == 'test':
            self.df = test_df

        # Create a list of all samples (original and augmented)
        self.samples = []

        for _, row in self.df.iterrows():
            patient_id = row['patient_id']
            label = row['stage_label']

            # Add original sample
            patient_dir = os.path.join(data_dir, patient_id)
            ct_path = os.path.join(patient_dir, 'ct_scan.npy')

            if os.path.exists(ct_path):
                self.samples.append((ct_path, label))

            # Add augmented samples if in training mode
            if mode == 'train':
                aug_dir = os.path.join(patient_dir, 'augmentations')
                if os.path.exists(aug_dir):
                    aug_ct_files = [f for f in os.listdir(aug_dir) if f.startswith('ct_scan_aug_') and f.endswith('.npy')]

                    for aug_ct_file in aug_ct_files:
                        aug_ct_path = os.path.join(aug_dir, aug_ct_file)
                        self.samples.append((aug_ct_path, label))

        print(f"Created {mode} dataset with {len(self.samples)} samples")

        # Calculate class weights for weighted loss
        if mode == 'train':
            class_counts = self.df['stage_label'].value_counts().sort_index()
            self.class_weights = 1.0 / class_counts
            self.class_weights = self.class_weights / self.class_weights.sum()
            self.class_weights = torch.tensor(self.class_weights.values, dtype=torch.float32)
            print(f"Class weights: {self.class_weights}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, label = self.samples[idx]

        # Load CT scan
        ct_scan = np.load(ct_path)

        # Extract a random patch if in training mode, or center patch otherwise
        if self.mode == 'train':
            patch = self._extract_random_patch(ct_scan)
        else:
            patch = self._extract_center_patch(ct_scan)

        # Apply transforms if specified
        if self.transform:
            patch = self.transform(patch)

        # Convert to tensor and add channel dimension
        patch = torch.from_numpy(patch).float().unsqueeze(0)  # Add channel dimension

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

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                num_epochs=50, patience=10, checkpoint_dir='checkpoints', save_freq=5,
                resume_from=None):
    """
    Train the model with early stopping and checkpointing

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

    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    import os
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
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accs = checkpoint['train_accs']
        val_accs = checkpoint['val_accs']
        print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
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

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    # Create class names dynamically based on actual classes present
    present_class_names = [class_names[i] for i in unique_classes]

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(present_class_names))
    plt.xticks(tick_marks, present_class_names, rotation=45)
    plt.yticks(tick_marks, present_class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Main function
def main():
    # Configuration
    data_dir = "preprocessed_data_integrated"  # Directory with preprocessed data
    csv_path = "E:/cancer stage/lung_csv.csv"  # Path to CSV with labels
    patch_size = (64, 64, 64)  # Size of patches to extract
    batch_size = 8  # Batch size
    num_epochs = 50  # Maximum number of epochs
    patience = 10  # Patience for early stopping
    learning_rate = 0.001  # Initial learning rate
    weight_decay = 1e-4  # Weight decay for regularization

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
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
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        save_freq=5,  # Save checkpoint every 5 epochs
        resume_from=resume_from
    )

    # Save trained model
    torch.save(model.state_dict(), 'resnet18_3d_cancer_stage.pth')
    print("Model saved to 'resnet18_3d_cancer_stage.pth'")

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
    # Get unique classes present in the data
    unique_classes = sorted(np.unique(all_labels))
    # Create class names dynamically based on actual classes present
    present_class_names = [class_names[i] for i in unique_classes]
    print(classification_report(all_labels, all_preds, target_names=present_class_names))

if __name__ == "__main__":
    main()
