"""
train_model.py: Module avec une fonction pour entraîner le modèle (corrigé pour PyTorch 2.4+).
Charge un sous-ensemble de données à partir de balanced_data_subset.csv.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from torch.optim.lr_scheduler import StepLR
from torch.amp import GradScaler, autocast

def train_model(model_name='resnet18', force_cpu=False, num_epochs=20):
    """
    Entraîne un modèle de classification d'images pour le cancer du poumon.

    Args:
        model_name (str): Nom du modèle ('resnet18', 'efficientnetb7', 'efficientnetb0').
        force_cpu (bool): Si True, force l'entraînement sur CPU même si un GPU est disponible.
        num_epochs (int): Nombre d'époques pour l'entraînement.
    """
    # Configuration initiale
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print(f"Using device: {device}")

    # Chemins des fichiers
    DATA_PATH = "D:/PLUMOSCAN/PulmoScan/processed_images/balanced_data_subset.csv"  # Chargement du sous-ensemble
    MODEL_SAVE_PATH = f"D:/PLUMOSCAN/PulmoScan/best_model_{model_name}.pth"
    CHECKPOINT_PATH = f"D:/PLUMOSCAN/PulmoScan/checkpoint_{model_name}.pth"

    # Paramètres d'entraînement
    BATCH_SIZE = 4 if model_name == 'resnet18' else 2
    ACCUMULATION_STEPS = 4
    NUM_EPOCHS = num_epochs
    LEARNING_RATE = 0.0005
    
        
    DROPOUT_RATE = 0.5

    # Libérer la mémoire GPU si possible
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Charger dynamiquement les catégories et définir le nombre de classes
    df = pd.read_csv(DATA_PATH)
    label_map = {label: idx for idx, label in enumerate(df['major_category'].unique())}
    NUM_CLASSES = len(label_map)
    print(f"Categories: {list(label_map.keys())}")
    print(f"Number of classes: {NUM_CLASSES}")

    # Vérifier l'équilibre des classes
    print("Class distribution in full dataset:")
    print(df.groupby(["split", "major_category"]).size())

    # Séparer les données en train et val en fonction du split
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    # Vérifier que les splits existent
    if len(train_df) == 0:
        raise ValueError("No data found for 'train' split in balanced_data_subset.csv")
    if len(val_df) == 0:
        raise ValueError("No data found for 'val' split in balanced_data_subset.csv")

    print("\nClass distribution in train set:")
    print(train_df['major_category'].value_counts())
    print("\nClass distribution in validation set:")
    print(val_df['major_category'].value_counts())

    # Calculer les poids de classe pour gérer le déséquilibre
    class_counts = train_df['major_category'].value_counts()
    total_samples = len(train_df)
    class_weights = torch.tensor([total_samples / (NUM_CLASSES * class_counts[cat]) for cat in label_map.keys()], dtype=torch.float)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")

    # Transformations pour les images
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

    # Dataset personnalisé
    class LungCancerDataset(Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform
            self.label_map = {label: idx for idx, label in enumerate(dataframe['major_category'].unique())}
        
        def __len__(self):
            return len(self.dataframe)
        
        def __getitem__(self, idx):
            img_path = self.dataframe.iloc[idx]['image_path']
            image = Image.open(img_path).convert('RGB')
            label_name = self.dataframe.iloc[idx]['major_category']
            label = self.label_map[label_name]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

    # Charger et diviser les données
    print("\nLoading dataset...")
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")

    # Créer les datasets et dataloaders
    train_dataset = LungCancerDataset(train_df, transform=data_transforms['train'])
    val_dataset = LungCancerDataset(val_df, transform=data_transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Charger le modèle selon l'argument et ajouter du dropout
    if model_name == 'resnet18':
        model = models.resnet18(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, NUM_CLASSES)
        )
    elif model_name == 'efficientnetb7':
        model = models.efficientnet_b7(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, NUM_CLASSES)
        )
    elif model_name == 'efficientnetb0':
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(in_features, NUM_CLASSES)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)

    # Définir la fonction de perte et l'optimiseur avec les poids de classe
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = GradScaler('cuda')

    # Fonction pour sauvegarder un checkpoint
    def save_checkpoint(epoch, model, optimizer, val_accuracy):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch + 1}: {CHECKPOINT_PATH}")

    # Fonction pour charger un checkpoint
    def load_checkpoint(model, optimizer):
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            val_accuracy = checkpoint['val_accuracy']
            print(f"Checkpoint loaded from {CHECKPOINT_PATH}, resuming from epoch {epoch + 1}")
            return epoch + 1, val_accuracy
        return 0, 0.0

    # Charger un checkpoint existant
    start_epoch, best_val_accuracy = load_checkpoint(model, optimizer)

    # Entraînement avec early stopping et accumulation de gradients
    print("Starting training...")
    patience_counter = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0) * ACCUMULATION_STEPS
        
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        save_checkpoint(epoch, model, optimizer, val_accuracy)
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved: {MODEL_SAVE_PATH} (Val Accuracy: {val_accuracy:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
        
        scheduler.step()

    print("Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return best_val_accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a lung cancer classification model")
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'efficientnetb7', 'efficientnetb0'],
                        help="Model to use: 'resnet18', 'efficientnetb7', or 'efficientnetb0'")
    parser.add_argument('--force-cpu', action='store_true', help="Force training on CPU even if GPU is available")
    parser.add_argument('--num-epochs', type=int, default=20, help="Number of epochs to train")
    args = parser.parse_args()
    
    train_model(model_name=args.model, force_cpu=args.force_cpu, num_epochs=args.num_epochs)