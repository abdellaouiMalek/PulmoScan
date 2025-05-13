"""
generate_balanced_data_subset.py: Script pour générer un fichier CSV avec un sous-ensemble équilibré des images (5 000 pour train).
"""

import pandas as pd
import os
from sklearn.utils import shuffle

# Chemins des dossiers
BASE_PATH = "D:/PLUMOSCAN/PulmoScan/balanced_images"
SPLITS = ["train", "test", "val"]
CATEGORIES = ["Adenocarcinoma", "Other", "Squamous_Cell_Carcinoma"]
OUTPUT_CSV = "D:/PLUMOSCAN/PulmoScan/processed_images/balanced_data_subset.csv"

# Nombre cible d'images pour chaque split
TARGET_SIZES = {
    "train": 8000,  # 8 000 images pour l'entraînement
    "val": 2000,    # 2 000 images pour la validation
    "test": 2000    # 2 000 images pour le test
}

# Liste pour stocker les données
data = []

# Parcourir chaque split (train, test, val) et chaque catégorie
for split in SPLITS:
    for category in CATEGORIES:
        folder_path = os.path.join(BASE_PATH, split, category)
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}")
            continue
        
        # Lister toutes les images dans le dossier
        for img_name in os.listdir(folder_path):
            if img_name.endswith((".png", ".jpg", ".jpeg")):  # Filtrer les fichiers image
                img_path = os.path.join(folder_path, img_name)
                data.append({
                    "image_path": img_path,
                    "major_category": category,
                    "split": split
                })

# Créer un DataFrame
df = pd.DataFrame(data)
print(f"Total images found: {len(df)}")
print("Original class distribution:")
print(df.groupby(["split", "major_category"]).size())

# Créer un sous-ensemble équilibré
subset_dfs = []
for split in SPLITS:
    split_df = df[df['split'] == split]
    target_size = TARGET_SIZES[split]
    
    # Calculer le nombre d'images par classe (équilibré)
    num_classes = len(CATEGORIES)
    images_per_class = target_size // num_classes
    
    # Ajuster pour que la somme corresponde exactement à target_size
    remaining = target_size - (images_per_class * num_classes)
    class_counts = {cat: images_per_class for cat in CATEGORIES}
    for i, cat in enumerate(CATEGORIES):
        if i < remaining:
            class_counts[cat] += 1
    
    # Sélectionner un sous-ensemble pour chaque classe
    split_subset = []
    for category in CATEGORIES:
        cat_df = split_df[split_df['major_category'] == category]
        if len(cat_df) < class_counts[category]:
            print(f"Warning: Not enough images for {category} in {split} (required: {class_counts[category]}, available: {len(cat_df)})")
            sampled_df = cat_df
        else:
            sampled_df = cat_df.sample(n=class_counts[category], random_state=42)
        split_subset.append(sampled_df)
    
    # Combiner les sous-ensembles pour ce split
    split_subset_df = pd.concat(split_subset)
    subset_dfs.append(split_subset_df)

# Combiner tous les splits et mélanger
subset_df = pd.concat(subset_dfs)
subset_df = shuffle(subset_df, random_state=42)

# Afficher la nouvelle distribution
print("\nSubset class distribution:")
print(subset_df.groupby(["split", "major_category"]).size())

# Sauvegarder dans un fichier CSV
subset_df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV file saved: {OUTPUT_CSV}")