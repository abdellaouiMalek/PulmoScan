"""
generate_balanced_data.py: Script pour générer un fichier CSV avec toutes les images.
"""

import pandas as pd
import os
from pathlib import Path

# Chemins des dossiers
BASE_PATH = "D:/PLUMOSCAN/PulmoScan/balanced_images"
SPLITS = ["train", "test", "val"]
CATEGORIES = ["Adenocarcinoma", "Other", "Squamous_Cell_Carcinoma"]
OUTPUT_CSV = "D:/PLUMOSCAN/PulmoScan/processed_images/balanced_data.csv"

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
print("Class distribution:")
print(df.groupby(["split", "major_category"]).size())

# Sauvegarder dans un fichier CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV file saved: {OUTPUT_CSV}")