import pandas as pd
import numpy as np
import cv2
from openslide import OpenSlide
from pathlib import Path
import os

# Chemins
INPUT_CSV = "D:/PLUMOSCAN/PulmoScan/balanced_images/balanced_data.csv"
OUTPUT_DIR = "D:/PLUMOSCAN/PulmoScan/processed_images"
OUTPUT_CSV = "D:/PLUMOSCAN/PulmoScan/processed_images/balanced_data.csv"

# Créer le répertoire de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fonction pour prétraiter les images SVS en JPG
def preprocess_svs_to_jpg(svs_path, output_dir, target_size=(224, 224)):
    output_filename = f"{Path(svs_path).stem}_{Path(svs_path).suffix.replace('.svs', '')}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        return output_path
    try:
        with OpenSlide(svs_path) as slide:
            region = slide.read_region((0, 0), 0, (1000, 1000))
            region = np.array(region.convert('RGB'))
            region = cv2.resize(region, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, cv2.cvtColor(region, cv2.COLOR_RGB2BGR))
            return output_path
    except Exception as e:
        print(f"Error processing SVS {svs_path}: {str(e)}")
        return None

# Charger balanced_data.csv
print("Loading balanced_data.csv...")
df = pd.read_csv(INPUT_CSV)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Recherche dynamique de la colonne contenant les chemins
image_col = None
processed_image_col = None
category_col = None
for col in df.columns:
    if col.lower() in ['original_filename', 'filename', 'image_path', 'file_name', 'path']:
        image_col = col
    if col.lower() in ['processed_filename', 'processed_image']:
        processed_image_col = col
    if col.lower() in ['category', 'major_category', 'label']:
        category_col = col

# Vérifier si les colonnes ont été trouvées
if image_col is None:
    raise KeyError("No column found for original image paths (expected 'original_filename', 'filename', 'image_path', etc.)")
if processed_image_col is None:
    raise KeyError("No column found for processed image names (expected 'processed_filename', etc.)")
if category_col is None:
    raise KeyError("No column found for categories (expected 'category', 'major_category', etc.)")

print(f"Found original image column: {image_col}")
print(f"Found processed image column: {processed_image_col}")
print(f"Found category column: {category_col}")

# Convertir les SVS en JPG si nécessaire
print("Preprocessing images...")
df['processed_filename'] = df[image_col].apply(
    lambda x: preprocess_svs_to_jpg(x, OUTPUT_DIR) if x.endswith('.svs') else x
)

# Ajouter le chemin complet à processed_filename
df['processed_filename'] = df['processed_filename'].apply(
    lambda x: os.path.join(OUTPUT_DIR, x) if not x.startswith(OUTPUT_DIR) else x
)

# Supprimer les lignes où le prétraitement a échoué
df = df.dropna(subset=['processed_filename'])
print(f"Dataset after preprocessing: {len(df)} samples")

# Renommer les colonnes pour train_model.py
df = df.rename(columns={image_col: 'original_filename', processed_image_col: 'image_path', category_col: 'major_category'})
print("Columns renamed to 'original_filename', 'image_path', and 'major_category'")

# Sauvegarder le fichier corrigé
df.to_csv(OUTPUT_CSV, index=False)
print(f"Corrected dataset saved to {OUTPUT_CSV}")