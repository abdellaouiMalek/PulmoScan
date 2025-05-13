"""
Pipeline complet de prétraitement des données de cancer du poumon
Version Finale Complète avec analyse d'images et données cliniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import logging
import warnings
from datetime import datetime
from sklearn.utils import resample
from openslide import OpenSlide
import concurrent.futures
from typing import Tuple, Dict, Optional, List
from pathlib import Path

# Configuration initiale
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
pd.set_option('display.max_columns', 100)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"D:/PLUMOSCAN/PulmoScan/output/pipeline_log_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

# Chemins des données
IMAGE_DIRS = [
    "D:/PLUMOSCAN/PulmoScan/data/batch_3_adenocarcinoma",
    "D:/PLUMOSCAN/PulmoScan/data/batch_5_squamous",
    "D:/PLUMOSCAN/PulmoScan/data/batch_8_BAC"
]
OUTPUT_DIR = "D:/PLUMOSCAN/PulmoScan/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les données cliniques et pathologiques"""
    logging.info("Chargement des données...")
    try:
        clinical_path = "D:/PLUMOSCAN/PulmoScan/data/lung_cancer.csv"
        pathology_path = "D:/PLUMOSCAN/PulmoScan/data/Pathology_Images.csv"
        
        if not os.path.exists(clinical_path):
            raise FileNotFoundError(f"Fichier clinique introuvable: {clinical_path}")
        if not os.path.exists(pathology_path):
            raise FileNotFoundError(f"Fichier pathologique introuvable: {pathology_path}")
        
        clinical = pd.read_csv(clinical_path)
        pathology = pd.read_csv(pathology_path)
        
        # Validation basique
        assert not clinical.empty, "Données cliniques vides"
        assert not pathology.empty, "Données pathologiques vides"
        assert 'pid' in clinical.columns, "Colonne 'pid' manquante dans les données cliniques"
        assert 'pid' in pathology.columns, "Colonne 'pid' manquante dans les données pathologiques"
        
        logging.info(f"Données cliniques chargées: {clinical.shape}")
        logging.info(f"Données pathologiques chargées: {pathology.shape}")
        
        return clinical, pathology
        
    except Exception as e:
        logging.error(f"Erreur de chargement: {str(e)}")
        raise

def clean_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Nettoie les données en supprimant valeurs manquantes et doublons"""
    logging.info(f"Nettoyage des données {name}...")
    initial_shape = df.shape
    
    # Suppression colonnes avec >70% de valeurs manquantes
    df = df.loc[:, df.isnull().mean() < 0.7]
    
    # Imputation des valeurs manquantes
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        elif pd.api.types.is_string_dtype(df[col]):
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN'
            df[col] = df[col].fillna(mode_val).str.upper().str.strip()
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Suppression des doublons
    df = df.drop_duplicates()
    
    logging.info(f"Shape initiale: {initial_shape} -> Finale: {df.shape}")
    return df

def merge_datasets(clinical: pd.DataFrame, pathology: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les données cliniques et pathologiques"""
    logging.info("Fusion des datasets...")
    
    # Nettoyage des IDs
    clinical['pid'] = clinical['pid'].astype(str).str.strip()
    pathology['pid'] = pathology['pid'].astype(str).str.strip()
    
    # Suppression des doublons
    clinical = clinical.drop_duplicates(subset=['pid'])
    pathology = pathology.drop_duplicates(subset=['pid'])
    
    # Fusion avec validation
    merged = pd.merge(clinical, pathology, on='pid', how='inner', validate='one_to_one')
    
    if merged.empty:
        raise ValueError("Aucun cas commun trouvé après fusion")
    
    logging.info(f"Dataset fusionné: {merged.shape[0]} cas, {merged.shape[1]} features")
    return merged

def analyze_cancer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse et classifie les types de cancer"""
    logging.info("Analyse des types de cancer...")
    
    cancer_map = {
        "8140": "Adenocarcinoma",
        "8070": "Squamous Cell Carcinoma",
        "8046": "Neuroendocrine Carcinoma",
        "8250": "Bronchioloalveolar Carcinoma",
        "8012": "Large Cell Carcinoma",
        "8041": "Small Cell Carcinoma"
    }
    
    if 'lc_morph' not in df.columns:
        logging.warning("Colonne 'lc_morph' manquante, création de 'cancer_type' avec 'Unknown'")
        df['cancer_type'] = 'Unknown'
    else:
        df['cancer_type'] = df['lc_morph'].astype(str).str.strip().map(cancer_map).fillna("Unknown")
    
    df['major_category'] = df['cancer_type'].apply(
        lambda x: 'Adenocarcinoma' if 'adeno' in x.lower()
        else 'Squamous' if 'squamous' in x.lower()
        else 'Neuroendocrine' if 'neuro' in x.lower()
        else 'Large Cell' if 'large' in x.lower()
        else 'Small Cell' if 'small' in x.lower()
        else 'Other'
    )
    
    # Visualisation
    plt.figure(figsize=(15,6))
    df['cancer_type'].value_counts().plot(kind='bar')
    plt.title('Distribution des Types de Cancer')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cancer_type_distribution.png'))
    plt.close()
    
    logging.info("Distribution:\n" + str(df['cancer_type'].value_counts()))
    return df

def analyze_histology_subtypes(df: pd.DataFrame) -> Optional[pd.Series]:
    """Analyse la distribution des sous-types histologiques"""
    logging.info("Analyse des sous-types histologiques...")
    
    if 'roi_histology_subtype1' not in df.columns:
        logging.warning("Colonne 'roi_histology_subtype1' non trouvée")
        return None
    
    subtypes = (
        df['roi_histology_subtype1']
        .str.upper().str.strip()
        .replace('', 'UNKNOWN')
        .value_counts()
    )
    
    plt.figure(figsize=(15,6))
    subtypes.head(20).plot(kind='bar')
    plt.title('Top 20 Sous-Types Histologiques')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'histology_subtypes.png'))
    plt.close()
    
    logging.info("Top 20 sous-types:\n" + str(subtypes.head(20)))
    return subtypes

def analyze_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse les données cliniques (âge, sexe, statut tabagique, etc.)"""
    logging.info("Analyse des données cliniques...")
    
    # Recherche dynamique des colonnes cliniques
    clinical_keywords = ['age', 'sex', 'gender', 'smoking', 'smoker', 'stage', 'stag', 'grade', 'patient']
    clinical_columns = [col for col in df.columns if any(kw.lower() in col.lower() for kw in clinical_keywords)]
    available_columns = [col for col in clinical_columns if col in df.columns]
    
    if not available_columns:
        logging.warning("Aucune colonne clinique trouvée dans le dataset")
        return df
    
    # Résumé statistique
    print("\nRésumé des colonnes cliniques :")
    print(df[available_columns].describe(include='all'))
    
    # Visualisations
    for col in available_columns:
        plt.figure(figsize=(12, 6))
        if df[col].dtype in ['int64', 'float64']:
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution de {col}')
        else:
            sns.countplot(y=df[col].dropna())
            plt.title(f'Distribution de {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'clinical_{col}_distribution.png'))
        plt.close()
    
    # Valeurs manquantes
    print("\nValeurs manquantes dans les colonnes cliniques :")
    print(df[available_columns].isna().sum())
    
    logging.info("Analyse clinique terminée")
    return df

def analyze_image_data(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse approfondie des données d'images avec parallélisation"""
    logging.info("Analyse des données d'images...")
    
    if 'image_path' not in df.columns:
        raise ValueError("Colonne 'image_path' manquante pour l'analyse")
    
    # Initialisation des nouvelles colonnes
    df['image_width'] = np.nan
    df['image_height'] = np.nan
    df['level_count'] = np.nan
    
    def process_image(row):
        if pd.isna(row['image_path']) or row['image_path'] is None:
            return {'index': row.name, 'image_width': np.nan, 'image_height': np.nan, 'level_count': np.nan}
        try:
            with OpenSlide(row['image_path']) as slide:
                return {
                    'index': row.name,
                    'image_width': slide.dimensions[0],
                    'image_height': slide.dimensions[1],
                    'level_count': slide.level_count
                }
        except Exception as e:
            logging.warning(f"Erreur avec l'image {row['image_path']}: {str(e)}")
            return {'index': row.name, 'image_width': np.nan, 'image_height': np.nan, 'level_count': np.nan}
    
    # Parallélisation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, [row for _, row in df.iterrows()]))
    
    # Mise à jour du DataFrame
    for result in results:
        idx = result['index']
        df.at[idx, 'image_width'] = result['image_width']
        df.at[idx, 'image_height'] = result['image_height']
        df.at[idx, 'level_count'] = result['level_count']
    
    # Visualisation des dimensions
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df['image_width'], y=df['image_height'], hue=df['major_category'])
    plt.title('Dimensions des Images par Catégorie')
    plt.savefig(os.path.join(OUTPUT_DIR, 'image_dimensions.png'))
    plt.close()
    
    valid_images = df['image_path'].notna().sum()
    logging.info(f"Analyse d'images terminée: {valid_images} images valides")
    return df

def build_image_index() -> Dict[str, str]:
    """Construit un index des chemins d'images disponibles"""
    logging.info("Construction de l'index des images...")
    image_paths = {}
    for img_dir in IMAGE_DIRS:
        if os.path.exists(img_dir):
            for root, _, files in os.walk(img_dir):
                for f in files:
                    if f.lower().endswith(('.svs', '.tiff', '.tif')):
                        base = os.path.splitext(f)[0]
                        clean = re.sub(r'[^A-Z0-9]', '', base.upper())
                        image_paths[clean] = os.path.join(root, f)
                        image_paths[base.upper()] = os.path.join(root, f)
                        # Ajouter le nom complet du fichier
                        image_paths[f.upper()] = os.path.join(root, f)
        else:
            logging.warning(f"Répertoire d'images introuvable: {img_dir}")
    logging.info(f"Index construit: {len(image_paths)} images trouvées")
    return image_paths

def match_images(df: pd.DataFrame) -> pd.DataFrame:
    """Apparie les images aux cas cliniques"""
    logging.info("Appariement des images...")
    image_paths = build_image_index()
    
    # Trouver la colonne des noms de fichiers
    possible_cols = ['image_filename', 'filename', 'nom_image', 'image_name']
    image_col = None
    for col in possible_cols:
        if col in df.columns:
            image_col = col
            break
    if image_col is None:
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.endswith('.svs').any():
                image_col = col
                break
    if image_col is None:
        raise ValueError("Aucune colonne contenant des noms de fichiers SVS trouvée")
    
    logging.info(f"Colonne des noms de fichiers: {image_col}")
    
    def find_match(filename: str) -> Optional[str]:
        if pd.isna(filename):
            return None
        base = os.path.splitext(str(filename))[0]
        variations = [
            re.sub(r'[^A-Z0-9]', '', base.upper()),
            base.upper().replace(' ', '_'),
            base.upper().replace('-', ''),
            base.upper(),
            filename.upper()  # Nom complet du fichier
        ]
        for var in variations:
            if var in image_paths:
                return image_paths[var]
        # Recherche partielle
        for key in image_paths:
            if base.upper() in key or key in base.upper():
                return image_paths[key]
        return None
    
    df['image_path'] = df[image_col].apply(find_match)
    
    # Sauvegarder les lignes sans correspondance
    missing_images = df[df['image_path'].isna()][['pid', image_col, 'cancer_type', 'major_category']]
    if not missing_images.empty:
        missing_images.to_csv(os.path.join(OUTPUT_DIR, 'missing_images.csv'), index=False)
        logging.info(f"Fichiers manquants sauvegardés dans {os.path.join(OUTPUT_DIR, 'missing_images.csv')}")
    
    valid_images = df['image_path'].notna().sum()
    logging.info(f"Appariement: {valid_images}/{len(df)} ({valid_images/len(df):.1%})")
    return df

def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    """Valide l'existence et lisibilité des images"""
    logging.info("Validation des images...")
    
    def validate(path: str) -> bool:
        if pd.isna(path):
            return False
        try:
            with OpenSlide(path) as slide:
                return True
        except Exception as e:
            logging.warning(f"Image invalide: {path} - {str(e)}")
            return False
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(validate, df['image_path'].fillna('')))
    
    df['image_valid'] = results
    valid = df[df['image_valid']]
    logging.info(f"Images valides: {len(valid)}/{len(df)}")
    return valid

def balance_data(df: pd.DataFrame, n_samples: int = 50) -> pd.DataFrame:
    """Équilibre les classes par sur/sous-échantillonnage"""
    logging.info("Équilibrage des classes...")
    
    category_counts = df['major_category'].value_counts()
    logging.info(f"Catégories initiales:\n{category_counts}")
    
    # Avertir si une catégorie a trop peu d'échantillons
    for cat, count in category_counts.items():
        if count < n_samples // 2:
            logging.warning(f"Catégorie {cat} a seulement {count} échantillons, sur-échantillonnage intensif requis")
    
    balanced = []
    for category in category_counts.index:
        subset = df[df['major_category'] == category]
        if len(subset) == 0:
            continue
        if len(subset) > n_samples:
            balanced.append(subset.sample(n_samples, random_state=42))
        else:
            balanced.append(resample(subset, 
                                  replace=True, 
                                  n_samples=n_samples, 
                                  random_state=42))
    
    if not balanced:
        raise ValueError("Aucune donnée équilibrée générée")
    
    balanced_df = pd.concat(balanced).sample(frac=1, random_state=42)
    logging.info("Distribution équilibrée:\n" + str(balanced_df['major_category'].value_counts()))
    return balanced_df

def prepare_final_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare le dataset final pour la modélisation"""
    logging.info("Préparation du dataset final...")
    
    features = [
        'pid', 'cancer_type', 'major_category',
        'lc_grade', 'clinical_stag', 
        'roi_histology_subtype1', 'image_path',
        'image_width', 'image_height', 'level_count'
    ]
    
    final_df = df[[f for f in features if f in df.columns]].copy()
    output_path = os.path.join(OUTPUT_DIR, 'modeling_data.csv')
    final_df.to_csv(output_path, index=False)
    
    logging.info(f"Dataset final sauvegardé: {output_path} ({len(final_df)} cas)")
    return final_df

def main():
    """Point d'entrée principal du pipeline"""
    print("="*80)
    print("PIPELINE DE PRÉTRAITEMENT - CLASSIFICATION DU CANCER DU POUMON")
    print("="*80)
    
    try:
        # 1. Chargement et nettoyage
        clinical, pathology = load_data()
        clinical = clean_data(clinical, "cliniques")
        pathology = clean_data(pathology, "pathologiques")
        
        # 2. Fusion
        merged = merge_datasets(clinical, pathology)
        
        # 3. Analyse
        merged = analyze_cancer_types(merged)
        subtype_dist = analyze_histology_subtypes(merged)
        if subtype_dist is not None:
            print("\nDistribution des sous-types histologiques:")
            print(subtype_dist.head(20))
        merged = analyze_clinical_data(merged)
        
        # 4. Traitement des images
        merged = match_images(merged)
        merged = validate_images(merged)
        merged = analyze_image_data(merged)
        
        # 5. Équilibrage
        balanced = balance_data(merged)
        
        # 6. Préparation finale
        final_df = prepare_final_data(balanced)
        
        print("\n" + "="*80)
        print("PRÉTRAITEMENT TERMINÉ AVEC SUCCÈS")
        print("="*80)
        print(f"Jeu de données final: {final_df.shape}")
        print(f"Catégories: {final_df['major_category'].nunique()}")
        print(f"Images valides: {final_df['image_path'].notna().sum()}")
        print("\nStatistiques des images:")
        if 'image_width' in final_df.columns:
            print(final_df[['image_width', 'image_height', 'level_count']].describe())
        print(f"\nRésultats sauvegardés dans: {OUTPUT_DIR}")
        
    except Exception as e:
        logging.error(f"ERREUR: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()