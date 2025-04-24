"""
Cancer Type Data Understanding and Preprocessing

This script performs exploratory data analysis and preprocessing on the cancer type dataset.
It analyzes the structure of the data, visualizes distributions, and prepares the data for modeling.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """
    Load the cancer type datasets
    """
    print("Loading datasets...")
    try:
        clinical = pd.read_csv("../Data/type/Lung Cancer/lung_cancer.csv")
        pathology = pd.read_csv("../Data/type/Pathology Images/pathology_images.csv")
        print(f"Successfully loaded datasets:")
        print(f"  - Clinical data: {clinical.shape[0]} rows, {clinical.shape[1]} columns")
        print(f"  - Pathology data: {pathology.shape[0]} rows, {pathology.shape[1]} columns")
        return clinical, pathology
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None

def merge_datasets(clinical, pathology):
    """
    Merge clinical and pathology datasets on patient ID
    """
    print("\nMerging datasets on patient ID (pid)...")
    # Check overlap between datasets
    clinical_ids = set(clinical['pid'])
    pathology_ids = set(pathology['pid'])
    common_ids = clinical_ids.intersection(pathology_ids)

    print(f"Clinical dataset has {len(clinical_ids)} unique patients")
    print(f"Pathology dataset has {len(pathology_ids)} unique patients")
    print(f"Datasets have {len(common_ids)} patients in common")

    # Merge datasets
    merged = pd.merge(clinical, pathology, on='pid', how='inner')
    print(f"Merged dataset has {merged.shape[0]} rows and {merged.shape[1]} columns")

    return merged

def explore_data_types(df, name="Dataset"):
    """
    Explore data types and missing values
    """
    print(f"\n{name} Data Types and Missing Values:")

    # Get data types and missing values
    dtypes = df.dtypes
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    # Create a summary dataframe
    summary = pd.DataFrame({
        'Data Type': dtypes,
        'Missing Values': missing,
        'Missing Percent': missing_percent
    })

    # Sort by missing percent
    summary = summary.sort_values('Missing Percent', ascending=False)

    # Print summary for columns with missing values
    print(summary[summary['Missing Values'] > 0].head(20))

    return summary

def analyze_cancer_types(merged_df):
    """
    Analyze cancer types in the dataset
    """
    print("\nAnalyzing Cancer Types:")

    # Create cancer type mapping
    cancer_types = {
        "8140": "Adenocarcinoma",
        "8046": "Neuroendocrine Carcinoma",
        "8070": "Squamous Cell Carcinoma",
        "8250": "Bronchioloalveolar Carcinoma",
        "8249": "Atypical Carcinoid",
        "8240": "Carcinoid Tumor",
        "8041": "Small Cell Carcinoma",
        "8071": "Keratinizing Squamous Cell Carcinoma",
        "8042": "Small Cell Carcinoma, Combined",
        "8045": "Large Cell Neuroendocrine Carcinoma",
        "8010": "Carcinoma, NOS",
        "8012": "Large Cell Carcinoma, NOS",
        "8013": "Large Cell Neuroendocrine Carcinoma",
        "8252": "Lepidic Predominant Adenocarcinoma",
        "8253": "Adenocarcinoma with Mixed Subtypes",
        "8255": "Adenocarcinoma with Colloid Features",
        "8260": "Papillary Adenocarcinoma",
        "8310": "Clear Cell Adenocarcinoma",
        "8480": "Mucinous Adenocarcinoma",
        "8481": "Mucin-Producing Adenocarcinoma",
        "8550": "Acinar Cell Carcinoma",
        "8560": "Adenosquamous Carcinoma",
        "8246": "Neuroendocrine Carcinoma, NOS",
        "8000": "Neoplasm, Malignant",
        "8033": "Sarcomatoid Carcinoma",
        "8072": "Non-Keratinizing Squamous Cell Carcinoma",
        "8323": "Mixed Cell Adenocarcinoma",
        "8490": "Signet Ring Cell Carcinoma"
    }

    # Clean lc_morph codes
    merged_df['lc_morph'] = merged_df['lc_morph'].astype(str).str.strip()

    # Map codes to cancer types
    merged_df['cancer_type'] = merged_df['lc_morph'].map(cancer_types)

    # Analyze distribution of cancer types
    cancer_dist = merged_df['cancer_type'].value_counts()
    print("Distribution of Cancer Types:")
    print(cancer_dist.head(10))

    # Visualize distribution
    plt.figure(figsize=(14, 8))
    sns.countplot(y='cancer_type', data=merged_df, order=cancer_dist.index[:10])
    plt.title('Top 10 Cancer Types')
    plt.xlabel('Count')
    plt.ylabel('Cancer Type')
    plt.tight_layout()
    plt.savefig('cancer_type_distribution.png')

    # Group cancer types into major categories
    major_categories = {
        'Adenocarcinoma': ['Adenocarcinoma', 'Bronchioloalveolar Carcinoma', 'Lepidic Predominant Adenocarcinoma',
                          'Adenocarcinoma with Mixed Subtypes', 'Papillary Adenocarcinoma', 'Clear Cell Adenocarcinoma',
                          'Mucinous Adenocarcinoma', 'Mucin-Producing Adenocarcinoma', 'Acinar Cell Carcinoma',
                          'Mixed Cell Adenocarcinoma', 'Adenocarcinoma with Colloid Features', 'Signet Ring Cell Carcinoma'],
        'Squamous Cell Carcinoma': ['Squamous Cell Carcinoma', 'Keratinizing Squamous Cell Carcinoma',
                                   'Non-Keratinizing Squamous Cell Carcinoma'],
        'Neuroendocrine Carcinoma': ['Neuroendocrine Carcinoma', 'Small Cell Carcinoma', 'Carcinoid Tumor',
                                    'Atypical Carcinoid', 'Large Cell Neuroendocrine Carcinoma',
                                    'Small Cell Carcinoma, Combined', 'Neuroendocrine Carcinoma, NOS'],
        'Large Cell Carcinoma': ['Large Cell Carcinoma, NOS'],
        'Other': ['Carcinoma, NOS', 'Neoplasm, Malignant', 'Sarcomatoid Carcinoma', 'Adenosquamous Carcinoma']
    }

    # Function to map cancer type to major category
    def map_to_major_category(cancer_type):
        for category, types in major_categories.items():
            if cancer_type in types:
                return category
        return 'Other'

    # Apply mapping
    merged_df['major_category'] = merged_df['cancer_type'].apply(map_to_major_category)

    # Analyze distribution of major categories
    major_dist = merged_df['major_category'].value_counts()
    print("\nDistribution of Major Cancer Categories:")
    print(major_dist)

    # Visualize major categories
    plt.figure(figsize=(12, 6))
    sns.countplot(x='major_category', data=merged_df, order=major_dist.index)
    plt.title('Distribution of Major Cancer Categories')
    plt.xlabel('Cancer Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('major_category_distribution.png')

    return merged_df

def analyze_histology_subtypes(merged_df):
    """
    Analyze histology subtypes from ROI data
    """
    print("\nAnalyzing Histology Subtypes from ROI data:")

    # Extract histology subtype columns
    subtype_cols = [col for col in merged_df.columns if 'roi_histology_subtype' in col]

    # Count non-null values in each column
    subtype_counts = {col: merged_df[col].count() for col in subtype_cols}
    print("Number of non-null values in each histology subtype column:")
    for col, count in subtype_counts.items():
        print(f"  - {col}: {count}")

    # Analyze most common subtypes in first ROI
    subtype1_dist = merged_df['roi_histology_subtype1'].value_counts()
    print("\nMost common histology subtypes in first ROI:")
    print(subtype1_dist.head(10))

    # Visualize distribution of subtypes
    plt.figure(figsize=(14, 8))
    sns.countplot(y='roi_histology_subtype1', data=merged_df, order=subtype1_dist.index[:10])
    plt.title('Top 10 Histology Subtypes (ROI 1)')
    plt.xlabel('Count')
    plt.ylabel('Histology Subtype')
    plt.tight_layout()
    plt.savefig('histology_subtype_distribution.png')

    return subtype1_dist

def analyze_image_data(merged_df):
    """
    Analyze image data characteristics
    """
    print("\nAnalyzing Image Data:")

    # Check image filename format
    print("Image filename examples:")
    print(merged_df['image_filename'].head(5))

    # Analyze image file sizes
    if 'image_filesize' in merged_df.columns:
        # Convert to MB
        merged_df['image_size_mb'] = merged_df['image_filesize'] / (1024 * 1024)

        print(f"\nImage file size statistics (MB):")
        print(f"  - Min: {merged_df['image_size_mb'].min():.2f}")
        print(f"  - Max: {merged_df['image_size_mb'].max():.2f}")
        print(f"  - Mean: {merged_df['image_size_mb'].mean():.2f}")
        print(f"  - Median: {merged_df['image_size_mb'].median():.2f}")

        # Visualize image size distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(merged_df['image_size_mb'], bins=50, kde=True)
        plt.title('Distribution of Image File Sizes')
        plt.xlabel('File Size (MB)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('image_size_distribution.png')

    # Check if images directory exists
    image_dir = "../Data/type/Pathology Images/images"
    if os.path.exists(image_dir):
        print(f"\nImage directory exists: {image_dir}")
        # Count image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.svs')]
        print(f"Found {len(image_files)} .svs image files")

        # Check overlap with dataset
        dataset_images = set(merged_df['image_filename'])
        image_files_set = set(image_files)
        overlap = dataset_images.intersection(image_files_set)
        print(f"Dataset contains {len(dataset_images)} unique image filenames")
        print(f"Images directory contains {len(image_files_set)} image files")
        print(f"Overlap between dataset and images directory: {len(overlap)} files")
    else:
        print(f"\nImage directory not found: {image_dir}")

    return merged_df

def analyze_clinical_data(merged_df):
    """
    Analyze clinical data characteristics
    """
    print("\nAnalyzing Clinical Data:")

    # Analyze staging information
    stage_cols = [col for col in merged_df.columns if 'stag' in col]

    for col in stage_cols:
        if col in merged_df.columns:
            print(f"\nDistribution of {col}:")
            stage_dist = merged_df[col].value_counts()
            print(stage_dist.head(10))

    # Analyze tumor grade
    if 'lc_grade' in merged_df.columns:
        print("\nDistribution of tumor grade (lc_grade):")
        grade_dist = merged_df['lc_grade'].value_counts()
        print(grade_dist)

        # Visualize grade distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='lc_grade', data=merged_df)
        plt.title('Distribution of Tumor Grades')
        plt.xlabel('Grade')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('grade_distribution.png')

    # Analyze relationship between cancer type and stage
    if 'cancer_type' in merged_df.columns and 'clinical_stag' in merged_df.columns:
        print("\nRelationship between cancer type and clinical stage:")
        cross_tab = pd.crosstab(merged_df['major_category'], merged_df['clinical_stag'])
        print(cross_tab)

        # Visualize relationship
        plt.figure(figsize=(14, 8))
        sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Cancer Type vs Clinical Stage')
        plt.xlabel('Clinical Stage')
        plt.ylabel('Cancer Type')
        plt.tight_layout()
        plt.savefig('cancer_type_vs_stage.png')

    return merged_df

def prepare_data_for_modeling(merged_df):
    """
    Prepare data for modeling by creating a clean dataset with relevant features
    """
    print("\nPreparing Data for Modeling:")

    # Select relevant columns
    relevant_cols = [
        'pid', 'image_filename', 'cancer_type', 'major_category',
        'lc_grade', 'clinical_stag', 'clinical_t', 'clinical_n', 'clinical_m',
        'roi_histology_subtype1', 'block_worst_grade'
    ]

    # Filter columns that exist in the dataframe
    modeling_cols = [col for col in relevant_cols if col in merged_df.columns]

    # Create modeling dataset
    modeling_df = merged_df[modeling_cols].copy()

    # Handle missing values
    for col in modeling_df.columns:
        missing = modeling_df[col].isnull().sum()
        if missing > 0:
            print(f"Column {col} has {missing} missing values ({missing/len(modeling_df)*100:.2f}%)")

    # Fill missing values or drop rows with missing values in key columns
    key_cols = ['cancer_type', 'major_category', 'image_filename']
    key_cols = [col for col in key_cols if col in modeling_df.columns]

    modeling_df = modeling_df.dropna(subset=key_cols)
    print(f"After dropping rows with missing values in key columns: {modeling_df.shape[0]} rows")

    # Save prepared dataset
    modeling_df.to_csv('cancer_type_modeling_data.csv', index=False)
    print(f"Saved prepared dataset to 'cancer_type_modeling_data.csv'")

    return modeling_df

def main():
    """
    Main function to run the data understanding and preprocessing pipeline
    """
    print("=" * 80)
    print("CANCER TYPE DATA UNDERSTANDING AND PREPROCESSING")
    print("=" * 80)

    # Load data
    clinical, pathology = load_data()
    if clinical is None or pathology is None:
        print("Error: Could not load datasets. Exiting.")
        return

    # Explore data types and missing values
    explore_data_types(clinical, "Clinical")
    explore_data_types(pathology, "Pathology")

    # Merge datasets
    merged_df = merge_datasets(clinical, pathology)

    # Analyze cancer types
    merged_df = analyze_cancer_types(merged_df)

    # Analyze histology subtypes
    analyze_histology_subtypes(merged_df)

    # Analyze image data
    merged_df = analyze_image_data(merged_df)

    # Analyze clinical data
    merged_df = analyze_clinical_data(merged_df)

    # Prepare data for modeling
    modeling_df = prepare_data_for_modeling(merged_df)

    print("\n" + "=" * 80)
    print("DATA UNDERSTANDING AND PREPROCESSING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()