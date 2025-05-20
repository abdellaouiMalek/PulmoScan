"""
Cancer Type Data Visualization

This module provides functions for visualizing cancer type data, including:
- Pathology image visualization
- Cancer type distribution visualization
- Feature correlation visualization
- Augmentation visualization

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set plot style
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def visualize_cancer_distribution(
    df,
    column="major_category",
    title="Cancer Type Distribution",
    save_path="cancer_distribution.png",
):
    """
    Visualize the distribution of cancer types

    Args:
        df: DataFrame containing cancer type data
        column: Column to visualize (default: 'major_category')
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    # Count values
    value_counts = df[column].value_counts()

    # Create bar plot
    ax = sns.barplot(x=value_counts.index, y=value_counts.values)

    # Add value labels on top of bars
    for i, v in enumerate(value_counts.values):
        ax.text(i, v + 5, str(v), ha="center")

    # Set labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(column.replace("_", " ").title(), fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)
    plt.show()


def visualize_feature_correlation(
    df,
    features=None,
    title="Feature Correlation Matrix",
    save_path="feature_correlation.png",
):
    """
    Visualize correlation between features

    Args:
        df: DataFrame containing features
        features: List of features to include (default: None, uses all numeric columns)
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Select numeric columns if features not specified
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlation matrix
    corr_matrix = df[features].corr()

    # Create heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.5},
    )

    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_pathology_image(
    image_path, title="Pathology Image", save_path="pathology_image.png"
):
    """
    Visualize a pathology image

    Args:
        image_path: Path to the image file
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    try:
        # Load image
        img = Image.open(image_path)

        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(title, fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)
        plt.show()

        return img
    except Exception as e:
        print(f"Error visualizing image {image_path}: {e}")
        return None


def visualize_image_grid(
    image_paths,
    titles=None,
    cols=3,
    figsize=(15, 15),
    title="Pathology Image Grid",
    save_path="image_grid.png",
):
    """
    Visualize a grid of pathology images

    Args:
        image_paths: List of paths to image files
        titles: List of titles for each image (default: None)
        cols: Number of columns in the grid (default: 3)
        figsize: Figure size (default: (15, 15))
        title: Main title for the grid
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Calculate number of rows needed
    rows = (len(image_paths) + cols - 1) // cols

    # Create figure
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, y=0.95)

    # Add images to grid
    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            img = Image.open(image_path)

            # Add subplot
            ax = fig.add_subplot(rows, cols, i + 1)

            # Display image
            ax.imshow(img)

            # Set title if provided
            if titles is not None and i < len(titles):
                ax.set_title(titles[i])

            # Turn off axis
            ax.axis("off")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save figure
    plt.savefig(save_path)
    plt.show()


def visualize_augmentation_examples(
    original_image_path,
    augmentation_func,
    n_augmentations=5,
    title="Augmentation Examples",
    save_path="augmentation_examples.png",
):
    """
    Visualize examples of data augmentation

    Args:
        original_image_path: Path to the original image
        augmentation_func: Function that takes an image and returns an augmented image
        n_augmentations: Number of augmentation examples to show (default: 5)
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    try:
        # Load original image
        original_img = Image.open(original_image_path)

        # Create augmented images
        augmented_images = [original_img]
        for _ in range(n_augmentations):
            augmented_img = augmentation_func(original_img)
            augmented_images.append(augmented_img)

        # Create figure
        fig, axes = plt.subplots(
            1, n_augmentations + 1, figsize=(3 * (n_augmentations + 1), 4)
        )

        # Display original image
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Display augmented images
        for i in range(n_augmentations):
            axes[i + 1].imshow(augmented_images[i + 1])
            axes[i + 1].set_title(f"Augmentation {i + 1}")
            axes[i + 1].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)
        plt.show()
    except Exception as e:
        print(f"Error visualizing augmentation examples: {e}")


def visualize_class_examples(
    df,
    image_dir,
    category_col="major_category",
    n_examples=3,
    figsize=(15, 20),
    title="Examples by Class",
    save_path="class_examples.png",
):
    """
    Visualize examples of images from each class

    Args:
        df: DataFrame containing image information
        image_dir: Directory containing images
        category_col: Column containing category labels (default: 'major_category')
        n_examples: Number of examples per class (default: 3)
        figsize: Figure size (default: (15, 20))
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Get unique categories
    categories = df[category_col].unique()

    # Create figure
    fig, axes = plt.subplots(len(categories), n_examples, figsize=figsize)

    # Iterate over categories
    for i, category in enumerate(categories):
        # Get examples for this category
        category_df = df[df[category_col] == category]

        # Get random examples
        if len(category_df) > 0:
            examples = category_df.sample(min(n_examples, len(category_df)))

            # Iterate over examples
            for j, (_, row) in enumerate(examples.iterrows()):
                if j < n_examples:
                    # Get image path
                    if "processed_filename" in row:
                        image_path = os.path.join(image_dir, row["processed_filename"])
                    elif "image_filename" in row:
                        image_path = os.path.join(image_dir, row["image_filename"])
                    else:
                        print(f"No image filename found for {category} example {j}")
                        continue

                    # Check if file exists
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        continue

                    # Load image
                    try:
                        img = Image.open(image_path)

                        # Display image
                        if len(categories) == 1:
                            ax = axes[j]
                        else:
                            ax = axes[i, j]

                        ax.imshow(img)
                        ax.set_title(f"{category}")
                        ax.axis("off")
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)
    plt.show()


def visualize_tsne_embedding(
    features,
    labels,
    title="t-SNE Embedding of Features",
    save_path="tsne_embedding.png",
):
    """
    Visualize t-SNE embedding of features

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Labels for each sample
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    # Create DataFrame for plotting
    tsne_df = pd.DataFrame(
        {"x": tsne_result[:, 0], "y": tsne_result[:, 1], "label": labels}
    )

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot points
    sns.scatterplot(
        x="x", y="y", hue="label", data=tsne_df, palette="viridis", s=100, alpha=0.7
    )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)
    plt.show()


def visualize_image_with_overlay(
    image_path,
    mask_path=None,
    alpha=0.5,
    title="Image with Overlay",
    save_path="image_with_overlay.png",
):
    """
    Visualize an image with an overlay mask

    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file (default: None)
        alpha: Transparency of the overlay (default: 0.5)
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    try:
        # Load image
        img = np.array(Image.open(image_path))

        # Create figure
        plt.figure(figsize=(10, 10))

        # Display image
        plt.imshow(img)

        # Add overlay if mask provided
        if mask_path is not None:
            mask = np.array(Image.open(mask_path))
            plt.imshow(mask, alpha=alpha, cmap="jet")

        plt.title(title, fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)
        plt.show()
    except Exception as e:
        print(f"Error visualizing image with overlay: {e}")


def visualize_roi_extraction(
    slide_path,
    coords,
    level=0,
    tile_size=1024,
    title="ROI Extraction",
    save_path="roi_extraction.png",
):
    """
    Visualize ROI extraction from a slide

    Args:
        slide_path: Path to the slide file
        coords: List of (x, y) coordinates for ROIs
        level: Magnification level (default: 0)
        tile_size: Size of tiles to extract (default: 1024)
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    try:
        # Import openslide
        import openslide
        from openslide import OpenSlide

        # Load slide
        slide = OpenSlide(slide_path)

        # Get slide dimensions
        width, height = slide.dimensions

        # Create a thumbnail for visualization
        thumbnail_size = 1000
        scale_factor = max(width, height) / thumbnail_size
        thumb_w, thumb_h = int(width / scale_factor), int(height / scale_factor)
        thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
        thumbnail_np = np.array(thumbnail)

        # Create figure
        plt.figure(figsize=(12, 12))

        # Display thumbnail
        plt.imshow(thumbnail_np)

        # Draw ROI boxes
        for x, y in coords:
            # Scale coordinates to thumbnail size
            x_thumb = int(x / scale_factor)
            y_thumb = int(y / scale_factor)
            w_thumb = int(tile_size / scale_factor)
            h_thumb = int(tile_size / scale_factor)

            # Draw rectangle
            rect = plt.Rectangle(
                (x_thumb, y_thumb),
                w_thumb,
                h_thumb,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

        plt.title(title, fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)
        plt.show()
    except Exception as e:
        print(f"Error visualizing ROI extraction: {e}")


def visualize_augmentation_comparison(
    original_image_path,
    augmentation_funcs,
    func_names,
    title="Augmentation Comparison",
    save_path="augmentation_comparison.png",
):
    """
    Compare different augmentation techniques

    Args:
        original_image_path: Path to the original image
        augmentation_funcs: List of augmentation functions
        func_names: List of names for the augmentation functions
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    try:
        # Load original image
        original_img = Image.open(original_image_path)

        # Create figure
        fig, axes = plt.subplots(
            1,
            len(augmentation_funcs) + 1,
            figsize=(3 * (len(augmentation_funcs) + 1), 4),
        )

        # Display original image
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Apply and display each augmentation
        for i, (func, name) in enumerate(zip(augmentation_funcs, func_names)):
            augmented_img = func(original_img)
            axes[i + 1].imshow(augmented_img)
            axes[i + 1].set_title(name)
            axes[i + 1].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)
        plt.show()
    except Exception as e:
        print(f"Error visualizing augmentation comparison: {e}")


def visualize_preprocessing_steps(
    image_path,
    preprocessing_funcs,
    func_names,
    title="Preprocessing Steps",
    save_path="preprocessing_steps.png",
):
    """
    Visualize preprocessing steps

    Args:
        image_path: Path to the image file
        preprocessing_funcs: List of preprocessing functions
        func_names: List of names for the preprocessing functions
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    try:
        # Load image
        img = Image.open(image_path)

        # Create figure
        fig, axes = plt.subplots(
            1,
            len(preprocessing_funcs) + 1,
            figsize=(3 * (len(preprocessing_funcs) + 1), 4),
        )

        # Display original image
        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Apply and display each preprocessing step
        current_img = img
        for i, (func, name) in enumerate(zip(preprocessing_funcs, func_names)):
            processed_img = func(current_img)
            current_img = processed_img
            axes[i + 1].imshow(processed_img)
            axes[i + 1].set_title(name)
            axes[i + 1].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path)
        plt.show()
    except Exception as e:
        print(f"Error visualizing preprocessing steps: {e}")


def visualize_class_balance(
    df,
    category_col="major_category",
    split_col="split",
    title="Class Balance by Split",
    save_path="class_balance.png",
):
    """
    Visualize class balance across train/val/test splits

    Args:
        df: DataFrame containing category and split information
        category_col: Column containing category labels (default: 'major_category')
        split_col: Column containing split labels (default: 'split')
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    # Create cross-tabulation
    cross_tab = pd.crosstab(df[category_col], df[split_col])

    # Calculate percentages
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=0), axis=1) * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot counts
    cross_tab.plot(kind="bar", ax=ax1)
    ax1.set_title("Class Counts by Split", fontsize=14)
    ax1.set_xlabel(category_col.replace("_", " ").title(), fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.legend(title="Split")

    # Plot percentages
    cross_tab_pct.plot(kind="bar", ax=ax2)
    ax2.set_title("Class Percentages by Split", fontsize=14)
    ax2.set_xlabel(category_col.replace("_", " ").title(), fontsize=12)
    ax2.set_ylabel("Percentage (%)", fontsize=12)
    ax2.legend(title="Split")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)
    plt.show()

    return cross_tab, cross_tab_pct


def visualize_image_statistics(
    image_dir,
    pattern="*.png",
    max_images=100,
    title="Image Statistics",
    save_path="image_statistics.png",
):
    """
    Visualize statistics of images in a directory

    Args:
        image_dir: Directory containing images
        pattern: Pattern to match image files (default: '*.png')
        max_images: Maximum number of images to analyze (default: 100)
        title: Plot title
        save_path: Path to save the visualization

    Returns:
        None
    """
    import glob

    # Get image files
    image_files = glob.glob(os.path.join(image_dir, pattern))

    # Limit number of images
    if len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)

    # Initialize lists for statistics
    widths = []
    heights = []
    aspect_ratios = []
    mean_intensities = []
    std_intensities = []

    # Process images
    for image_file in tqdm(image_files, desc="Analyzing images"):
        try:
            # Load image
            img = Image.open(image_file)
            img_array = np.array(img)

            # Get dimensions
            width, height = img.size
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)

            # Get intensity statistics
            if len(img_array.shape) == 3:  # Color image
                mean_intensity = np.mean(img_array) / 255.0
                std_intensity = np.std(img_array) / 255.0
            else:  # Grayscale image
                mean_intensity = np.mean(img_array) / 255.0
                std_intensity = np.std(img_array) / 255.0

            mean_intensities.append(mean_intensity)
            std_intensities.append(std_intensity)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot width distribution
    sns.histplot(widths, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Width Distribution", fontsize=14)
    axes[0, 0].set_xlabel("Width (pixels)", fontsize=12)
    axes[0, 0].set_ylabel("Count", fontsize=12)

    # Plot height distribution
    sns.histplot(heights, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Height Distribution", fontsize=14)
    axes[0, 1].set_xlabel("Height (pixels)", fontsize=12)
    axes[0, 1].set_ylabel("Count", fontsize=12)

    # Plot aspect ratio distribution
    sns.histplot(aspect_ratios, kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Aspect Ratio Distribution", fontsize=14)
    axes[0, 2].set_xlabel("Aspect Ratio (width/height)", fontsize=12)
    axes[0, 2].set_ylabel("Count", fontsize=12)

    # Plot mean intensity distribution
    sns.histplot(mean_intensities, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Mean Intensity Distribution", fontsize=14)
    axes[1, 0].set_xlabel("Mean Intensity (normalized)", fontsize=12)
    axes[1, 0].set_ylabel("Count", fontsize=12)

    # Plot standard deviation distribution
    sns.histplot(std_intensities, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Standard Deviation Distribution", fontsize=14)
    axes[1, 1].set_xlabel("Standard Deviation (normalized)", fontsize=12)
    axes[1, 1].set_ylabel("Count", fontsize=12)

    # Plot mean vs std scatter
    axes[1, 2].scatter(mean_intensities, std_intensities, alpha=0.5)
    axes[1, 2].set_title("Mean vs. Standard Deviation", fontsize=14)
    axes[1, 2].set_xlabel("Mean Intensity (normalized)", fontsize=12)
    axes[1, 2].set_ylabel("Standard Deviation (normalized)", fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)
    plt.show()

    # Return statistics
    return {
        "width": {
            "mean": np.mean(widths),
            "std": np.std(widths),
            "min": np.min(widths),
            "max": np.max(widths),
        },
        "height": {
            "mean": np.mean(heights),
            "std": np.std(heights),
            "min": np.min(heights),
            "max": np.max(heights),
        },
        "aspect_ratio": {
            "mean": np.mean(aspect_ratios),
            "std": np.std(aspect_ratios),
            "min": np.min(aspect_ratios),
            "max": np.max(aspect_ratios),
        },
        "mean_intensity": {
            "mean": np.mean(mean_intensities),
            "std": np.std(mean_intensities),
            "min": np.min(mean_intensities),
            "max": np.max(mean_intensities),
        },
        "std_intensity": {
            "mean": np.mean(std_intensities),
            "std": np.std(std_intensities),
            "min": np.min(std_intensities),
            "max": np.max(std_intensities),
        },
    }
