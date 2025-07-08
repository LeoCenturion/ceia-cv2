# %% [markdown]
# # Image Analysis Notebook
#
# This notebook performs a series of analyses on a dataset of categorized images.
#
# ## Analyses performed:
# 1.  **File Metadata:** Image dimensions, aspect ratio, and file size.
# 2.  **Low-Level Features:** Brightness, contrast, and sharpness.
# 3.  **Texture:** Homogeneity, energy, and correlation from GLCM.
# 4.  **Color:** Average color histograms per category.

# %%
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from collections import Counter
from IPython.display import display

# %% [markdown]
# ## Configuration
# Set the data and output directories here.

# %%
data_dir = './tp1/data/1/dataset-resized'

# %% [markdown]
# ## Helper Functions

# %%
def get_image_paths(data_dir: str) -> pd.DataFrame:
    """Gathers image paths and their categories from the data directory."""
    records = []
    for category_dir in Path(data_dir).iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for image_path in category_dir.iterdir():
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                records.append({'path': str(image_path), 'category': category})
    return pd.DataFrame(records)

def plot_distribution(df: pd.DataFrame, column: str, title: str):
    """Plots the distribution of a given column, grouped by category."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='category', y=column)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Analysis Functions

# %%
def analyze_file_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes image dimensions and file sizes."""
    print("Analyzing file metadata (dimensions, file size)...")
    metadata = []
    for path in df['path']:
        p = Path(path)
        try:
            file_size = p.stat().st_size
            img = cv2.imread(str(p))
            if img is not None:
                height, width, _ = img.shape
                aspect_ratio = width / height
                metadata.append({
                    'width': width, 'height': height, 'aspect_ratio': aspect_ratio,
                    'file_size_kb': file_size / 1024
                })
            else:
                metadata.append({'width': None, 'height': None, 'aspect_ratio': None, 'file_size_kb': None})
        except Exception as e:
            print(f"Could not read metadata for {p}: {e}")
            metadata.append({'width': None, 'height': None, 'aspect_ratio': None, 'file_size_kb': None})
            
    return df.join(pd.DataFrame(metadata, index=df.index)).dropna()


def analyze_low_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes brightness, contrast, and sharpness."""
    print("Analyzing low-level features (brightness, contrast, sharpness)...")
    features = []
    for path in df['path']:
        try:
            img = cv2.imread(path)
            if img is not None:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = gray_img.mean()
                contrast = gray_img.std()
                sharpness = cv2.Laplacian(gray_img, cv2.CV_64F).var()
                features.append({'brightness': brightness, 'contrast': contrast, 'sharpness': sharpness})
            else:
                features.append({'brightness': None, 'contrast': None, 'sharpness': None})
        except Exception as e:
            print(f"Could not analyze low-level features for {path}: {e}")
            features.append({'brightness': None, 'contrast': None, 'sharpness': None})
            
    return df.join(pd.DataFrame(features, index=df.index)).dropna()


def analyze_texture(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes texture features using GLCM."""
    print("Analyzing texture features...")
    texture_features = []
    for path in df['path']:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                glcm = graycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                texture_features.append({'homogeneity': homogeneity, 'energy': energy, 'correlation': correlation})
            else:
                texture_features.append({'homogeneity': None, 'energy': None, 'correlation': None})
        except Exception as e:
            print(f"Could not analyze texture for {path}: {e}")
            texture_features.append({'homogeneity': None, 'energy': None, 'correlation': None})
            
    return df.join(pd.DataFrame(texture_features, index=df.index)).dropna()


def plot_average_color_histogram(df: pd.DataFrame):
    """Calculates and plots the average color histogram for each category."""
    print("Plotting average color histograms...")
    categories = df['category'].unique()
    colors = ('b', 'g', 'r')

    for category in categories:
        plt.figure(figsize=(12, 8))
        avg_hist = np.zeros((256, len(colors)))
        
        category_paths = df[df['category'] == category]['path']
        image_count = 0
        for path in category_paths:
            img = cv2.imread(path)
            if img is not None:
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                    avg_hist[:, i] += hist.flatten()
                image_count += 1
        
        if image_count > 0:
            avg_hist /= image_count
            
        for i, color in enumerate(colors):
            plt.plot(avg_hist[:, i], color=color, label=f'{color.upper()} channel')
        
        plt.title(f'Average Color Histogram for Category: {category}')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

# %% [markdown]
# ## Run Analysis
#
# ### 1. Get Image Paths

# %%
df = get_image_paths(data_dir)
if df.empty:
    print(f"No images found in {data_dir}. Exiting.")
else:
    print(f"Found {len(df)} images in {len(df['category'].unique())} categories.")
    display(df.head())

# %% [markdown]
# ### 2. Analyze and Plot Basic Metadata

# %%
if not df.empty:
    df_meta = analyze_file_metadata(df.copy())
    plot_distribution(df_meta, 'width', 'Image Width Distribution')
    plot_distribution(df_meta, 'height', 'Image Height Distribution')
    plot_distribution(df_meta, 'aspect_ratio', 'Aspect Ratio Distribution')
    plot_distribution(df_meta, 'file_size_kb', 'File Size (KB) Distribution')
    display(df_meta.head())

# %% [markdown]
# ### 3. Analyze and Plot Low-Level Visual Features

# %%
if not df.empty:
    df_low_level = analyze_low_level_features(df.copy())
    plot_distribution(df_low_level, 'brightness', 'Brightness Distribution')
    plot_distribution(df_low_level, 'contrast', 'Contrast Distribution')
    plot_distribution(df_low_level, 'sharpness', 'Sharpness (Laplacian Variance) Distribution')
    display(df_low_level.head())

# %% [markdown]
# ### 4. Analyze and Plot Texture

# %%
if not df.empty:
    df_texture = analyze_texture(df.copy())
    plot_distribution(df_texture, 'homogeneity', 'Texture Homogeneity Distribution')
    plot_distribution(df_texture, 'energy', 'Texture Energy Distribution')
    plot_distribution(df_texture, 'correlation', 'Texture Correlation Distribution')
    display(df_texture.head())

# %% [markdown]
# ### 5. Plot Color Histograms

# %%
if not df.empty:
    plot_average_color_histogram(df)
    print("\nAnalysis complete. Plots are displayed inline.")
