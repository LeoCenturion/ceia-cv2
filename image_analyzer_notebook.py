# %% [markdown]
# # Image Analysis Notebook
#
# This notebook performs a series of analyses on a dataset of categorized images.
#
# ## Analyses performed:
# 1.  **File Metadata:** Image dimensions, aspect ratio, and file size.
# 2.  **Low-Level Features:** Brightness, contrast, and sharpness.
# 3.  **Texture:** Homogeneity, energy, and correlation from GLCM.
# 4.  **Color:** Average color histograms and dominant color analysis.
# 5.  **SIFT Features:** Bag of Visual Words from SIFT descriptors.
# 6.  **Classification:** XGBoost model trained on all features.

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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import MiniBatchKMeans

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


def analyze_dominant_colors(df: pd.DataFrame, n_colors: int = 3) -> pd.DataFrame:
    """Finds the N dominant colors in each image using KMeans clustering."""
    print(f"Analyzing dominant colors (top {n_colors})...")
    
    dominant_colors_data = []
    
    for path in df['path']:
        img = cv2.imread(path)
        if img is None:
            colors = {}
            for i in range(n_colors):
                colors[f'dom_color_{i+1}_r'] = None
                colors[f'dom_color_{i+1}_g'] = None
                colors[f'dom_color_{i+1}_b'] = None
            dominant_colors_data.append(colors)
            continue
            
        # Convert to RGB and reshape for KMeans
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = np.float32(img_rgb.reshape((-1, 3)))
        
        try:
            # Using KMeans to find dominant colors
            kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, n_init='auto')
            kmeans.fit(pixels)
            
            # Get the colors and sort them by prevalence
            rgb_colors = kmeans.cluster_centers_.astype(int)
            _, counts = np.unique(kmeans.labels_, return_counts=True)
            sorted_indices = np.argsort(-counts)
            sorted_rgb_colors = rgb_colors[sorted_indices]
            
            # Store as a dictionary, flattened
            colors = {}
            for i in range(n_colors):
                colors[f'dom_color_{i+1}_r'] = sorted_rgb_colors[i][0]
                colors[f'dom_color_{i+1}_g'] = sorted_rgb_colors[i][1]
                colors[f'dom_color_{i+1}_b'] = sorted_rgb_colors[i][2]
            dominant_colors_data.append(colors)
        except Exception as e:
            print(f"Could not analyze dominant colors for {path}: {e}")
            colors = {}
            for i in range(n_colors):
                colors[f'dom_color_{i+1}_r'] = None
                colors[f'dom_color_{i+1}_g'] = None
                colors[f'dom_color_{i+1}_b'] = None
            dominant_colors_data.append(colors)
            
    return df.join(pd.DataFrame(dominant_colors_data, index=df.index))


def analyze_sift_features(df: pd.DataFrame, vocabulary_size: int = 100) -> pd.DataFrame:
    """Extracts SIFT features using a bag of visual words model."""
    print(f"Analyzing SIFT features with vocabulary size {vocabulary_size}...")
    sift = cv2.SIFT_create()
    
    # 1. Extract descriptors from all images to build vocabulary
    all_descriptors = []
    
    print("Extracting SIFT descriptors for vocabulary...")
    for path in df['path']:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)

    if not all_descriptors:
        print("No SIFT descriptors found to build vocabulary.")
        sift_cols = [f'sift_{i}' for i in range(vocabulary_size)]
        return pd.DataFrame(np.nan, index=df.index, columns=sift_cols)

    all_descriptors_np = np.vstack(all_descriptors)
    
    # 2. Build vocabulary using KMeans
    print(f"Building vocabulary with {len(all_descriptors_np)} descriptors...")
    kmeans = MiniBatchKMeans(n_clusters=vocabulary_size, random_state=42, batch_size=256*4, n_init='auto')
    kmeans.fit(all_descriptors_np)
    
    # 3. Create histograms for each image
    print("Creating feature histograms for each image...")
    histograms = []
    for path in df['path']:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        hist = np.zeros(vocabulary_size)
        if img is not None:
            _, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                words = kmeans.predict(descriptors)
                for word in words:
                    hist[word] += 1
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        histograms.append(hist)
        
    sift_features_df = pd.DataFrame(histograms, index=df.index).add_prefix('sift_')
    return sift_features_df


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
# ### 5. Analyze Dominant Colors

# %%
if not df.empty:
    df_dominant_colors = analyze_dominant_colors(df.copy(), n_colors=3)
    display(df_dominant_colors.head())

# %% [markdown]
# ### 6. Plot Color Histograms

# %%
if not df.empty:
    plot_average_color_histogram(df)
    print("\nPlotting complete. Plots are displayed inline.")

# %% [markdown]
# ## 7. Model Training with XGBoost
#
# This section combines all the previously defined features (metadata, low-level, texture, and SIFT) to train a classifier. The feature extraction steps are chained together to ensure that we only train on images for which all features could be successfully extracted.
#
# The model will be trained using CUDA if a compatible GPU and XGBoost installation are available.

# %%
if not df.empty:
    print("--- Preparing data for model training ---")
    
    features_cache_path = 'all_features.csv'
    
    if os.path.exists(features_cache_path):
        print(f"Loading features from cache: {features_cache_path}")
        all_features_df = pd.read_csv(features_cache_path)
    else:
        print("Cache not found. Extracting features...")
        # 1. Chain all feature extraction steps to get a consolidated DataFrame.
        features_df = analyze_file_metadata(df.copy())
        features_df = analyze_low_level_features(features_df)
        features_df = analyze_texture(features_df)
        features_df = analyze_dominant_colors(features_df, n_colors=3)

        # 2. Extract SIFT features
        sift_features = analyze_sift_features(features_df)
        
        # 3. Combine all features
        all_features_df = features_df.join(sift_features)
        
        # Drop rows with any NaNs that might have been produced
        all_features_df.dropna(inplace=True)
        
        print(f"Saving features to cache: {features_cache_path}")
        all_features_df.to_csv(features_cache_path, index=False)
    
    print(f"\nTraining on {len(all_features_df)} images with {len(all_features_df.columns) - 2} features.")
    display(all_features_df.head())
    
    # 4. Prepare data for XGBoost
    y = all_features_df['category']
    X = all_features_df.drop(columns=['path', 'category'])

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    
    # 5. Train XGBoost model
    print("\nTraining XGBoost model with GPU...")
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), use_label_encoder=False, eval_metric='mlogloss', tree_method='gpu_hist')
    model.fit(X_train, y_train)
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    # Convert test data to DMatrix to avoid device mismatch warnings
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 7. Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
