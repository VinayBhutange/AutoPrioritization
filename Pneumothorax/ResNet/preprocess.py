import os
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast

# --- Configuration ---
TRAIN_RLE_CSV = 'data/pneumothorax/train-rle.csv'
TRAIN_IMG_DIR = 'data/pneumothorax/dicom-images-train'
PREPROCESSED_DIR = 'data/preprocessed'
GENERATED_LABELS_CSV = 'data/labels.csv'  # Output CSV
IMG_SIZE = 256

# --- Augmentation Pipeline ---
AUGMENT = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    RandomBrightnessContrast(p=0.5)
])

# --- Preprocessing Functions ---
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def preprocess_dicom(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = apply_clahe(img)
    return img

def generate_labels_from_rle():
    """Creates labels.csv from the Kaggle RLE file."""
    df = pd.read_csv(TRAIN_RLE_CSV)
    df.columns = df.columns.str.strip() # Strip whitespace from column names
    # Group by ImageId, if any 'EncodedPixels' is not '-1', label is 1 (pneumothorax)
    df_agg = df.groupby('ImageId')['EncodedPixels'].apply(lambda x: int(any(y != ' -1' for y in x))).reset_index()
    df_agg.columns = ['ImageId', 'label']
    df_agg['filename'] = df_agg['ImageId'] + '.dcm'
    # Save only filename and label
    df_agg[['filename', 'label']].to_csv(GENERATED_LABELS_CSV, index=False)
    print(f"Generated labels file at: {GENERATED_LABELS_CSV}")
    return df_agg

# --- Main Execution ---
def main():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    # 1. Generate labels from RLE file
    labels_df = generate_labels_from_rle()
    
    # 2. Build a map of all image paths for quick lookup
    print("Building image path map...")
    image_paths = {}
    for dirname, _, filenames in os.walk(TRAIN_IMG_DIR):
        for filename in filenames:
            if filename.endswith('.dcm'):
                image_paths[filename] = os.path.join(dirname, filename)

    # 3. Process and save images
    print("Processing and augmenting images...")
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        filename = row['filename']
        img_path = image_paths.get(filename)

        if not img_path:
            # print(f"Warning: File not found in path map for {filename}") # Optional: uncomment for debugging
            continue

        img = preprocess_dicom(img_path)
        
        # Save original and one augmented version
        for i, aug in enumerate([None, AUGMENT]):
            out_img = img.copy()
            if aug:
                out_img = aug(image=out_img)['image']
            
            out_fn = f"{os.path.splitext(filename)[0]}_aug{i}.png"
            out_path = os.path.join(PREPROCESSED_DIR, out_fn)
            cv2.imwrite(out_path, out_img)
            
    print(f"\nPreprocessing complete. Preprocessed images are in: {PREPROCESSED_DIR}")

if __name__ == '__main__':
    main()
