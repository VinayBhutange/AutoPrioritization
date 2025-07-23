import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import random

# --- Configuration ---
MODEL_PATH = 'pneumothorax_efficientnet_finetuned.h5'
PREPROCESSED_DIR = '../data/preprocessed/'
LABELS_CSV = '../data/labels.csv'
IMG_SIZE = 224
NUM_NORMAL = 8
NUM_PNEUMO = 2

# --- Image Loading ---
def get_image_paths(num_normal, num_pneumo):
    """Gets a specific number of normal and pneumothorax image paths."""
    if not os.path.exists(LABELS_CSV):
        print(f"Error: Labels file not found at {LABELS_CSV}")
        return []

    df = pd.read_csv(LABELS_CSV)
    # Get base filenames for augmented images
    df['base_filename'] = df['filename'].apply(lambda x: os.path.splitext(x)[0].replace('_aug0', '').replace('_aug1', ''))
    
    pneumo_df = df[df['label'] == 1]['base_filename'].unique()
    normal_df = df[df['label'] == 0]['base_filename'].unique()

    pneumo_samples = random.sample(list(pneumo_df), min(num_pneumo, len(pneumo_df)))
    normal_samples = random.sample(list(normal_df), min(num_normal, len(normal_df)))
    
    image_files = []
    # We'll just use the first augmentation for prediction
    for base in pneumo_samples + normal_samples:
        image_files.append(f"{base}_aug0.png")
        
    return [os.path.join(PREPROCESSED_DIR, fname) for fname in image_files]

# --- Grad-CAM Functions ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    """Generates a Grad-CAM heatmap for a given image and model."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def get_superimposed_image(original_img, heatmap, alpha=0.5):
    """Overlays the heatmap on the original image."""
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + original_img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# --- Main Execution ---
def main():
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    model = load_model(MODEL_PATH)

    print(f"Selecting {NUM_NORMAL} normal and {NUM_PNEUMO} pneumothorax images for prediction...")
    image_paths = get_image_paths(NUM_NORMAL, NUM_PNEUMO)
    if not image_paths:
        print("No images found to process. Exiting.")
        return

    results = []
    print(f"Processing {len(image_paths)} images...")
    for img_path in image_paths:
        original_img = cv2.imread(img_path)
        # Ensure image is in the correct format for display and preprocessing
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        img_resized = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array_processed = preprocess_input(img_array.copy())

        # Predict
        pred = model.predict(img_array_processed)[0][0]

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_array_processed, model)
        superimposed_img = get_superimposed_image(img_resized, heatmap)
        
        results.append({
            'path': img_path,
            'original': original_img,
            'superimposed': superimposed_img,
            'prediction': pred
        })

    # Sort results by prediction score (high to low)
    results.sort(key=lambda x: x['prediction'], reverse=True)

    # --- Text Summary ---
    print("\n--- Prediction Results (Sorted High to Low) ---")
    for i, res in enumerate(results):
        prediction_label = 'Pneumothorax' if res['prediction'] > 0.5 else 'Normal'
        print(f"{i+1}. File: {os.path.basename(res['path'])}")
        print(f"   Prediction Score: {res['prediction']:.4f} -> {prediction_label}\n")

    # --- Visualization ---
    num_results = len(results)
    # Adjusted figsize for better aspect ratio
    plt.figure(figsize=(12, 2.5 * num_results)) 
    plt.suptitle('Model Predictions with Grad-CAM (Sorted High to Low)', fontsize=16, weight='bold')

    for i, res in enumerate(results):
        # Display Original Image
        ax1 = plt.subplot(num_results, 2, 2 * i + 1)
        ax1.imshow(res['original'])
        # Use a smaller font for the title to prevent overlap
        ax1.set_title(f"Original: {os.path.basename(res['path'])}", fontsize=8)
        ax1.axis('off')

        # Display Grad-CAM Superimposed Image
        ax2 = plt.subplot(num_results, 2, 2 * i + 2)
        ax2.imshow(res['superimposed'])
        prediction_label = 'Pneumothorax' if res['prediction'] > 0.5 else 'Normal'
        ax2.set_title(f"Prediction: {res['prediction']:.3f} ({prediction_label})", fontsize=10)
        ax2.axis('off')

    # Use subplots_adjust for more reliable spacing than tight_layout
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.1)
    plt.show()

if __name__ == '__main__':
    main()

