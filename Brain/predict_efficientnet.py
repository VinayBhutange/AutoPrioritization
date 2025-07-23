import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.efficientnet import preprocess_input # <-- Import correct preprocessor
import matplotlib.pyplot as plt
import random

# --- Configuration ---
MODEL_PATH = 'brain_tumor_efficientnet_finetuned.h5' # <-- Updated model path
TEST_DIR = 'Data/Testing/'
IMG_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_IMAGES_TO_SHOW = 4 # Show one from each class

# --- Pre-processing for Grad-CAM ---
def preprocess_for_gradcam(image):
    """Applies CLAHE and then EfficientNet's required preprocessing."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    rgb_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    return preprocess_input(rgb_img)

# --- Grad-CAM Functions ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='top_conv'):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def get_superimposed_image(original_img, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + original_img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# --- Main Prediction Script ---
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train_efficientnet.py first.")
        return

    model = load_model(MODEL_PATH)

    # --- 1. Collect and Predict on Sample Images ---
    image_paths = []
    for label in CLASSES:
        class_dir = os.path.join(TEST_DIR, label)
        if os.path.exists(class_dir) and os.listdir(class_dir):
            random_image = random.choice(os.listdir(class_dir))
            image_paths.append(os.path.join(class_dir, random_image))

    if not image_paths:
        print("No test images found. Cannot run predictions.")
        return

    results = []
    print("Running predictions on sample images...")
    for img_path in image_paths:
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img_resized = cv2.resize(original_img, IMG_SIZE)

        processed_img = preprocess_for_gradcam(original_img_resized)
        img_array = np.expand_dims(processed_img, axis=0)

        preds = model.predict(img_array)[0]
        pred_class_idx = np.argmax(preds)
        
        results.append({
            'path': img_path,
            'true_label': os.path.basename(os.path.dirname(img_path)),
            'pred_label': CLASSES[pred_class_idx],
            'confidence': preds[pred_class_idx],
            'img_array': img_array,
            'original_resized': original_img_resized
        })

    # --- 2. Sort Results by Priority (Tumors First) ---
    for res in results:
        res['is_tumor'] = 0 if res['pred_label'] == 'notumor' else 1
    results.sort(key=lambda x: (x['is_tumor'], x['confidence']), reverse=True)

    # --- 3. Display Table in Console ---
    print("\n" + "="*80)
    print("Prediction Results (Sorted by Priority: Tumors First)")
    print("="*80)
    header = f"{'Filename':<25} | {'True Label':<15} | {'Predicted Label':<15} | {'Confidence'}"
    print(header)
    print("-"*len(header))
    for res in results:
        filename = os.path.basename(res['path'])
        confidence_str = f"{res['confidence']:.2%}"
        print(f"{filename:<25} | {res['true_label']:<15} | {res['pred_label']:<15} | {confidence_str}")
    print("="*80 + "\n")

    # --- 4. Generate Sorted Plot with Grad-CAM ---
    print("Generating sorted Grad-CAM plot...")
    num_results = len(results)
    plt.figure(figsize=(10, 3 * num_results)) # Adjust size for better vertical layout
    plt.suptitle('Brain Tumor Predictions with Grad-CAM', fontsize=16, weight='bold')

    for i, res in enumerate(results):
        heatmap = make_gradcam_heatmap(res['img_array'], model)
        superimposed_img = get_superimposed_image(res['original_resized'], heatmap)

        # Plot Original Image (Left Column)
        ax1 = plt.subplot(num_results, 2, 2 * i + 1)
        ax1.imshow(res['original_resized'])
        ax1.set_title(f"Original: {os.path.basename(res['path'])}\nTrue: {res['true_label']}", fontsize=10)
        ax1.axis('off')

        # Plot Grad-CAM (Right Column)
        ax2 = plt.subplot(num_results, 2, 2 * i + 2)
        ax2.imshow(superimposed_img)
        ax2.set_title(f"Prediction: {res['pred_label']}\nConfidence: {res['confidence']:.2%}", fontsize=10)
        ax2.axis('off')

    plt.subplots_adjust(top=0.95, hspace=0.4)
    plt.savefig('gradcam_predictions_sorted.png')
    print("Sorted Grad-CAM predictions saved to gradcam_predictions_sorted.png")

if __name__ == '__main__':
    main()
