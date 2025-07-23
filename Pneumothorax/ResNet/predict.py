import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_PATH = 'pneumothorax_resnet50_finetuned.h5'
IMG_SIZE = 256

# --- Preprocessing for a single DICOM image ---
def preprocess_dicom_for_prediction(img_path):
    """Reads and preprocesses a single DICOM file for model prediction."""
    # 1. Read DICOM
    dcm = pydicom.dcmread(img_path)
    img = dcm.pixel_array

    # 2. Normalize to 8-bit for CLAHE
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    # 4. Convert to 3 channels for ResNet
    img_rgb = np.stack([img_clahe]*3, axis=-1)

    # 5. Resize to model's expected input size
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    return img_resized

# --- Grad-CAM Functions (copied from gradcam.py) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(original_img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + original_img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
def main():
    # IMPORTANT: Change this path to the DICOM image you want to test
    #DICOM_IMG_PATH = 'data/pneumothorax/dicom-images-test/1.2.276.0.7230010.3.1.2.8323329.581.1517875163.538759/1.2.276.0.7230010.3.1.3.8323329.581.1517875163.538758.dcm'
    DICOM_IMG_PATH = r"C:\CodeBase\PlayGround\Pneumothorax\data\pneumothorax\dicom-images-test\1.2.276.0.7230010.3.1.2.8323329.581.1517875163.538759\1.2.276.0.7230010.3.1.3.8323329.581.1517875163.538758\1.2.276.0.7230010.3.1.4.8323329.581.1517875163.538760.dcm"
    if not os.path.exists(DICOM_IMG_PATH):
        print(f"Error: Image not found at {DICOM_IMG_PATH}")
        print("Please update the DICOM_IMG_PATH variable in predict.py to a valid path.")
        return

    # 1. Preprocess the image
    processed_img = preprocess_dicom_for_prediction(DICOM_IMG_PATH)
    img_array = np.expand_dims(processed_img, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize

    # 2. Load model and make prediction
    model = load_model(MODEL_PATH)
    prediction = model.predict(img_array)[0][0]
    
    # 3. Display result
    print(f"Model Prediction Probability: {prediction:.4f}")
    # Use the optimal threshold found during evaluation for higher recall
    OPTIMAL_THRESHOLD = 0.0227
    if prediction > OPTIMAL_THRESHOLD:
        print("Result: Pneumothorax DETECTED")
    else:
        print("Result: Normal")

    # 4. Generate and display Grad-CAM
    print("\nGenerating Grad-CAM visualization...")
    last_conv_layer_name = 'conv5_block3_out'
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(processed_img, heatmap)

if __name__ == '__main__':
    main()
