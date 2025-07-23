import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input # <-- Import correct preprocessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
TEST_DIR = 'Data/Testing/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
MODEL_PATH = 'brain_tumor_efficientnet_finetuned.h5' # <-- Updated model path
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Pre-processing Pipeline for Evaluation ---
def create_eval_preprocess_fn():
    """Creates a pipeline that applies CLAHE and then EfficientNet's required preprocessing."""
    def preprocess_fn(image):
        # 1. Apply CLAHE
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        rgb_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
        
        # 2. Apply EfficientNet's specific preprocessing
        return preprocess_input(rgb_img)
    return preprocess_fn

# --- Main Evaluation Script ---
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train_efficientnet.py first.")
        return

    # Preprocessing is not part of the model graph, so no custom_objects needed
    model = load_model(MODEL_PATH)

    eval_preprocessor = create_eval_preprocess_fn()
    test_datagen = ImageDataGenerator(preprocessing_function=eval_preprocessor)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES, shuffle=False)

    print("Evaluating model performance on the test set...")
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    print("\nGenerating classification report and confusion matrix...")
    y_pred_probs = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=CLASSES))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    # plt.show() # Uncomment to display the plot

if __name__ == '__main__':
    main()
