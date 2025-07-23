import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Configuration ---
# Note the adjusted relative paths to the data directory, assuming this script is run from the EfficientNet folder
PREPROCESSED_DIR = '../data/preprocessed'
LABELS_CSV = '../data/labels.csv'
IMG_SIZE = 224
BATCH_SIZE = 16
MODEL_PATH = 'pneumothorax_efficientnet_finetuned.h5'

def get_dataframe():
    """Loads the labels CSV and creates entries for augmented images."""
    df = pd.read_csv(LABELS_CSV)
    imgs, labels = [], []
    for _, row in df.iterrows():
        base = os.path.splitext(row['filename'])[0]
        for i in range(2): # Original + 1 augmentation
            imgs.append(f"{base}_aug{i}.png")
            labels.append(row['label'])
    df_out = pd.DataFrame({'filename': imgs, 'label': labels})
    df_out['label'] = df_out['label'].astype(str)
    return df_out

def main():
    df = get_dataframe()
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Use EfficientNet's specific preprocessing function
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = datagen.flow_from_dataframe(
        test_df, PREPROCESSED_DIR, x_col='filename', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(MODEL_PATH)
    y_pred_prob = model.predict(test_gen).ravel()
    y_true = test_gen.classes

    print("--- Performance at Default Threshold (0.5) ---")
    y_pred_default = (y_pred_prob > 0.5).astype(int)
    print(classification_report(y_true, y_pred_default, target_names=['0', '1']))

    # --- Find and Verify Optimal Threshold ---
    print("\n--- Finding and Verifying Optimal Threshold for High Recall ---")
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    thresholds = np.append(thresholds, 1)

    TARGET_RECALL = 0.75
    high_recall_indices = np.where(recall >= TARGET_RECALL)
    
    if high_recall_indices[0].size > 0:
        best_precision_idx = np.argmax(precision[high_recall_indices])
        final_idx = high_recall_indices[0][best_precision_idx]
        
        optimal_threshold = thresholds[final_idx]
        optimal_precision = precision[final_idx]
        optimal_recall = recall[final_idx]

        print(f"Recommended Threshold for ~{TARGET_RECALL*100:.0f}% Recall: {optimal_threshold:.4f}")
        print(f"Expected Precision: ~{optimal_precision:.2f}, Expected Recall: ~{optimal_recall:.2f}")

        print("\n--- Verifying Performance at Recommended Threshold ---")
        y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
        print(classification_report(y_true, y_pred_optimal, target_names=['0', '1']))
    else:
        print(f"Could not find a threshold to achieve {TARGET_RECALL*100:.0f}% recall.")

if __name__ == '__main__':
    main()
