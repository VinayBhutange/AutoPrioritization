import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split

# --- Configuration ---
PREPROCESSED_DIR = 'data/preprocessed'
LABELS_CSV = 'data/labels.csv'
IMG_SIZE = 256
BATCH_SIZE = 16
MODEL_PATH = 'pneumothorax_resnet50_finetuned.h5' # Using the fine-tuned model

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
    # IMPORTANT: Use the same split as training to get the correct test set
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Data Generator for the test set
    datagen = ImageDataGenerator(rescale=1./255.)
    test_gen = datagen.flow_from_dataframe(
        test_df,
        PREPROCESSED_DIR,
        x_col='filename',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='binary',
        batch_size=BATCH_SIZE,
        shuffle=False  # Important: Do not shuffle the test set
    )

    # Load the model and make predictions
    model = load_model(MODEL_PATH)
    y_pred_prob = model.predict(test_gen)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    # --- Print Classification Report ---
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=['0', '1'], output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=['0', '1'])
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(report_str)

    # --- Log results to CSV ---
    LOG_FILE = 'evaluation_log.csv'
    pneumothorax_metrics = report_dict.get('1', {}) # Get metrics for class '1' (pneumothorax)
    log_entry = {
        'timestamp': pd.Timestamp.now(),
        'model_name': MODEL_PATH,
        'accuracy': accuracy,
        'precision_pneumothorax': pneumothorax_metrics.get('precision', 0),
        'recall_pneumothorax': pneumothorax_metrics.get('recall', 0),
        'f1_score_pneumothorax': pneumothorax_metrics.get('f1-score', 0)
    }
    log_df = pd.DataFrame([log_entry])

    if os.path.exists(LOG_FILE):
        log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        log_df.to_csv(LOG_FILE, mode='w', header=True, index=False)
    
    print(f"\nResults logged to {LOG_FILE}")

    # --- Find Optimal Threshold for a Target Recall ---
    print("\n--- Finding Optimal Threshold for High Recall ---")
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob.ravel())
    
    # The 'thresholds' array is one element shorter than precision/recall
    # We append a value to thresholds to make them align with precision/recall arrays
    # This doesn't affect the result, just makes indexing easier.
    thresholds = np.append(thresholds, 1)

    # Set a target for recall (e.g., 75%)
    TARGET_RECALL = 0.75
    
    # Find all points on the curve where recall is >= our target
    high_recall_indices = np.where(recall >= TARGET_RECALL)
    
    if high_recall_indices[0].size > 0:
        # From those points, find the one with the highest precision
        best_precision_idx = np.argmax(precision[high_recall_indices])
        # Get the final index in the original array
        final_idx = high_recall_indices[0][best_precision_idx]
        
        optimal_threshold = thresholds[final_idx]
        optimal_precision = precision[final_idx]
        optimal_recall = recall[final_idx]

        print(f"To achieve a recall of at least {TARGET_RECALL*100:.0f}%, you can lower the prediction threshold.")
        print(f"Recommended Threshold: {optimal_threshold:.4f}")
        print(f"At this threshold, Precision will be ~{optimal_precision:.2f} and Recall will be ~{optimal_recall:.2f}")
        print("\nYou can use this threshold in 'predict.py' for more sensitive detection.")

        print("\n--- Verifying Performance at Recommended Threshold ---")
        y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
        print(classification_report(y_true, y_pred_optimal, target_names=['0', '1']))
    else:
        print(f"Could not find a threshold to achieve {TARGET_RECALL*100:.0f}% recall.")

if __name__ == '__main__':
    main()