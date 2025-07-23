import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration ---
MODEL_PATH = 'pneumothorax_resnet50_finetuned.h5'
PREPROCESSED_DIR = 'data/preprocessed'
LABELS_CSV = 'data/labels.csv'
IMG_SIZE = 256
BATCH_SIZE = 16

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
    """Main function to load data, get predictions, and plot performance curves."""
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = get_dataframe()
    # Use the same split as training/evaluation to ensure consistency
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    datagen = ImageDataGenerator(rescale=1./255.)
    test_gen = datagen.flow_from_dataframe(
        test_df,
        PREPROCESSED_DIR,
        x_col='filename',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='binary',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Load model and get prediction probabilities
    print("Loading model and making predictions...")
    model = load_model(MODEL_PATH)
    y_pred_prob = model.predict(test_gen).ravel() # Flatten to 1D array
    y_true = test_gen.classes

    # --- 1. Precision-Recall Curve ---
    print("Generating Precision-Recall Curve...")
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
    
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Pneumothorax Detection')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall_curve.png'))
    print(f"Saved precision_recall_curve.png to {PLOTS_DIR}/")

    # --- 2. ROC Curve and AUC ---
    print("Generating ROC Curve...")
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
    print(f"Saved roc_curve.png to {PLOTS_DIR}/")

    plt.show()

if __name__ == '__main__':
    main()
