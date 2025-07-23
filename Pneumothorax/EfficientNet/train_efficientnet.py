import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# --- Configuration ---
PREPROCESSED_DIR = '../data/preprocessed'
LABELS_CSV = '../data/labels.csv'
IMG_SIZE = 224  # EfficientNetB0 is optimized for 224x224
BATCH_SIZE = 16
EPOCHS = 20 # Reverting to a known good number of epochs
MODEL_PATH = 'pneumothorax_efficientnet_finetuned.h5'
LOG_FILE = 'training_run_log.csv'
INITIAL_LR = 1e-5

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
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    train_class_weights = dict(enumerate(class_weights))
    print(f"Class Weights: {train_class_weights}")

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = datagen.flow_from_dataframe(
        train_df, PREPROCESSED_DIR, x_col='filename', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE)
    val_gen = datagen.flow_from_dataframe(
        val_df, PREPROCESSED_DIR, x_col='filename', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE)

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Simple, proven fine-tuning strategy
    set_trainable = False
    for layer in base_model.layers:
        if layer.name.startswith('block6a'):
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss='binary_crossentropy', metrics=['accuracy'])

    # Use standard, reliable Keras callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
    csv_logger = CSVLogger(LOG_FILE)

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[checkpoint, csv_logger],
        class_weight=train_class_weights
    )

if __name__ == '__main__':
    main()