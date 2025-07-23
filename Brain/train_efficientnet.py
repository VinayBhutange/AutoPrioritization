import os
import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input # <-- Import correct preprocessor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.utils import class_weight

# --- Configuration ---
TRAIN_DIR = 'Data/Training/'
VAL_DIR = 'Data/Testing/'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
MODEL_PATH = 'brain_tumor_efficientnet_finetuned.h5' # <-- Updated model name
LOG_FILE = 'training_log.csv'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
INITIAL_LR = 1e-5 # <-- Lower learning rate for fine-tuning

# --- Pre-processing Pipeline ---
def create_preprocessing_pipeline(use_augmentation=False):
    """Creates a pipeline that applies CLAHE, optional augmentations, and then EfficientNet's required preprocessing."""
    def preprocess_fn(image):
        # 1. Apply CLAHE
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        rgb_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)

        # 2. Apply EfficientNet's specific preprocessing FIRST
        preprocessed_img = preprocess_input(rgb_img)

        # 3. Apply Augmentations if enabled on the preprocessed image
        if use_augmentation:
            # Note: Some augmentations might not work as expected on float images
            # in the range [-1, 1]. We will keep it simple for now.
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
            ])
            preprocessed_img = transform(image=preprocessed_img)['image']

        return preprocessed_img
    return preprocess_fn

# --- Main Training Script ---
def main():
    train_datagen = ImageDataGenerator(preprocessing_function=create_preprocessing_pipeline(use_augmentation=True))
    val_datagen = ImageDataGenerator(preprocessing_function=create_preprocessing_pipeline(use_augmentation=False))

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES, shuffle=True)

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES, shuffle=False)

    # --- Calculate Class Weights ---
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    train_class_weights = dict(enumerate(class_weights))
    print(f"Calculated Class Weights: {train_class_weights}")

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # --- Fine-Tuning Strategy ---
    # Freeze all layers initially
    base_model.trainable = False
    # Unfreeze from block6a onwards, similar to your sample script
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block6a_expand_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    preds = Dense(len(CLASSES), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
    csv_logger = CSVLogger(LOG_FILE)

    # --- Phase 1: Train the head ---
    print("--- Phase 1: Training the model head ---")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_gen,
        epochs=5, # Train head for 5 epochs
        validation_data=val_gen,
        class_weight=train_class_weights
    )

    # --- Phase 2: Fine-tune the model ---
    print("\n--- Phase 2: Fine-tuning the entire model ---")
    # Unfreeze the layers for fine-tuning
    base_model.trainable = True
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block6a_expand_conv':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=INITIAL_LR), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting model fine-tuning...")
    model.fit(
        train_gen, 
        epochs=EPOCHS, # Continue for the full number of epochs
        validation_data=val_gen,
        callbacks=[checkpoint, csv_logger],
        class_weight=train_class_weights
    )
    print(f"Fine-tuning complete. Best model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
