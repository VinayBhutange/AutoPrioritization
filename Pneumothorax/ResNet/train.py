import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.utils import class_weight

# Configuration
PREPROCESSED_DIR = 'data/preprocessed'
LABELS_CSV = 'data/labels.csv'
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20 # Increased for fine-tuning
MODEL_PATH = 'pneumothorax_resnet50_finetuned.h5' # New model name

# Prepare data
def get_dataframe():
    df = pd.read_csv(LABELS_CSV)
    # Match augmented images
    imgs = []
    labels = []
    for _, row in df.iterrows():
        base = os.path.splitext(row['filename'])[0]
        for i in range(2):
            img_name = f"{base}_aug{i}.png"
            imgs.append(img_name)
            labels.append(row['label'])
    df_out = pd.DataFrame({'filename': imgs, 'label': labels})
    df_out['label'] = df_out['label'].astype(str)
    return df_out

def main():
    df = get_dataframe()
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Calculate Class Weights to handle data imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    train_class_weights = dict(enumerate(class_weights))
    print(f"Class Weights: {train_class_weights}")

    # Data generators
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_dataframe(
        train_df, PREPROCESSED_DIR, x_col='filename', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE)
    val_gen = datagen.flow_from_dataframe(
        val_df, PREPROCESSED_DIR, x_col='filename', y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE), class_mode='binary', batch_size=BATCH_SIZE)
    # Model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the top layers for fine-tuning (from conv5_block1 onwards)
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'conv5_block1_out':
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
    # Compile with a very low learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    # Training
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        class_weight=train_class_weights
    )
    print(f"Model training complete. Best model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
