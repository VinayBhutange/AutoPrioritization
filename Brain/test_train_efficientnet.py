import pytest
import numpy as np
from train_efficientnet import create_preprocessing_pipeline

# Test for create_preprocessing_pipeline function
def test_create_preprocessing_pipeline_without_augmentation():
    """
    Tests the preprocessing pipeline without augmentation.
    """
    # 1. Create a dummy input image
    dummy_image_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # 2. Get the preprocessing function
    preprocess_fn = create_preprocessing_pipeline(use_augmentation=False)

    # 3. Process the image
    processed_image = preprocess_fn(dummy_image_rgb)

    # 4. Assertions
    # Check output shape
    assert processed_image.shape == (224, 224, 3)
    # Check if the output is preprocessed by efficientnet's preprocess_input
    # This is harder to check directly, but we can check the data type (should be float32)
    # and that the values are not in the 0-255 range anymore.
    assert processed_image.dtype == np.float32
    assert np.max(processed_image) <= 1.0
    assert np.min(processed_image) >= -1.0

def test_create_preprocessing_pipeline_with_augmentation():
    """
    Tests the preprocessing pipeline with augmentation.
    """
    # 1. Create a dummy input image
    dummy_image_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # 2. Get the preprocessing function
    preprocess_fn = create_preprocessing_pipeline(use_augmentation=True)

    # 3. Process the image
    processed_image = preprocess_fn(dummy_image_rgb)

    # 4. Assertions
    # Check output shape
    assert processed_image.shape == (224, 224, 3)
    # Check data type
    assert processed_image.dtype == np.float32
    # Check value range (post-preprocessing)
    assert np.max(processed_image) <= 1.0
    assert np.min(processed_image) >= -1.0
