import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Configuration ---
# Path is relative to where app.py is run from
MODEL_PATH = 'Efficient/pneumothorax_efficientnet_finetuned.h5'
IMG_SIZE = 224

# --- Initialize Flask App and Load Model ---
app = Flask(__name__)

print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print("Error: Model file not found. Please ensure the model exists.")
    model = None
else:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

def get_priority(score):
    """Determines the priority level based on the prediction score."""
    # Note: The prediction score is a float between 0 and 1.
    if score > 0.90:
        return 'Urgent'
    elif score > 0.80:
        return 'High'
    elif score > 0.50:
        return 'Medium'
    else:
        return 'Low'

def preprocess_image(image_bytes):
    """Reads image bytes, preprocesses, and prepares for the model."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize and expand dimensions
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)
    # Apply EfficientNet's specific preprocessing
    return preprocess_input(img_array)

@app.route('/Chest/predict', methods=['POST'])
def predict():
    """Receives an image, preprocesses it, and returns a prediction."""
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Get prediction
        prediction_score = model.predict(processed_image)[0][0]
        
        # Determine diagnosis (using a standard 0.5 threshold for the API)
        diagnosis = 'Pneumothorax' if prediction_score > 0.5 else 'Normal'
        priority = get_priority(prediction_score)

        return jsonify({
            'prediction': diagnosis,
            'confidence': round(float(prediction_score) * 100, 2),
            'priority': priority
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    # Runs the Flask app and makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
