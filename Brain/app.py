import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
# Make sure the model file is in the same directory as this script,
# or provide the full path.
MODEL_PATH = 'brain_tumor_efficientnet_finetuned.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    # If you have custom objects, you might need to load them like this:
    # from tensorflow.keras.utils import custom_object_scope
    # with custom_object_scope({'CustomLayer': CustomLayer}):
    #     model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Error loading model: {e}")
    model = None

# --- ASSUMPTIONS ---
# You may need to change these values based on your model's training configuration.

# 1. Image dimensions
# EfficientNet models have specific input sizes. B0 is 224x224, B1 is 240x240, etc.
# We'll assume 224x224 here.
IMG_WIDTH, IMG_HEIGHT = 224, 224

# 2. Class labels
# The order of these labels must match the output of your model's softmax layer.
# A common brain tumor dataset has these four classes.
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- END OF ASSUMPTIONS ---

def preprocess_image(image_file):
    """Preprocesses the image for model prediction."""
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch

    # EfficientNet models usually require this preprocessing step.
    # If you used a different one, please update this line.
    processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return processed_img

@app.route('/Brain/predict', methods=['POST'])
def predict():
    """API endpoint for prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            processed_image = preprocess_image(file)
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_label = CLASS_LABELS[predicted_class_index]
            confidence = float(np.max(predictions[0]))

            score = confidence * 100  # Convert confidence to percentage
            priority = "Unknown"  # Default priority

            prediction_lower = predicted_class_label.lower()

            if prediction_lower in ["glioma", "meningioma", "pituitary"]:
                if score > 90:
                    priority = "Urgent"
                elif 80 < score <= 90:
                    priority = "High"
                elif 70 < score <= 80:
                    priority = "Medium"
                else:  # Covers scores <= 70
                    priority = "Low"
            elif prediction_lower == "no tumor":
                if score > 80:
                    priority = "Low"
                else:
                    priority = "N/A"

            return jsonify({
                'prediction': predicted_class_label,
                'confidence': score,
                'priority': priority
            })
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    return "<h1>Brain Tumor Classification API</h1><p>Send a POST request to /Brain/predict with an image file.</p>"

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from your network
    app.run(host='0.0.0.0', port=5001, debug=True)
