# High-Performance Pneumothorax Detection with EfficientNet

This project provides a complete, end-to-end pipeline for detecting pneumothorax in chest X-ray images using a fine-tuned `EfficientNetB0` model. The final model achieves significantly higher precision and accuracy compared to a ResNet50 baseline, making it a more reliable tool for medical imaging analysis.

The project includes:
- **Data Preprocessing**: Converts raw DICOM images and RLE masks into a labeled, augmented dataset.
- **Optimized Training**: A robust training script (`Efficient/train_efficientnet.py`) with class weighting to handle data imbalance.
- **In-depth Evaluation**: A script to find the optimal prediction threshold to maximize recall (`Efficient/evaluate_efficientnet.py`).
- **Explainable AI (XAI)**: Generates Grad-CAM heatmaps to visualize the model's decision-making process (`Efficient/predict_efficientnet.py`).
- **API Deployment**: A ready-to-use Flask API (`app.py`) to serve the model over a network.

---

## Project Structure
```
.
├── Efficient/                  # Contains the final, high-performance model scripts
│   ├── train_efficientnet.py
│   ├── evaluate_efficientnet.py
│   └── predict_efficientnet.py
├── data/                       # Holds the raw and preprocessed data
├── app.py                      # Flask API for model deployment
├── preprocess.py               # Initial data preprocessing script
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

---

## Quickstart: Setup and Training

### 1. Install Dependencies
Activate your virtual environment and install the required packages.
```bash
pip install -r requirements.txt
# Also install Flask for the API
pip install Flask
```

### 2. Prepare Data
This only needs to be done once. This script reads the raw data, generates labels from the RLE CSV, and saves preprocessed PNG images.
```bash
python preprocess.py
```

### 3. Train the EfficientNet Model
Navigate to the `Efficient` directory and run the training script. This will create the `pneumothorax_efficientnet_finetuned.h5` model file.
```bash
cd Efficient
python train_efficientnet.py
```

--- 

## How to Use the Deployed API

This project includes a simple Flask web server to deploy the trained model as an API.

### 1. Start the API Server
From the main project directory, run:
```bash
python app.py
```
The server will start and be accessible on your local network at `http://0.0.0.0:5000`.

### 2. Send a Prediction Request
Open a new terminal and use a tool like `curl` to send an image to the `/predict` endpoint. Replace `"path/to/your/image.png"` with the actual path to a chest X-ray image.

```bash
curl -X POST -F "file=@path/to/your/image.png" http://127.0.0.1:5000/predict
```

### Example Response
The API will return a JSON response with the model's prediction:
```json
{
  "diagnosis": "Pneumothorax",
  "prediction_score": 0.8734
}
```


3.  **Evaluate Model**
    ```bash
    python evaluate.py
    ```

4.  **Visualize with Grad-CAM**
    ```bash
    python gradcam.py
    ```

---

For details on each step, see the respective script and code comments.
