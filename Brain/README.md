# Brain Tumor Classification using EfficientNet and Grad-CAM

## 1. Project Overview

This project implements a deep learning solution for multi-class brain tumor classification from MRI images. It leverages the power of transfer learning with the `EfficientNetB0` architecture to accurately classify tumors into four categories: `glioma`, `meningioma`, `pituitary`, and `notumor`.

The project is structured into three modular scripts for training, evaluation, and prediction. A key focus has been placed on model interpretability through Grad-CAM visualizations, which highlight the specific regions in an MRI scan that the model uses to make its predictions.

## 2. Key Features

- **High Accuracy**: Utilizes the state-of-the-art `EfficientNetB0` model with a robust two-phase fine-tuning strategy.
- **Advanced Preprocessing**: Employs Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance MRI image contrast, making features more prominent.
- **Data Augmentation**: Enriches the training dataset using a variety of augmentations (flips, rotations, brightness/contrast, Gaussian blur) to build a more robust model.
- **Class Imbalance Handling**: Calculates and applies class weights during training to prevent model bias towards more frequent classes.
- **Model Interpretability**: Integrates Grad-CAM to generate heatmaps, providing visual evidence of where the model is "looking" when it makes a prediction.
- **Prioritized Reporting**: The prediction script intelligently sorts results to prioritize potential tumor cases and presents them in a clean, readable table and a two-column plot.

## 3. Architectural Diagram

The project follows a clear pipeline from data ingestion to final prediction and visualization:

```
+-----------------------+
|   Raw MRI Image Data  |
| (Training & Testing)  |
+-----------+-----------+
            |
            v
+-----------------------+
|  CLAHE Preprocessing  |
| (Enhance Contrast)    |
+-----------+-----------+
            |
            v
+-----------------------+      +--------------------------+
|   Data Augmentation   |----->| EfficientNetB0 Preprocessing |
| (For Training Data)   |      | (Scale to Model's Needs) |
+-----------------------+      +-------------+------------+
            |                                  |
            v                                  v
+----------------------------------------------------------+
|                     EfficientNetB0 Model                 |
| (Two-Phase Fine-Tuning: Head Training -> Full Fine-Tune) |
+--------------------------+-------------------------------+
                           |
         +-----------------+-----------------+
         |                                   |
         v                                   v
+--------------------+             +--------------------------+
|   Model Evaluation |             |  Prediction & Grad-CAM   |
| (Accuracy, Report, |             | (Prioritized & Visualized)|
|  Confusion Matrix) |             +--------------------------+
+--------------------+ 
```

## 4. How to Use the Project

Follow these steps to set up and run the project.

### Step 1: Setup Environment

It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
```

### Step 2: Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

Run the training script. This will execute the two-phase training process and save the best model as `brain_tumor_efficientnet_finetuned.h5`.

```bash
python train_efficientnet.py
```

### Step 4: Evaluate the Model

After training is complete, run the evaluation script to see the model's performance metrics on the test set. This will print a classification report and save `confusion_matrix.png`.

```bash
python evaluate_efficientnet.py
```

### Step 5: Visualize Predictions

Run the prediction script to see prioritized Grad-CAM visualizations for sample test images. This will print a summary table to the console and save `gradcam_predictions_sorted.png`.

```bash
python predict_efficientnet.py
```

## 5. Results

This section should be filled in after running the evaluation and prediction scripts.

### Performance Metrics

After running `evaluate_efficientnet.py`, the model achieved the following performance on the test set:

- **Test Accuracy**: **98.02%**
- **Classification Report**:

```
              precision    recall  f1-score   support

      glioma       0.97      0.96      0.96       300
  meningioma       0.95      0.97      0.96       306
     notumor       1.00      1.00      1.00       405
   pituitary       0.99      0.99      0.99       300

    accuracy                           0.98      1311
   macro avg       0.98      0.98      0.98      1311
weighted avg       0.98      0.98      0.98      1311
```

### Confusion Matrix

The confusion matrix provides a visual breakdown of the model's performance across all classes.

![Confusion Matrix](confusion_matrix.png)

### Grad-CAM Predictions

The following image shows the model's predictions on sample images, sorted by priority. The heatmaps clearly indicate that the model is focusing on the relevant tumor regions.

![Grad-CAM Predictions](gradcam_predictions_sorted.png)
