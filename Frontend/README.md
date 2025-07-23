# Medical Image Prioritization System

A web application that displays and prioritizes medical images using an AI-powered analysis system. The application features an F1-style leaderboard animation for displaying real-time priority changes.

## Features

- 🖼️ **Image Display**: Shows medical images from a specified directory
- 🧠 **AI Analysis**: Integrates with an external API for image analysis and priority determination
- 🏎️ **F1-style Animations**: Dynamic position changes with smooth transitions when priorities update
- 🚦 **Color-coded Priorities**: Visual indication of urgency levels (Urgent, High, Normal)
- 📊 **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **Backend**: Python (FastAPI)
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Animations**: Custom CSS animations with Tailwind CSS

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure the image paths:
   - Edit the `IMAGE_PATHS` list in `app.py` to include the absolute paths to your directories containing medical images.
   - The application supports multiple source directories.
   - Only JPG/JPEG and PNG images are supported.

4. Configure the external API:
   - Update the `BRAIN_API_URL` and `CHEST_API_URL` variables in `app.py` to point to your prediction APIs.
   - **Note:** The APIs used in this project can be run from a local Flask server. If you are running the corresponding Flask server application(s), you can update the endpoints in `app.py` (lines 26-27) to your local server's address (e.g., `http://127.0.0.1:5001/Brain/predict`).

5. Run the application:
   ```
   python app.py
   ```
   
6. Open your browser and navigate to `http://localhost:8000`

## Key Behaviors

- **In-Memory Caching**: The application uses an in-memory cache for image lists and analysis results to improve performance. The cache is cleared each time the main page is reloaded.
- **Random Image Selection**: If more than 10 images are found in the configured paths, the application will randomly select and display 10 of them.
- **Advanced Sorting**: Analyzed images are sorted based on a multi-level key: first by priority (`Urgent` > `High` > `Normal`), and then by pushing down images with a "Normal" prediction to the bottom of their priority group.

## API Endpoints

- `GET /`: Main web interface
- `GET /api/images`: List all images in the configured directory
- `GET /api/images/{image_name}`: Get a specific image
- `GET /api/analyze`: Get analysis results for all images (includes prediction, confidence, priority)
- `GET /api/health-check`: Checks the status of the external Brain and Chest APIs

## External API Format

The application expects the external API to return responses in the following format:

```json
{
  "prediction": "Glioma",
  "confidence": 99.12,
  "priority": "Urgent"
}
```

Where `priority` is one of: "Urgent", "High", or "Normal".