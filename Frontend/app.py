import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import glob
import json
import random
import base64
from pathlib import Path

# In-memory cache for images
image_cache = None
analysis_cache = None

app = FastAPI(title="Medical Image Prioritization System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# External API URLs
BRAIN_API_URL = "http://130.141.146.84:5001/Brain/predict"
CHEST_API_URL = "http://130.141.146.84:5000/Chest/predict"

# Health check URLs (simplify by checking just the base URLs)
BRAIN_API_HEALTH_URL = "http://130.141.146.84:5001"
CHEST_API_HEALTH_URL = "http://130.141.146.84:5000"

# Hardcoded paths for images
IMAGE_PATHS = [
    "C:\\Users\\320210377\\Desktop\\TEMP\\HackathonWindsurf\\sample_images\\Chest\\preprocessed",
    "C:\\Users\\320210377\\Desktop\\TEMP\\HackathonWindsurf\\sample_images\\Brain\\glioma",
    "C:\\Users\\320210377\\Desktop\\TEMP\\HackathonWindsurf\\sample_images\\Brain\\meningioma",
    "C:\\Users\\320210377\\Desktop\\TEMP\\HackathonWindsurf\\sample_images\\Brain\\notumor",
    "C:\\Users\\320210377\\Desktop\\TEMP\\HackathonWindsurf\\sample_images\\Brain\\pituitary"
]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    # Reset caches when the page is loaded to ensure fresh data
    global image_cache, analysis_cache
    image_cache = None
    analysis_cache = None
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/images")
async def get_images():
    """Get list of images from the hardcoded paths"""
    global image_cache
    
    # Use cached images if available
    if image_cache is not None:
        return {"images": image_cache}
    
    # Get all JPG and PNG files from the hardcoded paths
    image_files = []
    
    # Check all directories
    for image_path in IMAGE_PATHS:
        if os.path.exists(image_path):
            # Get all jpg and png files
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(glob.glob(os.path.join(image_path, ext)))
    
    # Randomly select 10 images if we have more than 10
    if len(image_files) > 10:
        image_files = random.sample(image_files, 10)
    
    # Process images
    images = []
    for file_path in image_files:
        file_name = os.path.basename(file_path)
        
        # Get category from path
        if "Brain" in file_path:
            # Get brain subcategory
            if "glioma" in file_path.lower():
                category = "Brain - Glioma"
            elif "meningioma" in file_path.lower():
                category = "Brain - Meningioma"
            elif "notumor" in file_path.lower():
                category = "Brain - No Tumor"
            elif "pituitary" in file_path.lower():
                category = "Brain - Pituitary"
            else:
                category = "Brain - Other"
        elif "Chest" in file_path:
            category = "Chest"
        else:
            category = "Other"
            
        # Create relative URL for the image
        # Use a base64 encoded path as the ID to avoid path issues
        encoded_path = base64.urlsafe_b64encode(file_path.encode()).decode()
        image_url = f"/api/images/{encoded_path}"
        
        images.append({
            "name": file_name,
            "path": file_path,
            "url": image_url,
            "category": category
        })
    
    # Cache the results
    image_cache = images
    
    return {"images": images}


@app.get("/api/images/{encoded_path}")
async def get_image(encoded_path: str):
    """Serve an image using its encoded path"""
    try:
        # Decode the path
        import base64
        image_path = base64.urlsafe_b64decode(encoded_path.encode()).decode()
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {"error": "Image not found"}
        
        # Serve the file using FileResponse
        from fastapi.responses import FileResponse
        return FileResponse(image_path)
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}


@app.get("/api/analyze")
async def analyze_images():
    """Get analysis results for all images by calling the external API"""
    global analysis_cache
    
    # Return cached analysis if available
    if analysis_cache is not None:
        return {"results": analysis_cache, "api_status": {"online": True}}
    
    # Get all images
    images_response = await get_images()
    images = images_response.get("images", [])
    
    results = []
    
    # Mock API calls for now
    # In production, you would call the actual external API
    for image in images:
        # Call the appropriate API based on the image category
        try:
            # Determine which API endpoint to use based on image category
            if "Brain" in image.get("category", ""):
                api_url = BRAIN_API_URL
            elif "Chest" in image.get("category", ""):
                api_url = CHEST_API_URL
            else:
                # Default to brain API if category is unknown
                api_url = BRAIN_API_URL
            
            # Make the actual API call to the external service
            async with httpx.AsyncClient(timeout=10.0) as client:  # Increased timeout
                try:
                    # Open the image file from its path and send it to the API
                    with open(image["path"], "rb") as img_file:
                        files = {"file": img_file}
                        print(f"Sending request to {api_url} for image {image['name']}")
                        response = await client.post(api_url, files=files)
                    
                    if response.status_code != 200:
                        print(f"API error: {response.status_code} for image {image['name']}")
                        raise Exception(f"API returned status code {response.status_code}")
                        
                    # Parse the JSON response
                    api_response = response.json()
                    print(f"Success for {image['name']}: {api_response}")
                    
                    # Debug API response structure
                    print(f"API response type: {type(api_response)}")
                    if isinstance(api_response, dict):
                        print(f"API response keys: {list(api_response.keys())}")
                    elif isinstance(api_response, list) and len(api_response) > 0:
                        print(f"API response is a list with {len(api_response)} items")
                        if isinstance(api_response[0], dict):
                            print(f"First item keys: {list(api_response[0].keys())}")
                    else:
                        print(f"API response: {api_response}")
                
                except Exception as e:
                    print(f"Exception for {image['name']}: {str(e)}")
                    raise e  # Re-raise to be caught by outer try/except
                
                # Extract prediction, confidence, and priority directly from the API response
                prediction = api_response.get("prediction", "Unknown")
                
                # Handle confidence which might be a string percentage or a number
                confidence_value = api_response.get("confidence", 80.0)
                if isinstance(confidence_value, str) and "%" in confidence_value:
                    # Convert string percentage "96.15%" to float 96.15
                    confidence = float(confidence_value.strip("%"))
                else:
                    confidence = float(confidence_value)
                
                # Use priority directly from the API response
                priority = api_response.get("priority", "Normal")
                
                api_result = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "priority": priority
                }
            
            # Combine image info with API result
            result = {**image, **api_result}
            results.append(result)
            
        except Exception as e:
            # Handle errors without fallbacks
            print(f"Error processing image {image['name']}: {str(e)}")
            
            # Add to results with error status
            results.append({
                **image,
                "error": str(e),
                "prediction": "Error",
                "confidence": 0,
                "priority": "Unknown"
            })
    
    # Enhanced sorting:
    # 1. Sort by priority (Urgent > High > Normal)
    # 2. Push down images with "Normal" prediction and N/A priorities
    priority_order = {"Urgent": 1, "High": 2, "Medium": 3, "Low": 4, "N/A": 5}
    
    def sort_key(image):
        # First level: priority sorting
        priority_score = priority_order.get(image.get("priority"), 999)
        
        # Second level: push down "Normal" predictions and unknown priorities
        prediction_modifier = 0
        prediction = image.get("prediction", "").lower()
        if prediction == "normal":
            prediction_modifier = 100  # Add 100 to push normal predictions down
        
        # If priority is N/A or Unknown, push it further down
        if image.get("priority") in [None, "Unknown", "N/A"]:
            priority_score += 500
        
        # Combine scores (priority is primary, prediction is secondary)
        return priority_score + prediction_modifier
    
    # Apply the multi-criteria sorting
    results.sort(key=sort_key)
    
    # Check if any real API calls succeeded or all were fallbacks
    all_fallbacks = all(result.get("is_fallback", False) for result in results)
    api_status = {
        "online": not all_fallbacks,
        "message": "API endpoints are responding normally" if not all_fallbacks else "Warning: API endpoints are not responding. Using fallback predictions."
    }
    
    # Cache the results
    analysis_cache = results
    
    return {"results": results, "api_status": api_status}


@app.get("/api/health-check")
async def check_api_health():
    """Check if the external APIs are healthy and responding"""
    results = {
        "brain_api": {"online": False, "message": "Brain API is not responding"},
        "chest_api": {"online": False, "message": "Chest API is not responding"},
        "all_healthy": False
    }
    
    # Check Brain API
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            print(f"Checking Brain API health at {BRAIN_API_HEALTH_URL}")
            response = await client.get(BRAIN_API_HEALTH_URL)
            if response.status_code < 500:  # Any non-server error is considered "online"
                results["brain_api"] = {"online": True, "message": "Brain API is online"}
                print("Brain API is online")
    except Exception as e:
        print(f"Brain API health check failed: {str(e)}")
    
    # Check Chest API
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            print(f"Checking Chest API health at {CHEST_API_HEALTH_URL}")
            response = await client.get(CHEST_API_HEALTH_URL)
            if response.status_code < 500:  # Any non-server error is considered "online"
                results["chest_api"] = {"online": True, "message": "Chest API is online"}
                print("Chest API is online")
    except Exception as e:
        print(f"Chest API health check failed: {str(e)}")
    
    # Set overall health status
    results["all_healthy"] = results["brain_api"]["online"] and results["chest_api"]["online"]
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
