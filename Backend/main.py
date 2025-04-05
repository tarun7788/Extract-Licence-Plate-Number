import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from scripts.textExtract import textExtract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import requests

# Create necessary directories
os.makedirs("cv", exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8006", "http://18.188.184.213:8006"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO('models/gpuVersion.pt')
executor = ThreadPoolExecutor(max_workers=4)

# Constants
MIN_CONFIDENCE = 0.7
IMAGE_PATH = "cv/cropped_plate_1.jpg"  
API_URL = "http://localhost:8000/extract-number-plate/"  

@app.get("/")
async def root():
    return {"message": "Welcome to the License Plate Extraction API"}

@app.post("/extract-number-plate/")
async def extract_number_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and convert image
        img_data = await file.read()
        image = Image.open(BytesIO(img_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model.predict(source=image, save=False)
        detected_plates = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if cls in [0, 11] and conf > MIN_CONFIDENCE:  # Check for license plate class
                    detected_plates.append((x1, y1, x2, y2))

        if not detected_plates:
            raise HTTPException(status_code=404, detail="No number plate found in the image.")

        # Process first detected plate
        x1, y1, x2, y2 = detected_plates[0]
        plate_crop = image[y1:y2, x1:x2]

        # Extract text from number plate
        extracted_text = textExtract(plate_crop)
        
        if not extracted_text:
            return JSONResponse(content={"error": "Failed to extract text from the image"}, status_code=400)

        return JSONResponse(content={"extracted_number": extracted_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# 🕒 Scheduler Function to Send Image Every 24 Hours
def call_api():
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            files = {"file": (IMAGE_PATH, image_file, "image/jpeg")}
            response = requests.post(API_URL, files=files)

        print("API Response:", response.json())  # Print the response from the server
    except Exception as e:
        print("Error calling API:", str(e))

# Start Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(call_api, "interval", hours=24)
scheduler.start()

# Stop scheduler gracefully when shutting down
@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()
