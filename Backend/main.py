import io
import os
import cv2
import asyncio
import numpy as np
from io import BytesIO
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from scripts.textExtract import textExtract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import requests
from fastapi.middleware.cors import CORSMiddleware



os.makedirs("cv", exist_ok=True)
app = FastAPI()


origins = [
    "http://localhost:8006",  # Adjust as needed
     "http://18.188.184.213:8006",
      # Allow all origins for development (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

model = YOLO('models/gpuVersion.pt')
executor = ThreadPoolExecutor(max_workers=4)

MIN_CONFIDENCE = 0.7
# MIN_AREA = 600


@app.get("/")
async def root():
    return {"message": "Welcome to the License Plate Extraction API"}


@app.post("/extract-number-plate/")
async def extract_number_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    img = await file.read()
    img = Image.open(BytesIO(img))
    img = np.array(img)
    
    # try:
    #     results = model.predict(source=img, save=False)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error during YOLO inference: {str(e)}")
    
    # detected_plates = []
    # for r in results:
    #     for box in r.boxes:
    #         # Extract box coordinates, confidence, and class
    #         x1, y1, x2, y2 = map(int, box.xyxy[0])
    #         conf = float(box.conf[0])
    #         cls = int(box.cls[0])
            
    #         # Debugging logs
    #         print(f"Box coordinates: {x1, y1, x2, y2}, Confidence: {conf}, Class: {cls}")
            
    #         # Ensure confidence and class are valid
    #         if cls == (0 or 11) and conf > MIN_CONFIDENCE:
    #             detected_plates.append((x1, y1, x2, y2))

    # if not detected_plates:
    #     raise HTTPException(status_code=404, detail="No number plate found in the image.")
    
    # # Process the first detected plate (or modify to process all plates if needed)
    # x1, y1, x2, y2 = detected_plates[0]
    extracted_text = textExtract(img)

    if not extracted_text:
        return JSONResponse(content={"error": "Failed to extract text from the image"}, status_code=400)

    return JSONResponse(content={"extracted_number": extracted_text})
