import io
import os
import cv2
import asyncio
import numpy as np
import pytesseract
from io import BytesIO
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from scripts.classifyTextStyle import classify_text_style
from scripts.cleanText import clean_text
from scripts.processing import process_plate
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile



os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

os.makedirs("cv", exist_ok=True)
app = FastAPI()
model = YOLO('gpuversion.pt')
executor = ThreadPoolExecutor(max_workers=4)

MIN_CONFIDENCE = 0.2
# MIN_AREA = 600




def resize_image(img_array, max_size_kb= 800):
    try:
        img = Image.fromarray(img_array)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
            img.save(temp_file.name, format="JPEG")
            original_size = os.path.getsize(temp_file.name)
            if original_size <= (max_size_kb * 1024):
                print("Image size is within the limit. No resizing needed.")
                return img
            
            print(f"Original size: {original_size / 1024:.2f} KB. Resizing...")
            quality = 90 
            
            while quality >= 10:
                temp_file.seek(0) 
                img.save(temp_file.name, format="JPEG", quality=quality)
                resized_size = os.path.getsize(temp_file.name)

                if resized_size <= (max_size_kb * 1024):
                    print(f"Resized to {resized_size / 1024:.2f} KB.")
                    return Image.open(temp_file.name)  

                quality -= 10

            print("Warning: Image could not be resized below the desired file size limit.")
            return img
    except Exception as e:
        print(f"Error resizing image: {e}")
        raise

# Endpoint
@app.post("/extract-number-plate/")
async def extract_number_plate(file: UploadFile = File(...)):
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_data = await file.read()
    # image_data = np.array(image_data)
 
    if len(image_data) > 700 * 1024:
        img = np.array(Image.open(io.BytesIO(image_data)))
        img = resize_image(img) 
    else:
        img = np.array(Image.open(io.BytesIO(image_data)))  
    

    try:
        img = Image.open(io.BytesIO(image_data))
        img = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    if img is None or not isinstance(img, np.ndarray):
        raise HTTPException(status_code=500, detail="Failed to convert image to a NumPy array")
    
    # Resize the image
    # try:
    #     img = cv2.resize(img, (1000, 900))
    #     print("Resized to 800x800")
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail="Error resizing the image")

    
    try:
        results = model.predict(source=img, save=False)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during YOLO inference")
    
    
    detected_plates = []
    for r in results:
        for box in r.boxes:  # r.boxes contains the detected boxes in YOLOv8 format
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            if cls == 0 and conf > MIN_CONFIDENCE:  # Assuming class ID 0 is for license plates
                detected_plates.append((x1+10, y1, x2, y2, conf, cls))

    if not detected_plates:
        raise HTTPException(status_code=404, detail="No number plate found in the image.")
    
    tasks = [
        process_plate(img, idx, x1, y1, x2, y2)
        for idx, (x1, y1, x2, y2, _, _) in enumerate(detected_plates)
    ]

    all_results = await asyncio.gather(*tasks)

    # Flatten results
    detected_texts = [item for sublist in all_results for item in sublist]

    if not detected_texts:
        return JSONResponse(content={"error": "Failed to extract text from the image"}, status_code=400)

    # Classify and filter results by majority type
    text_type_counts = Counter(classify_text_style(text) for text, _ in detected_texts)
    majority_type = text_type_counts.most_common(1)[0][0]
    majority_texts = [text for text, text_type in detected_texts if classify_text_style(text) == majority_type]

    largest_majority_text = max(majority_texts, key=len) if majority_texts else None
    if not largest_majority_text:
        return JSONResponse(content={"error": "No valid text found for the majority type"}, status_code=400)

    return JSONResponse(content={"extracted_number": clean_text(largest_majority_text)})