import cv2
import easyocr
import pytesseract
import re
import numpy as np
from PIL import Image
from scripts.cleanText import clean_text
from scripts.correctSkew import correct_skew
from concurrent.futures import ThreadPoolExecutor
import asyncio

easyocr_reader = easyocr.Reader(['en'], gpu=False)
executor = ThreadPoolExecutor(max_workers=4)

# Function to process OCR with EasyOCR
def process_easyocr(img_roi, min_confidence=0.5):
    """Process OCR using EasyOCR."""
    results = easyocr_reader.readtext(img_roi, detail=1)
    return [
        (clean_text(result[1]), result[2])
        for result in results if result[2] >= min_confidence
    ]

# Function to process OCR with Tesseract
def process_tesseract(img_roi):
    """Process OCR using Tesseract."""
    text = pytesseract.image_to_string(img_roi, config='--psm 7')
    return [(clean_text(text.strip()), 1.0)]  # Confidence set to 1.0 for simplicity

def preprocess_plate(cropped_plate):
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

    # Resize to ideal dimensions
    resized_image = cv2.resize(grayscale_image, (400, 100), interpolation=cv2.INTER_CUBIC)

    return resized_image


async def process_plate(img, idx, x1, y1, x2, y2):
        
        img_roi = correct_skew(img)

        #
        img_roi = img_roi[y1:y2, x1:x2]
        
        
        # img_roi= preprocess_plate(img_roi)
        cv2.imwrite(f"cv/cropped_plate_{idx}.jpg", img_roi)

        # Run EasyOCR and Tesseract in parallel
        easyocr_future = executor.submit(process_easyocr, img_roi)
        tesseract_future = executor.submit(process_tesseract, img_roi)
        
        easyocr_results = await asyncio.wrap_future(easyocr_future)
        tesseract_results = await asyncio.wrap_future(tesseract_future)
        
        return easyocr_results + tesseract_results
