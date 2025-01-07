# import easyocr
# import re
# import cv2
# from scripts.removeSymbols import removeSymbol

# # Mapping for number to alphabet replacements
# number_to_alphabet = {
#     '0': 'O',
#     '1': 'I',
#     '2': 'Z',
#     '3': 'E',
#     '4': 'A',
#     '5': 'S',
#     '6': 'G',
#     '7': 'T',
#     '8': 'B',
#     '9': 'P'
# }

# def textExtract(img, x1, y1, x2, y2):
#     img = img[y1:y2, x1:x2]
#     cv2.imwrite(f"cv/cropped_plate_0.jpg", img)

#     # Remove extra symbols
#     # img = removeSymbol(img)  # Optional: Remove unwanted symbols if required

#     cv2.imwrite(f"cv/cropped_plate_2.jpg", img)

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 0, 70, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     img = gray
#     cv2.imwrite(f"cv/cropped_plate_1.jpg", img)

#     cleaned_text = ""
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(img)
#     print(result)

#     for detection in result:
#         text = detection[1]  # Extract the detected text
#         print(text)
#         if not re.match(r'^[A-Z][a-z]+$', text) or len(re.findall(r'[A-Z]', text)) > 1:
#             cleaned_text += text + " "  # Keep valid lines and add a space for separation
#             print("Second read", cleaned_text)
#             cleaned_text_no_uppercase_before_lowercase = re.sub(r'[A-Z](?=[a-z])', '', cleaned_text)
#             cleaned_text_no_lowercase = re.sub(r'[a-z]', '', cleaned_text_no_uppercase_before_lowercase)
#             print(f"Cleaned Text: {cleaned_text_no_lowercase.strip()}")
#             final_text = re.sub(r'[^A-Za-z0-9]', '', cleaned_text_no_lowercase)  # Remove non-alphanumeric characters
#             final = final_text.replace(' ', '')  # Remove spaces
            
#             # Apply max length check
#             final = final[:10] if len(final) > 10 else final

#     # Check if the first character is a number and replace it
#     if final and final[0] in number_to_alphabet:
#         final = number_to_alphabet[final[0]] + final[1:]

#     return final


import re
import cv2
from paddleocr import PaddleOCR

# Mapping for number to alphabet replacements
number_to_alphabet = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '7': 'T',
    '8': 'B',
    '9': 'P'
}

def textExtract(img, x1, y1, x2, y2):
    # Crop the region of interest (ROI)
    img = img[y1:y2, x1:x2]
    cv2.imwrite(f"cv/cropped_plate_0.jpg", img)

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 70, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = binary
    cv2.imwrite(f"cv/cropped_plate_1.jpg", img)

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Enable angle classification if needed
    results = ocr.ocr(img, cls=True)  # Perform OCR

    cleaned_text = ""
    print("OCR Results:", results)

    for detection in results[0]:
        text = detection[1][0]  # Extract the detected text
        print("Detected Text:", text)
        if not re.match(r'^[A-Z][a-z]+$', text) or len(re.findall(r'[A-Z]', text)) > 1:
            cleaned_text += text + " "  # Keep valid lines and add a space for separation
            print("Second read", cleaned_text)
            cleaned_text_no_uppercase_before_lowercase = re.sub(r'[A-Z](?=[a-z])', '', cleaned_text)
            cleaned_text_no_lowercase = re.sub(r'[a-z]', '', cleaned_text_no_uppercase_before_lowercase)
            print(f"Cleaned Text: {cleaned_text_no_lowercase.strip()}")
            final_text = re.sub(r'[^A-Za-z0-9]', '', cleaned_text_no_lowercase)  # Remove non-alphanumeric characters
            final = final_text.replace(' ', '')  # Remove spaces
            
            # Apply max length check
            final = final[:10] if len(final) > 10 else final

    # Check if the first character is a number and replace it
    if final and final[0] in number_to_alphabet:
        final = number_to_alphabet[final[0]] + final[1:]

    # Remove the last character if it is an alphabet
    if final and final[-1].isalpha():
        final = final[:-1]

    return final
