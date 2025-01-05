# import numpy as np
# import cv2

# Function to correct skew in detected text areas
# def correct_skew(img_roi):

#     gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if contours:
#         cnt = max(contours, key=cv2.contourArea)
#         epsilon = 0.02 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
        
#         if len(approx) == 4:
#             pts = approx.reshape(4, 2).astype(np.float32)
#             sum_pts = pts.sum(axis=1)
#             diff_pts = np.diff(pts, axis=1)
#             top_left = pts[np.argmin(sum_pts)]
#             bottom_right = pts[np.argmax(sum_pts)]
#             top_right = pts[np.argmin(diff_pts)]
#             bottom_left = pts[np.argmax(diff_pts)]
            
#             width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
#             height = int(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
#             dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            
#             M = cv2.getPerspectiveTransform(np.array([top_left, top_right, bottom_right, bottom_left]), dst_pts)
#             return cv2.warpPerspective(img_roi, M, (width, height))
    
#     return img_roi

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(img_roi, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    
    if not isinstance(img_roi, np.ndarray):
            img_roi = np.array(img_roi)

    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = img_roi.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(img_roi, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return corrected