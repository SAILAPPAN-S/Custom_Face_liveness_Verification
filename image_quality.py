import cv2
import numpy as np
from PIL import Image

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)

def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score

def check_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    return brightness

def validate_image_quality(image_file):
    image = load_image(image_file)

    blur_score = check_blur(image)
    brightness = check_brightness(image)

    # Thresholds (tuned for ID cards)
    if blur_score < 100:
        return {
            "status": "REJECTED",
            "reason": "Image is too blurry",
            "blur_score": round(blur_score, 2)
        }

    if brightness < 60:
        return {
            "status": "REJECTED",
            "reason": "Image is too dark",
            "brightness": round(brightness, 2)
        }

    if brightness > 200:
        return {
            "status": "REJECTED",
            "reason": "Image is too bright",
            "brightness": round(brightness, 2)
        }

    return {
        "status": "OK",
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2)
    }
