import pytesseract
import cv2
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\Sailappan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

def preprocess_image(image_file):
    image = np.array(Image.open(image_file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh

def extract_text_from_image(image_file):
    processed_image = preprocess_image(image_file)
    text = pytesseract.image_to_string(
        processed_image,
        config="--psm 6"
    )
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 2]
    return {
        "raw_text": "\n".join(lines),
        "lines": lines
    }

