import cv2
import numpy as np
import time
import easyocr
from ultralytics import YOLO

print("Starting YOLO-based document capture...")

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")  # lightweight, CPU-friendly

# -----------------------------
# OCR Reader
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

# -----------------------------
# Camera setup
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

cv2.namedWindow("Document Verification", cv2.WINDOW_NORMAL)

captured = False
ocr_text = []

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    h, w, _ = frame.shape

    if not captured:
        # Run YOLO inference
        results = model(frame, conf=0.4, verbose=False)

        document_found = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 🚫 BLOCK PERSON COMPLETELY
                if cls_id == 0:  # person
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                bw = x2 - x1
                bh = y2 - y1
                aspect_ratio = bw / float(bh)

                # ✅ DOCUMENT-LIKE HEURISTICS
                if (
                    conf > 0.5 and
                    bw > 200 and
                    bh > 120 and
                    1.2 < aspect_ratio < 2.2
                ):
                    document_found = True

                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(
                        display,
                        "DOCUMENT DETECTED - PRESS ENTER",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    if cv2.waitKey(1) & 0xFF == 13:
                        doc_crop = frame[y1:y2, x1:x2]
                        cv2.imwrite("document_detected.jpg", doc_crop)
                        captured = True

                    break


        if not document_found:
            cv2.putText(
                display,
                "Show your document to the camera",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    else:
        cv2.putText(
            display,
            "Document captured successfully",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        y = 100
        for text in ocr_text:
            cv2.putText(
                display,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            y += 30

    cv2.imshow("Document Verification", display)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("Finished. Saved as document_detected.jpg")
