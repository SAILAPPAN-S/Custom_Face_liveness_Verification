import cv2
import mediapipe as mp
import numpy as np
import time
import sys

print("Starting FULL liveness demo...")
print("Python:", sys.executable)

# -----------------------------
# Camera setup
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)

if not cap.isOpened():
    print("Camera not accessible")
    input("Press Enter to exit")
    sys.exit(1)

cv2.namedWindow("Liveness Demo", cv2.WINDOW_NORMAL)

# -----------------------------
# MediaPipe FaceMesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Eye + Face landmarks
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
eye_closed = False

# -----------------------------
# Liveness steps (ORDERED)
# -----------------------------
steps = [
    "LOOK_CENTER",
    "BLINK",
    "TURN_RIGHT",
    "TURN_LEFT",
    "TURN_UP",
    "TURN_DOWN",
    "DONE"
]

current_step = 0

# -----------------------------
# Helper functions
# -----------------------------
def eye_aspect_ratio(landmarks, eye, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def head_direction(landmarks, w, h):
    nose = landmarks[1]
    left = landmarks[234]
    right = landmarks[454]
    top = landmarks[10]
    bottom = landmarks[152]

    nose_x = nose.x * w
    nose_y = nose.y * h

    center_x = (left.x * w + right.x * w) / 2
    center_y = (top.y * h + bottom.y * h) / 2

    dx = nose_x - center_x
    dy = nose_y - center_y

    if dx > 15:
        return "RIGHT"
    if dx < -15:
        return "LEFT"
    if dy < -15:
        return "UP"
    if dy > 15:
        return "DOWN"

    return "CENTER"

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    cv2.imshow("Liveness Demo", frame)
    cv2.waitKey(1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    instruction = "No face detected"

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        step = steps[current_step]
        direction = head_direction(lm, w, h)

        # -------- STEP LOGIC --------
        if step == "LOOK_CENTER":
            instruction = "Look straight at the camera"
            if direction == "CENTER":
                current_step += 1

        elif step == "BLINK":
            instruction = "Please blink"
            ear = (
                eye_aspect_ratio(lm, LEFT_EYE, w, h) +
                eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ) / 2

            if ear < EAR_THRESHOLD and not eye_closed:
                eye_closed = True
            elif ear >= EAR_THRESHOLD and eye_closed:
                eye_closed = False
                current_step += 1

        elif step == "TURN_RIGHT":
            instruction = "Turn your head RIGHT"
            if direction == "RIGHT":
                current_step += 1

        elif step == "TURN_LEFT":
            instruction = "Turn your head LEFT"
            if direction == "LEFT":
                current_step += 1

        elif step == "TURN_UP":
            instruction = "Turn your head UP"
            if direction == "UP":
                current_step += 1

        elif step == "TURN_DOWN":
            instruction = "Turn your head DOWN"
            if direction == "DOWN":
                current_step += 1

        elif step == "DONE":
            instruction = "VERIFICATION SUCCESSFUL"

    # -----------------------------
    # UI
    # -----------------------------
    cv2.putText(
        frame,
        instruction,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 255),
        3
    )

    if steps[current_step] == "DONE":
        cv2.putText(
            frame,
            "LIVE USER CONFIRMED",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Liveness Demo", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
input("Liveness demo finished. Press Enter to exit.")
