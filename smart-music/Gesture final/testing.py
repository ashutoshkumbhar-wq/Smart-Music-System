# maintesting1.py
import os, sys
import cv2
import mediapipe as mp
import joblib
import numpy as np

# ==== Model ====
model  = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")
CLASSES = list(model.classes_)

# ==== Camera settings ====
IS_MAC = (sys.platform == "darwin")
CAM_INDEX = int(os.getenv("GESTURE_CAM_INDEX", "0"))

# Mirror ON by default only on macOS; override with GESTURE_MIRROR=0/1
env_mirror = os.getenv("GESTURE_MIRROR")
if env_mirror is not None:
    MIRROR_FEED = env_mirror in ("1", "true", "True")
else:
    MIRROR_FEED = IS_MAC

# Open camera (AVFoundation on Mac)
if IS_MAC:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAM_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

# ==== MediaPipe ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
draw = mp.solutions.drawing_utils

CONF_THRESHOLD = 0.80  # show only if p >= 0.80 and not "none"

print("🎥 Real-time gesture recognition started! Press 'q' to quit...")
try:
    while True:
        ok, img = cap.read()
        if not ok:
            continue

        if MIRROR_FEED:
            img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        gesture_text = "No hand detected"
        prob_text = ""

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Normalize using wrist (landmark 0) as base
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            feat = []
            for lm in hand_landmarks.landmark:
                feat.append(round(lm.x - base_x, 4))
                feat.append(round(lm.y - base_y, 4))

            # Scale & predict
            Xs = scaler.transform([feat])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xs)[0]
                max_prob = float(probs.max())
                predicted_label = CLASSES[int(probs.argmax())]
            else:
                predicted_label = model.predict(Xs)[0]
                max_prob = 1.0

            # Only show confident, non-"none"
            if predicted_label == "none" or max_prob < CONF_THRESHOLD:
                gesture_text = "❌ Not a gesture"
                prob_text = f"p={max_prob:.2f}"
            else:
                gesture_text = f"✅ {predicted_label}"
                prob_text = f"p={max_prob:.2f}"

        # HUD
        cv2.putText(img, gesture_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        if prob_text:
            cv2.putText(img, prob_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
        cv2.putText(img, f"Mirror:{MIRROR_FEED}  Thr:{CONF_THRESHOLD}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

        cv2.imshow("🤖 Real-Time Gesture Prediction", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Quit requested. Exiting...")
            break

except KeyboardInterrupt:
    print("💣 Interrupted by user. Exiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()

# GESTURE_MIRROR=1 GESTURE_CAM_INDEX=0 python3 testing.py
