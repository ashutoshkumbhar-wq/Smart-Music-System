# collect_gestures.py
# Same as before, but captures samples slower (cooldown) so you can vary distance.

import os, sys, json, time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import mediapipe as mp

# ================= CONFIG =================
OUTPUT_JSON = "testing1.json"

SAMPLES_PER_LABEL = int(os.getenv("SAMPLES_PER_LABEL", "500"))
SAMPLES_NONE      = int(os.getenv("SAMPLES_NONE", "2000"))

# Slow down capture
SAMPLE_COOLDOWN_MS = int(os.getenv("SAMPLE_COOLDOWN_MS", "350"))  # default 350ms
REQUIRED_STABLE_FRAMES = int(os.getenv("REQUIRED_STABLE_FRAMES", "3"))

IS_MAC = (sys.platform == "darwin")
CAM_INDEX = int(os.getenv("GESTURE_CAM_INDEX", "0"))
USE_AVFOUNDATION = True if IS_MAC else False

_ENV_MIRROR = os.getenv("GESTURE_MIRROR")
if _ENV_MIRROR is not None:
    MIRROR_INPUT = _ENV_MIRROR in ("1", "true", "True")
else:
    MIRROR_INPUT = False

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

RIGHT_LABELS = ["play", "pause", "next", "previous"]
LEFT_LABELS  = ["volume_up", "volume_down", "like", "skip30"]
NOISE_LABELS = ["none"]
LABELS = RIGHT_LABELS + LEFT_LABELS + NOISE_LABELS

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
draw = mp.solutions.drawing_utils

def expected_side_for(label: str) -> str:
    if label in RIGHT_LABELS:
        return "right"
    if label in LEFT_LABELS:
        return "left"
    return "either"

def open_camera(index: int, use_avfoundation: bool):
    if use_avfoundation:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}. Try 0/1/2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

def draw_label_bar(img, label, count, target, side_hint=None, color=(0,255,255)):
    t1 = f"Label: {label}  {count}/{target}"
    cv2.putText(img, t1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    if side_hint:
        cv2.putText(img, f"Use {side_hint.upper()} HAND", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,180,255), 2)
    if label == "none":
        cv2.putText(img, "Noise: idle / random poses / no hand",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,255), 2)
    cv2.putText(img, "Keys: n=next  r=redo  q=quit",
                (10, FRAME_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

def to_feature_vec(hand_landmarks):
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    vec = []
    for lm in hand_landmarks.landmark:
        vec.append(round(lm.x - base_x, 4))
        vec.append(round(lm.y - base_y, 4))
    return vec

def zero_vec():
    return [0.0] * 42

def main():
    print("=== Gesture Collector (slower capture) ===")
    print(f"Saving to: {OUTPUT_JSON}")
    print(f"Cooldown: {SAMPLE_COOLDOWN_MS}ms | Stable frames: {REQUIRED_STABLE_FRAMES}")
    print(f"Targets: per-gesture={SAMPLES_PER_LABEL} | none={SAMPLES_NONE}")

    out_path = Path(OUTPUT_JSON)
    if out_path.exists():
        out_path.unlink()

    cap = open_camera(CAM_INDEX, USE_AVFOUNDATION)

    data = []
    counts = defaultdict(int)
    last_time = 0
    stable_q = deque(maxlen=REQUIRED_STABLE_FRAMES)

    for label in LABELS:
        need_side = expected_side_for(label)
        target = SAMPLES_NONE if label == "none" else SAMPLES_PER_LABEL
        stable_q.clear()
        print(f"\n=== Recording: {label} ({target}) [expect {need_side.upper()}] ===")

        while counts[label] < target:
            ok, img = cap.read()
            if not ok: continue
            if MIRROR_INPUT: img = cv2.flip(img, 1)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            overlay = img.copy()
            side_hint = None if label == "none" else need_side
            draw_label_bar(overlay, label, counts[label], target, side_hint)

            counted = False
            now = time.time() * 1000.0

            if label == "none":
                if result.multi_hand_landmarks:
                    hand_lms = result.multi_hand_landmarks[0]
                    draw.draw_landmarks(overlay, hand_lms, mp_hands.HAND_CONNECTIONS)
                    vec = to_feature_vec(hand_lms)
                else:
                    vec = zero_vec()
                stable_q.append("none")
                if len(stable_q) == REQUIRED_STABLE_FRAMES and (now - last_time) >= SAMPLE_COOLDOWN_MS:
                    data.append({"X": vec, "y": "none"})
                    counts[label] += 1
                    counted = True
                    last_time = now
            else:
                if result.multi_hand_landmarks and result.multi_handedness:
                    hand_lms = result.multi_hand_landmarks[0]
                    handed = result.multi_handedness[0].classification[0].label.lower()
                    if handed == need_side:
                        draw.draw_landmarks(overlay, hand_lms, mp_hands.HAND_CONNECTIONS)
                        vec = to_feature_vec(hand_lms)
                        stable_q.append(f"{label}_{handed}")
                        if len(stable_q) == REQUIRED_STABLE_FRAMES and (now - last_time) >= SAMPLE_COOLDOWN_MS:
                            data.append({"X": vec, "y": f"{label}_{handed}"})
                            counts[label] += 1
                            counted = True
                            last_time = now

            cv2.circle(overlay, (30, FRAME_HEIGHT - 30), 10,
                       (0,255,0) if counted else (0,0,255), -1)
            cv2.imshow("Collect Gestures (slower)", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): sys.exit(0)
            elif key == ord('r'): counts[label] = 0; stable_q.clear()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done. Saved {sum(counts.values())} samples to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

# SAMPLES_PER_LABEL=500 SAMPLES_NONE=2000 GESTURE_MIRROR=1 python3 collect_gestures.py
