#!/usr/bin/env python3
# one_file_integrated.py
# Always prompts:
#   1) Feature: Face / Gesture / Both
#   2) If Face included -> Quality (High/Mid) then Sensitivity (High/Mid/Low)
# Overlays emotion bars + primary label and gesture class + prob.

import os, sys, argparse, warnings
from pathlib import Path
import numpy as np
import cv2
from fer import FER
import mediapipe as mp
import joblib

# ===============================
# ---- Sensitivity profiles -----
# ===============================
SENS_HIGH_GAINS = {"neutral": 0.85, "happy": 1.25, "sad": 1.25}
SENS_MID_GAINS  = {"neutral": 0.95, "happy": 1.10, "sad": 1.10}
SENS_LOW_GAINS  = {"neutral": 1.00, "happy": 1.00, "sad": 1.00}  # Original FER
HIGH_EMA, HIGH_MINCONF = 0.65, 0.15
MID_EMA,  MID_MINCONF  = 0.55, 0.20
LOW_EMA,  LOW_MINCONF  = 0.55, 0.20

# ---------- Utils ----------
def ema_update(prev, new, alpha):
    if prev is None:
        return new.copy()
    return {k: (alpha*new[k] + (1-alpha)*prev[k]) for k in new}

def softmax_from_vals(vals_dict):
    keys = list(vals_dict.keys())
    vals = np.array([vals_dict[k] for k in keys], dtype=np.float32)
    ex = np.exp(vals - vals.max())
    probs = ex / (ex.sum() + 1e-9)
    return {k: float(p) for k, p in zip(keys, probs)}

def map_to_trio(emodict):
    sad_sum = float(emodict.get("sad", 0.0)) \
            + float(emodict.get("angry", 0.0)) \
            + float(emodict.get("disgust", 0.0)) \
            + float(emodict.get("fear", 0.0))
    trio = {"neutral": float(emodict.get("neutral", 0.0)),
            "happy":   float(emodict.get("happy",   0.0)),
            "sad":     sad_sum}
    return softmax_from_vals(trio)

def apply_clahe_bgr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def renorm_sum1(d):
    s = sum(d.values()) + 1e-9
    return {k: float(max(0.0, v) / s) for k, v in d.items()}

def prompt_feature():
    while True:
        print("\nSelect feature:")
        print("  1) Face only")
        print("  2) Gesture only")
        print("  3) Both face + gesture")
        choice = input("Enter 1, 2, or 3 [default 3]: ").strip()
        if choice in ("1","2","3",""):
            if choice == "1": return "face"
            if choice == "2": return "gesture"
            return "both"
        print("Invalid. Type 1, 2, or 3.")

def prompt_quality():
    while True:
        print("\nSelect quality:")
        print("  1) High quality (slower, better small/angled faces) → MTCNN=True")
        print("  2) Mid quality (faster) → MTCNN=False")
        choice = input("Enter 1 or 2 [default 2]: ").strip()
        if choice in ("1","2",""):
            return True if choice == "1" else False
        print("Invalid. Type 1 or 2.")

def prompt_sensitivity():
    while True:
        print("\nSelect sensitivity:")
        print("  1) High Sensitive  (bias emotions more)")
        print("  2) Mid Sensitive   (mild bias)")
        print("  3) Low Sensitive   (Original FER, no bias)")
        choice = input("Enter 1, 2, or 3 [default 3]: ").strip()
        if choice in ("1","2","3",""):
            if choice == "1": return "high"
            if choice == "2": return "mid"
            return "low"
        print("Invalid. Type 1, 2, or 3.")

# ------------------------------
# ---- Gesture helper utils ----
# ------------------------------
mp_hands = mp.solutions.hands
_draw    = mp.solutions.drawing_utils
_styles  = mp.solutions.drawing_styles

def make_gesture_extractor(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
    return hands

def extract_gesture_features(hand_landmarks):
    # 42-D: (x - wrist_x, y - wrist_y) for 21 landmarks
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    feat = []
    for lm in hand_landmarks.landmark:
        feat.append(round(lm.x - base_x, 4))
        feat.append(round(lm.y - base_y, 4))
    return np.array(feat, dtype=np.float32).reshape(1, -1)

def draw_hand_landmarks(img, hand_landmarks):
    _draw.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        _styles.get_default_hand_landmarks_style(),
        _styles.get_default_hand_connections_style(),
    )

def find_file(candidates):
    for p in candidates:
        if p and Path(p).exists():
            return str(p)
    return None

def main():
    ap = argparse.ArgumentParser(description="Face / Gesture / Both with prompts. Emotion bars + primary + Gesture class.")
    ap.add_argument("--camera", type=int, default=None, help="Webcam index (overrides env GESTURE_CAM_INDEX)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    # Allow skipping prompts if you want
    ap.add_argument("--mode", choices=["face","gesture","both"])
    ap.add_argument("--quality", choices=["high","mid"])
    ap.add_argument("--sensitivity", choices=["high","mid","low"])
    # Gesture artifacts (we'll also search common paths)
    ap.add_argument("--gesture_model", default=None)
    ap.add_argument("--gesture_scaler", default=None)
    args = ap.parse_args()

    # --------- Top-level feature selection (ALWAYS prompt unless --mode given) ---------
    feature_mode = args.mode if args.mode else prompt_feature()
    include_face = feature_mode in ("face","both")
    include_gesture = feature_mode in ("gesture","both")

    # --------- Face: quality + sensitivity (ALWAYS prompt unless flags provided) ---------
    mtcnn_on = False
    level = "low"
    if include_face:
        if args.quality:
            mtcnn_on = (args.quality == "high")
        else:
            mtcnn_on = prompt_quality()
        if args.sensitivity:
            level = args.sensitivity
        else:
            level = prompt_sensitivity()

        if level == "high":
            gains = SENS_HIGH_GAINS.copy()
            ema_alpha, min_conf = HIGH_EMA, HIGH_MINCONF
        elif level == "mid":
            gains = SENS_MID_GAINS.copy()
            ema_alpha, min_conf = MID_EMA, MID_MINCONF
        else:
            gains = SENS_LOW_GAINS.copy()
            ema_alpha, min_conf = LOW_EMA, LOW_MINCONF

        # FER detector (always loads)
        try:
            detector = FER(mtcnn=mtcnn_on)
        except Exception:
            detector = FER(mtcnn=False)

    # --------- Gesture: load artifacts robustly ---------
    gesture_model = None
    gesture_scaler = None
    gesture_classes = None
    hands = None
    if include_gesture:
        script_dir = Path(__file__).resolve().parent
        # Build candidate paths
        model_candidates = [
            args.gesture_model,
            "gesture_model.pkl",
            script_dir / "gesture_model.pkl",
            Path("/mnt/data/gesture_model.pkl"),
        ]
        scaler_candidates = [
            args.gesture_scaler,
            "scaler.pkl",
            script_dir / "scaler.pkl",
            Path("/mnt/data/scaler.pkl"),
        ]
        model_path  = find_file(model_candidates)
        scaler_path = find_file(scaler_candidates)

        if model_path is None or scaler_path is None:
            print(f"[Gesture] Could not find model/scaler.\n"
                  f"  Tried model: {model_candidates}\n"
                  f"  Tried scaler: {scaler_candidates}", file=sys.stderr)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gesture_model = joblib.load(model_path)
                    gesture_scaler = joblib.load(scaler_path)
                    gesture_classes = getattr(gesture_model, "classes_", None)
                hands = make_gesture_extractor()
            except Exception as e:
                print(f"[Gesture] Failed to load model/scaler: {e}", file=sys.stderr)

    # --------- Camera ---------
    cam_index_env = int(os.getenv("GESTURE_CAM_INDEX", "0"))
    cam_index = cam_index_env if args.camera is None else int(args.camera)
    mirror = True  # <<< hardcode mirroring ON

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("Cannot open webcam", file=sys.stderr)
        sys.exit(1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    emotions3 = ["neutral", "happy", "sad"]
    ema3 = None
    use_clahe = False

    print("Press 'q' to quit, 'c' to toggle CLAHE (face only).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        frame_out = frame.copy()

        # ----- Face pipeline -----
        if include_face:
            frame_in = apply_clahe_bgr(frame_out) if use_clahe else frame_out
            rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            results = detector.detect_emotions(rgb)

            if results:
                faces = sorted(results, key=lambda r: r["box"][2]*r["box"][3], reverse=True)
                best = faces[0]
                (x, y, w, h) = best["box"]
                trio = map_to_trio(best["emotions"])
                ema3 = ema_update(ema3, trio, ema_alpha)

                gained = {k: ema3[k] * gains[k] for k in emotions3}
                trio_adj = renorm_sum1(gained)

                cv2.rectangle(frame_in, (x, y), (x+w, y+h), (0,255,0), 2)
                primary = max(trio_adj.items(), key=lambda kv: kv[1])
                # "Primary" guess
                cv2.putText(frame_in, f"Primary: {primary[0]} ({primary[1]:.2f})",
                            (10, 65), font, 0.85, (255,255,255), 2, cv2.LINE_AA)

                # Bars + labels
                bar_x, bar_y = 10, int(args.height*0.65)
                bar_w, bar_h, gap = 260, 18, 8
                qual_label = "HQ" if mtcnn_on else "MidQ"
                lvl_label = {"high":"High", "mid":"Mid", "low":"Low (Original)"}[level]
                cv2.putText(frame_in, f"{qual_label} | {lvl_label} Sensitivity | Neutral / Happy / Sad",
                            (bar_x, bar_y-15), font, 0.6, (200,200,200), 1, cv2.LINE_AA)
                for i, e in enumerate(emotions3):
                    p = float(trio_adj[e])
                    length = int(p * bar_w)
                    yrow = bar_y + i * (bar_h + gap)
                    cv2.rectangle(frame_in, (bar_x, yrow), (bar_x + bar_w, yrow + bar_h), (70,70,70), 1)
                    cv2.rectangle(frame_in, (bar_x, yrow), (bar_x + length, yrow + bar_h), (255,255,255), -1)
                    tag = f"{e:>7}: {p:0.2f}  gain={ {'neutral':SENS_LOW_GAINS,'happy':SENS_LOW_GAINS,'sad':SENS_LOW_GAINS}[e][e] if level=='low' else {'neutral':SENS_MID_GAINS,'happy':SENS_MID_GAINS,'sad':SENS_MID_GAINS}[e][e] if level=='mid' else {'neutral':SENS_HIGH_GAINS,'happy':SENS_HIGH_GAINS,'sad':SENS_HIGH_GAINS}[e][e] :.2f}"
                    cv2.putText(frame_in, tag, (bar_x + bar_w + 12, yrow + bar_h - 4),
                                font, 0.55, (230,230,230), 1, cv2.LINE_AA)
                frame_out = frame_in
            else:
                cv2.putText(frame_out, "No face detected", (10, 30), font, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # ----- Gesture pipeline -----
        if include_gesture and (gesture_model is not None) and (gesture_scaler is not None) and (hands is not None):
            rgb_for_hands = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_for_hands.flags.writeable = False
            result = hands.process(rgb_for_hands)
            rgb_for_hands.flags.writeable = True

            gesture_text = "Gesture: No hand"
            prob_text = ""
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                draw_hand_landmarks(frame_out, hand_landmarks)

                try:
                    feat = extract_gesture_features(hand_landmarks)
                    feat_scaled = gesture_scaler.transform(feat)
                    if hasattr(gesture_model, "predict_proba"):
                        probs = gesture_model.predict_proba(feat_scaled)[0]
                        pred_idx = int(np.argmax(probs))
                        classes = getattr(gesture_model, "classes_", None)
                        pred_cls = classes[pred_idx] if classes is not None else str(pred_idx)
                        pred_prob = float(probs[pred_idx])
                        gesture_text = f"Gesture: {pred_cls}"
                        prob_text = f"P={pred_prob:.2f}"
                    else:
                        pred_cls = gesture_model.predict(feat_scaled)[0]
                        gesture_text = f"Gesture: {str(pred_cls)}"
                        prob_text = ""
                except Exception as e:
                    gesture_text = f"Gesture error: {e.__class__.__name__}"
                    prob_text = ""
            # Draw top-left (under face text)
            gy = int(args.height*0.10) + 25
            cv2.putText(frame_out, gesture_text, (10, gy), font, 0.8, (180,180,180), 2, cv2.LINE_AA)
            if prob_text:
                cv2.putText(frame_out, prob_text, (10, gy + 28), font, 0.7, (180,180,180), 2, cv2.LINE_AA)
        elif include_gesture and (gesture_model is None or gesture_scaler is None):
            cv2.putText(frame_out, "Gesture model/scaler not loaded", (10, 100), font, 0.7, (0,0,255), 2, cv2.LINE_AA)

        # Final show
        window_title = {"face":"Face Only","gesture":"Gesture Only","both":"Face + Gesture"}[feature_mode]
        cv2.imshow(window_title, frame_out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            if include_face: use_clahe = not use_clahe

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
