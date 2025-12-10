#!/usr/bin/env python3
# one_file_integrated_final.py
# Shows full UI + switch menus + sends only stable emotions (~3.5 s) + gestures to orchestrator
# EXACT REQUIREMENT: Two face modes only.
# - High  => must use FER(mtcnn=True). If unavailable or fails at any time → EXIT with error.
# - Low   => FER(mtcnn=False).
# UI and on-screen text unchanged from your version.

import os, sys, argparse, warnings, json, socket, time, subprocess, importlib, shlex
from pathlib import Path
import numpy as np
import cv2
from fer import FER
import mediapipe as mp
import joblib

# ---------------- Socket to orchestrator ----------------
ORCH_HOST = "127.0.0.1"
ORCH_PORT = 5050
_sock = None

def _connect_socket():
    global _sock
    if _sock is not None:
        return
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ORCH_HOST, ORCH_PORT))
        s.setblocking(True)
        _sock = s
        print(f"[detector] Connected to orchestrator at {ORCH_HOST}:{ORCH_PORT}")
    except Exception as e:
        _sock = None
        print(f"[detector] Orchestrator not available ({e}). Will retry…", file=sys.stderr)

def send_state(gesture, prob, emotion):
    """Send exactly what appears on the UI window."""
    global _sock
    if _sock is None:
        _connect_socket()
        if _sock is None:
            return
    try:
        msg = json.dumps({
            "gesture": gesture if gesture else "None",
            "prob": float(prob if prob else 0.0),
            "emotion": {
                "neutral": float(emotion.get("neutral", 0.0)) if emotion else 0.0,
                "happy":   float(emotion.get("happy",   0.0)) if emotion else 0.0,
                "sad":     float(emotion.get("sad",     0.0)) if emotion else 0.0
            }
        }) + "\n"
        _sock.sendall(msg.encode("utf-8"))
    except Exception:
        try: _sock.close()
        except Exception: pass
        _sock = None

# ---------------- Emotion stabilizer ----------------
SAD_SATELLITE_WEIGHT = 0.25   # was 0.35
EMO_CONF_MIN = 0.45
STABLE_SECONDS = 3.5
MIN_FACE_AREA = 70 * 70

class EmotionStabilizer:
    def __init__(self, stable_seconds=STABLE_SECONDS, conf_min=EMO_CONF_MIN):
        self.stable_seconds = stable_seconds
        self.conf_min = conf_min
        self.current_label = "neutral"
        self.current_probs = {"neutral":1.0,"happy":0.0,"sad":0.0}
        self.candidate_label = None
        self.candidate_start = None

    def update(self, trio_probs):
        label, p = max(trio_probs.items(), key=lambda kv: kv[1])
        if p < self.conf_min:
            label = "neutral"

        now = time.time()
        if label == self.current_label:
            self.current_probs = trio_probs
            self.candidate_label = None
            self.candidate_start = None
            return False

        if self.candidate_label != label:
            self.candidate_label = label
            self.candidate_start = now
            return False

        if now - self.candidate_start >= self.stable_seconds:
            self.current_label = label
            self.current_probs = trio_probs
            self.candidate_label = None
            self.candidate_start = None
            return True
        return False

    def stable_probs(self):
        return self.current_probs

# ---------------- Utils ----------------
HAPPY_GAIN = 1.25
SAD_GAIN   = 0.80
NEUT_GAIN  = 1.00

def softmax_from_vals(vals_dict):
    keys=list(vals_dict.keys())
    vals=np.array([vals_dict[k] for k in keys],dtype=np.float32)
    ex=np.exp(vals-vals.max()); probs=ex/(ex.sum()+1e-9)
    return {k:float(p) for k,p in zip(keys,probs)}

def map_to_trio(emodict):
    # Raw FER
    sad_raw = float(emodict.get("sad",0.0))
    sats = float(emodict.get("angry",0.0)) + float(emodict.get("disgust",0.0)) + float(emodict.get("fear",0.0))
    sad_sum = sad_raw + SAD_SATELLITE_WEIGHT * sats
    trio_vals = {
        "neutral": float(emodict.get("neutral",0.0)),
        "happy":   float(emodict.get("happy",0.0)),
        "sad":     float(sad_sum)
    }
    # Apply gentle gains before softmax (bias happy up, sad down)
    trio_vals["neutral"] *= NEUT_GAIN
    trio_vals["happy"]   *= HAPPY_GAIN
    trio_vals["sad"]     *= SAD_GAIN
    return softmax_from_vals(trio_vals)

def apply_clahe_bgr(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l2=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2,a,b)),cv2.COLOR_LAB2BGR)

# --- Strict MTCNN requirement for High ---
def ensure_mtcnn_strict():
    """
    Ensure 'mtcnn' is importable for High mode.
    1) Try import.
    2) If missing, attempt auto-install into CURRENT interpreter.
    3) If still missing, EXIT with error (no fallback).
    """
    try:
        import importlib
        importlib.import_module("mtcnn")
        return
    except Exception:
        pass

    print("[Setup] 'mtcnn' not found. Attempting auto-install for High mode…")
    try:
        cmd = f"{shlex.quote(sys.executable)} -m pip install mtcnn"
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        print(f"[ERROR] pip install mtcnn failed: {e}")
        sys.exit(2)

    # Re-try import
    try:
        import importlib
        importlib.import_module("mtcnn")
        print("[Setup] 'mtcnn' installed successfully for High mode.")
    except Exception:
        print("[ERROR] 'mtcnn' still not importable after install. Cannot run High mode.")
        sys.exit(2)

# ---------------- Prompts (UNCHANGED) ----------------
def prompt_feature():
    while True:
        print("\nSelect feature:\n 1) Face only\n 2) Gesture only\n 3) Both face + gesture")
        c=input("Enter 1,2,or3 [default3]: ").strip()
        if c in ("1","2","3",""): return {"1":"face","2":"gesture","3":"both","":"both"}[c]

def prompt_quality():
    while True:
        print("\nSelect quality:\n 1) High (MTCNN=True)\n 2) Low  (MTCNN=False)")
        c=input("Enter 1 or 2 [default2]: ").strip()
        if c in ("1","2",""): return {"1":"high","2":"low","":"low"}[c]

# ---------------- Gesture utils (UNCHANGED) ----------------
mp_hands=mp.solutions.hands
_draw=mp.solutions.drawing_utils
_styles=mp.solutions.drawing_styles

def make_gesture_extractor():
    return mp_hands.Hands(static_image_mode=False,max_num_hands=1,
                          model_complexity=1,min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

def extract_gesture_features(hand_landmarks):
    bx=hand_landmarks.landmark[0].x; by=hand_landmarks.landmark[0].y
    feat=[]
    for lm in hand_landmarks.landmark:
        feat.append(round(lm.x-bx,4)); feat.append(round(lm.y-by,4))
    return np.array(feat,dtype=np.float32).reshape(1,-1)

def draw_hand_landmarks(img,hl):
    _draw.draw_landmarks(img,hl,mp_hands.HAND_CONNECTIONS,
                         _styles.get_default_hand_landmarks_style(),
                         _styles.get_default_hand_connections_style())

# --- Lists for strict hand enforcement (UNCHANGED) ---
RIGHT_HAND_GESTURES = ["play_right", "pause_right", "next_right", "previous_right", "thumbs_up_right", "stop_right"]
LEFT_HAND_GESTURES  = ["volume_down_left", "volume_up_left", "like_left", "skip30_left", "thumbs_down_left"]

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--width",type=int,default=960)
    ap.add_argument("--height",type=int,default=540)
    ap.add_argument("--gesture_model",default="gesture_model.pkl")
    ap.add_argument("--gesture_scaler",default="scaler.pkl")
    ap.add_argument("--mode",choices=["face","gesture","both"])
    ap.add_argument("--quality",choices=["high","low"])  # exactly two modes
    args=ap.parse_args()

    feature_mode=args.mode if args.mode else prompt_feature()
    include_face=feature_mode in ("face","both")
    include_gesture=feature_mode in ("gesture","both")

    # ---- Face init (STRICT High) ----
    detector=None
    mtcnn_flag=False
    quality_choice=None
    if include_face:
        quality_choice = args.quality if args.quality else prompt_quality()
        mtcnn_flag = (quality_choice == "high")

        if mtcnn_flag:
            # MUST ensure mtcnn exists; if not, try to install; if still not, EXIT (no fallback).
            ensure_mtcnn_strict()
            try:
                detector = FER(mtcnn=True)
            except Exception as e:
                print(f"[ERROR] FER(mtcnn=True) failed to initialize: {e}")
                sys.exit(2)
        else:
            # Low mode
            try:
                detector = FER(mtcnn=False)
            except Exception as e:
                print(f"[ERROR] FER(mtcnn=False) failed to initialize: {e}")
                sys.exit(2)

    # ---- Gesture init (UNCHANGED) ----
    gesture_model=gesture_scaler=None; hands=None
    if include_gesture:
        try:
            gesture_model=joblib.load(args.gesture_model)
            gesture_scaler=joblib.load(args.gesture_scaler)
            hands=make_gesture_extractor()
            print("[Gesture] Loaded model + scaler.")
        except Exception as e:
            print("[Gesture] load failed", e)

    # ---- Camera ----
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    if not cap.isOpened(): print("Cannot open cam"); sys.exit(1)
    font=cv2.FONT_HERSHEY_SIMPLEX; mirror=True; use_clahe=True

    stabilizer=EmotionStabilizer()
    current_gesture="None"; current_prob=0.0

    print("Press 'q' to quit, 'c' to toggle CLAHE.")
    while True:
        ok,frame=cap.read()
        if not ok: break
        if mirror: frame=cv2.flip(frame,1)
        frame_out=frame.copy()

        # Default emotion (if no face)
        emotion_to_send = {"neutral":1.0, "happy":0.0, "sad":0.0}

        # -------- Face (UNCHANGED UI; STRICT behavior for High) --------
        if include_face and detector is not None:
            rgb=cv2.cvtColor(apply_clahe_bgr(frame_out) if use_clahe else frame_out,cv2.COLOR_BGR2RGB)
            try:
                results=detector.detect_emotions(rgb)
            except Exception as e:
                # STRICT: if High was chosen, any runtime failure exits immediately (no fallback)
                if mtcnn_flag:
                    print(f"[ERROR] Runtime failure in High mode (mtcnn=True): {e}")
                    sys.exit(2)
                else:
                    # Low mode: report but keep running, just no face for this frame
                    results = []

            if results:
                best=max(results,key=lambda r:r["box"][2]*r["box"][3])
                x,y,w,h=best["box"]
                if (w*h)>=MIN_FACE_AREA:
                    trio = map_to_trio(best["emotions"])
                    _ = stabilizer.update(trio)
                    stable_probs = stabilizer.stable_probs()
                    emotion_to_send = stable_probs  # send stabilized when face present

                    # UI (as before)
                    primary=max(stable_probs.items(),key=lambda kv:kv[1])
                    cv2.rectangle(frame_out,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame_out,f"Primary(stable): {primary[0]} ({primary[1]:.2f})",
                                (10,65),font,0.8,(255,255,255),2,cv2.LINE_AA)
                    # Bars
                    bar_x,bar_y=10,int(args.height*0.65); bar_w,bar_h,gap=260,18,8
                    cv2.putText(frame_out,f"Stable={STABLE_SECONDS:.1f}s | Neutral/Happy/Sad",
                                (bar_x,bar_y-15),font,0.6,(200,200,200),1,cv2.LINE_AA)
                    for i,e in enumerate(["neutral","happy","sad"]):
                        p=float(stable_probs[e]); length=int(p*bar_w); yrow=bar_y+i*(bar_h+gap)
                        cv2.rectangle(frame_out,(bar_x,yrow),(bar_x+bar_w,yrow+bar_h),(70,70,70),1)
                        cv2.rectangle(frame_out,(bar_x,yrow),(bar_x+length,yrow+bar_h),(255,255,255),-1)
                        cv2.putText(frame_out,f"{e:>7}: {p:.2f}",
                                    (bar_x+bar_w+12,yrow+bar_h-4),font,0.55,(230,230,230),1,cv2.LINE_AA)
                else:
                    # Face too small → send neutral-only this frame
                    emotion_to_send = {"neutral":1.0, "happy":0.0, "sad":0.0}
                    cv2.putText(frame_out,"Face too small",(10,30),font,0.8,(0,0,255),2,cv2.LINE_AA)
            else:
                # No face → send neutral-only this frame
                emotion_to_send = {"neutral":1.0, "happy":0.0, "sad":0.0}
                cv2.putText(frame_out,"No face detected",(10,30),font,0.8,(0,0,255),2,cv2.LINE_AA)

        # -------- Gesture (UNCHANGED UI) --------
        if include_gesture and gesture_model and gesture_scaler and hands:
            rgb_for_hands=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            result=hands.process(rgb_for_hands)
            gesture_text="Gesture: None"; prob_text=""
            current_gesture,current_prob="None",0.0
            if result.multi_hand_landmarks and result.multi_handedness:
                hl=result.multi_hand_landmarks[0]
                handed=result.multi_handedness[0].classification[0].label
                draw_hand_landmarks(frame_out,hl)
                feat=extract_gesture_features(hl)
                feat_scaled=gesture_scaler.transform(feat)
                probs=gesture_model.predict_proba(feat_scaled)[0]
                idx=int(np.argmax(probs))
                classes=getattr(gesture_model,"classes_",None)
                pred_cls=classes[idx] if classes is not None else str(idx)
                pred_prob=float(probs[idx])
                if pred_cls in RIGHT_HAND_GESTURES and handed != "Right":
                    pred_cls, pred_prob = "None", 0.0
                elif pred_cls in LEFT_HAND_GESTURES and handed != "Left":
                    pred_cls, pred_prob = "None", 0.0
                current_gesture,current_prob=pred_cls,pred_prob
                gesture_text=f"Gesture: {pred_cls}"
                prob_text=f"P={pred_prob:.2f}" if pred_cls!="None" else ""
            gy=int(args.height*0.1)+25
            cv2.putText(frame_out,gesture_text,(10,gy),font,0.8,(180,180,180),2,cv2.LINE_AA)
            if prob_text:
                cv2.putText(frame_out,prob_text,(10,gy+28),font,0.7,(180,180,180),2,cv2.LINE_AA)

        # ---- Send to orchestrator ----
        if include_face:
            send_state(current_gesture, current_prob, emotion_to_send)
        else:
            send_state(current_gesture, current_prob, {})

        # Window title (UNCHANGED)
        window_title={"face":"Face Only","gesture":"Gesture Only","both":"Face + Gesture"}[feature_mode]
        cv2.imshow(window_title,frame_out)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('c'): use_clahe=not use_clahe

    cap.release(); cv2.destroyAllWindows()
    try:
        if _sock: _sock.close()
    except Exception: pass

if __name__=="__main__":
    main()
