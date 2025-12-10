# maintesting_spotify.py
# Live gesture → Spotify control (8 gestures + "none" ignored)
# Gestures:
#   Right  : play_right, pause_right, next_right, previous_right
#   Left   : volume_up_left, volume_down_left, like_left, skip30_left
#   None   : "none" (no action)
#
# Env you need:
#   SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, (optional) SPOTIPY_REDIRECT_URI
# Optional env:
#   GESTURE_CAM_INDEX=0|1|2
#   GESTURE_MIRROR=0|1
#   GESTURE_CONF_THRESHOLD=0.75
#   GESTURE_STABLE_FRAMES=5
#   SPOTIFY_CACHE_PATH=.cache-gesture-session

import os, sys, time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ======== Camera / Platform ========
IS_MAC = (sys.platform == "darwin")
CAM_INDEX = int(os.getenv("GESTURE_CAM_INDEX", "0"))

env_mirror = os.getenv("GESTURE_MIRROR")
if env_mirror is not None:
    MIRROR_FEED = env_mirror in ("1", "true", "True")
else:
    MIRROR_FEED = IS_MAC  # default: mirror on macOS only

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

def open_camera(idx: int):
    if IS_MAC:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap

# ======== Model ========
MODEL_PATH, SCALER_PATH = "gesture_model.pkl", "scaler.pkl"
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise FileNotFoundError("Missing gesture_model.pkl or scaler.pkl. Train first.")

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
CLASSES = list(model.classes_) if hasattr(model, "classes_") else None

# ======== MediaPipe Hands ========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
draw = mp.solutions.drawing_utils

def to_feature_vec(hand_landmarks):
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    vec = []
    for lm in hand_landmarks.landmark:
        vec.append(lm.x - base_x)
        vec.append(lm.y - base_y)
    return np.array(vec, dtype=np.float32).reshape(1, -1)  # (1,42)

# ======== Spotify Auth ========
SPOTIPY_CLIENT_ID     = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI  = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SPOTIFY_CACHE_PATH    = os.getenv("SPOTIFY_CACHE_PATH", ".cache-gesture-session")

if not (SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET):
    raise EnvironmentError("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.")

SCOPES = "user-modify-playback-state user-read-playback-state user-library-modify"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPES,
    cache_path=SPOTIFY_CACHE_PATH,
))

def get_device_id():
    devs = sp.devices().get("devices", [])
    if not devs:
        return None
    active = [d for d in devs if d.get("is_active")]
    return (active[0] if active else devs[0]).get("id")

# ======== Spotify Actions ========
ACTION_COOLDOWN_SEC = float(os.getenv("GESTURE_ACTION_COOLDOWN", "1.0"))
_last_action_at = 0.0
def cooldown_ok():
    global _last_action_at
    now = time.time()
    if now - _last_action_at >= ACTION_COOLDOWN_SEC:
        _last_action_at = now
        return True
    return False

def do_play():
    if cooldown_ok():
        did = get_device_id()
        sp.start_playback(device_id=did)

def do_pause():
    if cooldown_ok():
        did = get_device_id()
        sp.pause_playback(device_id=did)

def do_next():
    if cooldown_ok():
        did = get_device_id()
        sp.next_track(device_id=did)

def do_prev():
    if cooldown_ok():
        did = get_device_id()
        sp.previous_track(device_id=did)

def do_volume_change(delta=+10):
    if not cooldown_ok():
        return
    devs = sp.devices().get("devices", [])
    if not devs: return
    cur = [x for x in devs if x.get("is_active")]
    cur = (cur[0] if cur else devs[0])
    v = cur.get("volume_percent", 50)
    new_v = max(0, min(100, v + delta))
    sp.volume(new_v, device_id=cur.get("id"))

def do_like_current():
    if not cooldown_ok():
        return
    pb = sp.current_playback()
    if pb and pb.get("item"):
        tid = pb["item"]["id"]
        if tid:
            sp.current_user_saved_tracks_add([tid])

def do_seek_forward(ms=30000):
    if not cooldown_ok():
        return
    pb = sp.current_playback()
    if not pb or not pb.get("item"):
        return
    pos = pb.get("progress_ms", 0)
    dur = pb["item"].get("duration_ms", 0)
    new_pos = min(max(0, pos + ms), max(0, dur - 1000))
    sp.seek_track(new_pos, device_id=get_device_id())

ACTIONS = {
    # Right hand
    "play_right":       do_play,
    "pause_right":      do_pause,
    "next_right":       do_next,
    "previous_right":   do_prev,
    # Left hand
    "volume_up_left":    lambda: do_volume_change(+10),
    "volume_down_left":  lambda: do_volume_change(-10),
    "like_left":         do_like_current,
    "skip30_left":       lambda: do_seek_forward(30000),
}

# ======== Confidence + Smoothing ========
CONF_THRESHOLD = float(os.getenv("GESTURE_CONF_THRESHOLD", "0.75"))
STABLE_FRAMES  = int(os.getenv("GESTURE_STABLE_FRAMES",  "5"))
history = deque(maxlen=STABLE_FRAMES)

def stable_decision(probs, labels):
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx]
    top_prob  = float(probs[top_idx])

    if top_prob < CONF_THRESHOLD or top_label == "none":
        history.append("none")
        if history.count("none") == STABLE_FRAMES:
            return "none", top_prob
        return "none", top_prob

    history.append(top_label)
    if history.count(top_label) == STABLE_FRAMES:
        return top_label, top_prob
    return "none", top_prob

# ======== Main Loop ========
def main():
    cap = open_camera(CAM_INDEX)
    print("🎵 Gesture→Spotify running. Press 'q' to quit.")
    print(f"Classes: {CLASSES}")
    print(f"Mirror:{MIRROR_FEED}  Thr:{CONF_THRESHOLD}  Stable:{STABLE_FRAMES}")

    while True:
        ok, img = cap.read()
        if not ok:
            continue

        if MIRROR_FEED:
            img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        overlay = img.copy()
        shown_label, shown_prob = "none", 0.0

        if res.multi_hand_landmarks and res.multi_handedness:
            hand_lms = res.multi_hand_landmarks[0]
            draw.draw_landmarks(overlay, hand_lms, mp_hands.HAND_CONNECTIONS)

            feat = to_feature_vec(hand_lms)
            feat_s = scaler.transform(feat)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat_s)[0]
                labels = model.classes_
            else:
                pred = model.predict(feat_s)[0]
                labels = np.array(CLASSES)
                probs = np.ones(len(labels)) / len(labels)
                probs[labels.tolist().index(pred)] = 1.0

            stable_label, top_prob = stable_decision(probs, labels)
            shown_label, shown_prob = stable_label, top_prob

            if stable_label != "none":
                action = ACTIONS.get(stable_label)
                if action:
                    action()

        # HUD
        cv2.putText(overlay, f"Pred: {shown_label}  p={shown_prob:.2f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(overlay, f"Mirror:{MIRROR_FEED}  Thr:{CONF_THRESHOLD}  N:{STABLE_FRAMES}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)
        cv2.imshow("🎛️ Gesture → Spotify", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# GESTURE_MIRROR=1 GESTURE_CAM_INDEX=0 python3 maintesting_spotify.py
