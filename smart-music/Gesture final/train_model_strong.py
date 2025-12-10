import os, sys, time, socket
import cv2
import mediapipe as mp
import joblib
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ------------ Settings ------------
CONF_THRESHOLD = float(os.getenv("GESTURE_CONF_THRESHOLD", "0.80"))
COOLDOWN_SEC   = float(os.getenv("GESTURE_ACTION_COOLDOWN", "1.0"))
IS_MAC         = (sys.platform == "darwin")
CAM_INDEX      = int(os.getenv("GESTURE_CAM_INDEX", "0"))

# Mirror ON by default only on macOS; override with GESTURE_MIRROR=0/1
env_mirror = os.getenv("GESTURE_MIRROR")
if env_mirror is not None:
    MIRROR_FEED = env_mirror in ("1","true","True")
else:
    MIRROR_FEED = IS_MAC

# ------------ Spotify Auth ------------
sp_oauth = SpotifyOAuth(
    client_id="1dac75a3e61745f7bc7cbc82ff882af3",
    client_secret="1ebe2efb8edb4d4aabaaecf2de4cbeaa",
    redirect_uri="http://localhost:8888/callback",
    scope="user-modify-playback-state user-read-playback-state user-library-modify"
)
sp = spotipy.Spotify(auth_manager=sp_oauth)

# Global timeout for network ops
socket.setdefaulttimeout(5)

def refresh_spotify_token():
    global sp
    token_info = sp_oauth.get_cached_token()
    if token_info and sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            sp = spotipy.Spotify(auth=token_info['access_token'])
        except Exception as e:
            print("🌐 Token refresh failed:", e)

def device_id():
    try:
        refresh_spotify_token()
        devs = sp.devices().get("devices", [])
        if not devs: return None
        active = [d for d in devs if d.get("is_active")]
        return (active[0] if active else devs[0]).get("id")
    except Exception as e:
        print("⚠️ Device fetch failed:", e)
        return None

# ------------ Model & MediaPipe ------------
model  = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")
CLASSES = list(model.classes_) if hasattr(model, "classes_") else []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
draw = mp.solutions.drawing_utils

# ------------ Camera ------------
if IS_MAC:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

print(f"🎵 Spotify Gesture Controller — cam={CAM_INDEX} mirror={MIRROR_FEED} thr={CONF_THRESHOLD}")

# ------------ Spotify actions ------------
_last_action_time = 0
_last_toggle = None

def _cooldown_ok():
    global _last_action_time
    now = time.time()
    if now - _last_action_time >= COOLDOWN_SEC:
        _last_action_time = now
        return True
    return False

def play_current():
    did = device_id()
    if did and _cooldown_ok():
        try: sp.start_playback(device_id=did); print("▶️ play")
        except Exception as e: print("⚠️ play failed:", e)

def pause_current():
    did = device_id()
    if did and _cooldown_ok():
        try: sp.pause_playback(device_id=did); print("⏸ pause")
        except Exception as e: print("⚠️ pause failed:", e)

def next_song():
    did = device_id()
    if did and _cooldown_ok():
        try: sp.next_track(device_id=did); print("⏭ next")
        except Exception as e: print("⚠️ next failed:", e)

def previous_song():
    did = device_id()
    if did and _cooldown_ok():
        try: sp.previous_track(device_id=did); print("⏮ prev")
        except Exception as e: print("⚠️ previous failed:", e)

def volume_change(delta):
    did = device_id()
    if not did or not _cooldown_ok(): return
    try:
        devs = sp.devices().get("devices", [])
        cur = next((d for d in devs if d.get("is_active")), devs[0] if devs else None)
        if not cur: return
        v = cur.get("volume_percent", 50)
        new_v = max(0, min(100, v + delta))
        sp.volume(new_v, device_id=cur.get("id")); print(f"🔊 volume {new_v}%")
    except Exception as e:
        print("⚠️ volume change failed:", e)

def like_current():
    if not _cooldown_ok(): return
    try:
        pb = sp.current_playback()
        tid = pb.get("item", {}).get("id") if pb else None
        if tid: sp.current_user_saved_tracks_add([tid]); print("❤️ liked")
    except Exception as e:
        print("⚠️ like failed:", e)

def seek_forward(ms=30000):
    did = device_id()
    if not did or not _cooldown_ok(): return
    try:
        pb = sp.current_playback()
        if not pb or not pb.get("item"): return
        pos = pb.get("progress_ms", 0)
        dur = pb["item"].get("duration_ms", 0)
        new_pos = min(max(0, pos + ms), max(0, dur - 1000))
        sp.seek_track(new_pos, device_id=did); print(f"⏩ +{ms//1000}s")
    except Exception as e:
        print("⚠️ seek failed:", e)

# Map your 8 labels (with handedness) to actions
def handle_label(lbl):
    global _last_toggle
    if lbl == "none": return
    # Right-hand playback
    if lbl == "play_right":
        if _last_toggle != "play": play_current(); _last_toggle = "play"
    elif lbl == "pause_right":
        if _last_toggle != "pause": pause_current(); _last_toggle = "pause"
    elif lbl == "next_right":
        next_song()
    elif lbl == "previous_right":
        previous_song()
    # Left-hand utilities
    elif lbl == "volume_up_left":
        volume_change(+10)
    elif lbl == "volume_down_left":
        volume_change(-10)
    elif lbl == "like_left":
        like_current()
    elif lbl == "skip30_left":
        seek_forward(30000)

# ------------ Main loop ------------
last_heartbeat = time.time()

try:
    while True:
        ok, img = cap.read()
        if not ok: continue
        if MIRROR_FEED: img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        gesture_text = "No hand detected"
        now = time.time()
        if now - last_heartbeat > 10:
            print("💓 alive:", time.strftime("%H:%M:%S"))
            last_heartbeat = now

        if res.multi_hand_landmarks:
            hand_lm = res.multi_hand_landmarks[0]
            draw.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)

            # wrist-relative features (42)
            bx, by = hand_lm.landmark[0].x, hand_lm.landmark[0].y
            feat = []
            for lm in hand_lm.landmark:
                feat.append(round(lm.x - bx, 4))
                feat.append(round(lm.y - by, 4))

            Xs = scaler.transform([feat])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xs)[0]
                idx = int(np.argmax(probs))
                pred = CLASSES[idx]
                p = float(probs[idx])
            else:
                pred = model.predict(Xs)[0]
                p = 1.0

            if pred != "none" and p >= CONF_THRESHOLD:
                gesture_text = f"✅ {pred} ({p:.2f})"
                handle_label(pred)
            else:
                gesture_text = "❌ Not a gesture"
        else:
            pred, p = "none", 0.0

        cv2.putText(img, gesture_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(img, f"Mirror:{MIRROR_FEED} Thr:{CONF_THRESHOLD}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)
        cv2.imshow("🎵 Spotify Gesture Controller", img)

        if cv2.waitKey(1) & 0xFF == 'q':
            print("🛑 Quit requested.")
            break

except KeyboardInterrupt:
    print("💣 Interrupted.")

finally:
    cap.release()
    cv2.destroyAllWindows()
