from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import os
import sys
import joblib
import numpy as np
import cv2
import mediapipe as mp
import base64
from PIL import Image
import io
import json
import datetime
import traceback
import shutil

# Ensure Windows console can emit Unicode characters for logging
if os.name == "nt":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

import spotipy
from spotipy.oauth2 import SpotifyOAuth 
import socket
import queue
import threading
import time

# At the top of app.py, add:
def get_spotify_credentials():
    """Get Spotify credentials from environment or Config"""
    return {
        'SPOTIPY_CLIENT_ID': os.environ.get('SPOTIPY_CLIENT_ID') or getattr(Config, 'SPOTIPY_CLIENT_ID', ''),
        'SPOTIPY_CLIENT_SECRET': os.environ.get('SPOTIPY_CLIENT_SECRET') or getattr(Config, 'SPOTIPY_CLIENT_SECRET', ''),
        'SPOTIPY_REDIRECT_URI': os.environ.get('SPOTIPY_REDIRECT_URI') or getattr(Config, 'SPOTIPY_REDIRECT_URI', 'http://127.0.0.1:8888/callback')
    }

# Use this function everywhere you need credentials

# Import FER for face emotion detection
try:
    from fer import FER
    emotion_detector = FER(mtcnn=False)  # Use non-MTCNN for faster detection
    print("[OK] FER (face emotion detection) initialized")
except ImportError as e:
    print(f"[WARN] FER not available: {e}")
    print("       Install with: pip install fer opencv-python tensorflow")
    emotion_detector = None
except Exception as e:
    print(f"[WARN] FER initialization failed: {e}")
    emotion_detector = None

# Import configuration
try:
    from config import Config
except ImportError:
    print("Warning: config.py not found, using default values")
    class Config:
        GESTURE_CONFIDENCE_THRESHOLD = 0.8
        GESTURE_STABLE_FRAMES = 5
        GESTURE_ACTION_COOLDOWN = 1.0
        DJ_DEFAULT_BATCH_SIZE = 150
        DJ_STRICT_PRIMARY = True

# Add paths for gesture and model modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
GESTURE_DIR = os.path.join(PROJECT_ROOT, 'Gesture final')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Models')
FINA_RECOM_DIR = os.path.join(PROJECT_ROOT, 'fina recom')
FINA_CACHE_FILE = os.path.join(FINA_RECOM_DIR, '.cache_spotify_export')
FINA_METADATA_FILE = os.path.join(FINA_RECOM_DIR, 'metadata.json')

for path in (GESTURE_DIR, MODELS_DIR):
    if path not in sys.path:
        sys.path.append(path)


class ProfileGenerationError(Exception):
    """Raised when Fina Recom profile generation fails."""

    def __init__(self, message, stdout='', stderr=''):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


def _backend_cache_file():
    """Resolve the absolute path to the backend Spotify cache file."""
    cache_name = os.environ.get('SPOTIFY_CACHE_PATH', '.cache-dj-session')
    return cache_name if os.path.isabs(cache_name) else os.path.join(BASE_DIR, cache_name)


def sync_spotify_token_cache(destination=FINA_CACHE_FILE):
    """Copy backend Spotify token cache to Fina Recom cache."""
    source = _backend_cache_file()
    result = {
        "source": source,
        "destination": destination,
        "copied": False,
    }

    if not os.path.exists(source):
        raise FileNotFoundError(f"Spotify cache not found at {source}")

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source, destination)
    result["copied"] = True
    return result


def _summarize_profile(profile):
    """Return a compact summary of the generated metadata profile."""
    return {
        "user_name": profile.get("user", {}).get("name", "Unknown"),
        "top_artists_count": len(profile.get("top_artists", {}).get("long_term", [])),
        "top_tracks_count": len(profile.get("top_tracks", {}).get("long_term", [])),
        "genres_count": len(profile.get("top_genres_all_time", [])),
        "playlists_summary": bool(profile.get("playlists_summary", {}).get("top_artists")),
    }


def _require_spotify_token():
    """Make sure we have a cached Spotify token before running downstream tasks."""
    oauth = _spotify_oauth()
    if not oauth:
        raise RuntimeError("Spotify OAuth not configured")
    token = oauth.get_cached_token()
    if not token:
        raise RuntimeError("Not authenticated with Spotify. Use /api/spotify/login first.")
    return token


def run_fina_recom_profile_generation(timeout=300):
    """Run combocode.py to generate metadata.json and return its summary."""
    import subprocess

    combocode_path = os.path.join(FINA_RECOM_DIR, 'combocode.py')
    if not os.path.exists(combocode_path):
        raise FileNotFoundError("Fina Recom combocode.py not found")

    env = os.environ.copy()
    env.update(get_spotify_credentials())
    env.setdefault('SPOTIFY_MARKET', getattr(Config, 'SPOTIFY_MARKET', 'IN'))
    env.setdefault('SPOTIFY_SCOPES', 'playlist-read-private playlist-read-collaborative user-top-read')
    env.setdefault('SPOTIFY_CACHE_PATH', '.cache_spotify_export')

    result = subprocess.run(
        [sys.executable, combocode_path],
        capture_output=True,
        text=True,
        cwd=FINA_RECOM_DIR,
        env=env,
        timeout=timeout
    )

    if result.returncode != 0:
        raise ProfileGenerationError(
            f"Profile generation failed with return code {result.returncode}",
            stdout=result.stdout,
            stderr=result.stderr
        )

    if not os.path.exists(FINA_METADATA_FILE):
        raise ProfileGenerationError(
            "Profile generation completed but metadata.json not found",
            stdout=result.stdout,
            stderr=result.stderr
        )

    with open(FINA_METADATA_FILE, 'r', encoding='utf-8') as fp:
        profile = json.load(fp)

    return {
        "metadata_path": FINA_METADATA_FILE,
        "summary": _summarize_profile(profile),
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def run_full_spotify_bootstrap():
    """Ensure we have a token, sync caches, and (re)generate metadata."""
    _require_spotify_token()
    cache_info = sync_spotify_token_cache()
    profile_info = run_fina_recom_profile_generation()
    return cache_info, profile_info


def _start_artgig_background(
    mode: str,
    tag_profile: str,
    artists: list,
    *,
    initial_batch: int = 50,
    topup_batch: int = 50,
    changes_per_topup: int = 10,
    seed_uris: list | None = None,
    skip_initial_queue: bool = False,
    sp_client=None,
    device_id=None,
):
    """Utility to launch an artgig pump loop asynchronously."""
    if not spotify_client:
        return False, "Artgig functionality not available"
    tags = list(TAG_PROFILES.get(str(tag_profile), [])) if tag_profile else []
    try:
        client = sp_client or spotify_client()
        device = device_id or ensure_active_device(client)
        if not device:
            return False, "No active Spotify device found. Please open Spotify."

        def run():
            try:
                pump_loop(
                    client,
                    device,
                    mode,
                    artists,
                    tags,
                    initial_batch=initial_batch,
                    topup_batch=topup_batch,
                    changes_per_topup=changes_per_topup,
                    seed_uris=seed_uris,
                    skip_initial_queue=skip_initial_queue,
                )
            except Exception as exc:  # pragma: no cover - background thread logging
                print(f"[artgig-autostart] Pump loop error: {exc}")

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return True, {
            "device_id": device,
            "tags": tags,
            "artists": artists,
            "initial_batch": initial_batch,
            "topup_batch": topup_batch,
            "changes_per_topup": changes_per_topup,
        }
    except Exception as exc:
        print(f"[artgig-autostart] Bootstrap failed: {exc}")
        return False, str(exc)

# Import your existing modules
try:
    from artists_gig_backfriend import run_once as dj_run_once
    print("DJ module imported successfully")
except ImportError as e:
    print(f"Warning: DJ module not available: {e}")
    dj_run_once = None

# Import artgig module
try:
    from artgig import (
        spotify_client, ensure_active_device, choose_tags_menu, 
        search_by_artists, search_random, start_and_queue, pump_loop,
        TAG_PROFILES
    )
    print("Artgig module imported successfully")
except ImportError as e:
    print(f"Warning: Artgig module not available: {e}")
    spotify_client = None

app = Flask(__name__)
# Allow frontend origins including local file server and dev ports
_cors_origins = getattr(Config, 'CORS_ORIGINS', [
    'http://localhost:3000', 'http://127.0.0.1:3000',
    'http://localhost:5000', 'http://127.0.0.1:5000',
    'http://localhost:5500', 'http://127.0.0.1:5500',
    'null'
])
CORS(app, resources={r"/*": {"origins": _cors_origins}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    try:
        origin = request.headers.get('Origin')
        if origin and origin in _cors_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Vary'] = 'Origin'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    except Exception:
        pass
    return response

# Load gesture recognition models
MODEL_PATH = getattr(Config, 'GESTURE_MODEL_PATH', "../Gesture final/gesture_model.pkl")
SCALER_PATH = getattr(Config, 'GESTURE_SCALER_PATH', "../Gesture final/scaler.pkl")

try:
    gesture_model = joblib.load(MODEL_PATH)
    gesture_scaler = joblib.load(SCALER_PATH)
    print("Gesture models loaded successfully")
    print(f"Model classes: {list(gesture_model.classes_) if hasattr(gesture_model, 'classes_') else 'Unknown'}")
except Exception as e:
    print(f"Error loading gesture models: {e}")
    gesture_model = None
    gesture_scaler = None

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ===== Helper function for Spotify OAuth =====
def _spotify_oauth():
    client_id = getattr(Config, 'SPOTIPY_CLIENT_ID', None)
    client_secret = getattr(Config, 'SPOTIPY_CLIENT_SECRET', None)
    redirect_uri = getattr(Config, 'SPOTIPY_REDIRECT_URI', 'http://localhost:5000/callback')
    scopes = os.environ.get(
        'SPOTIFY_SCOPES',
        'user-modify-playback-state user-read-playback-state user-read-currently-playing user-library-modify'
    )
    cache_path = os.environ.get('SPOTIFY_CACHE_PATH', '.cache-dj-session')
    if not client_id or not client_secret:
        print("⚠️ Spotify credentials not configured")
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scopes,
        cache_path=cache_path,
        open_browser=False,
    )

def get_spotify_client():
    """Get authenticated Spotify client for artist mix"""
    try:
        oauth = _spotify_oauth()
        token = oauth.get_cached_token()
        if not token:
            raise Exception("Not authenticated with Spotify")
        return spotipy.Spotify(auth=token['access_token'])
    except Exception as e:
        print(f"Error getting Spotify client: {e}")
        raise

@app.route('/')
def index():
    return jsonify({
        "message": "Smart Music Backend API", 
        "status": "running",
        "version": "1.0.0",
        "features": {
            "gesture_recognition": gesture_model is not None,
            "dj_control": dj_run_once is not None,
            "spotify_integration": True,
            "artist_mix": True,
            "mood_mixer": True
        }
    })

@app.route('/api/gesture/predict', methods=['POST'])
def predict_gesture():
    if not gesture_model or not gesture_scaler:
        return jsonify({"error": "Gesture models not loaded"}), 500
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Debug: Log image dimensions
        print(f"🔍 Image dimensions: {rgb_image.shape}")
        
        results = hands.process(rgb_image)
        
        # Debug: Log hand detection results
        if results.multi_hand_landmarks:
            print(f"✅ Hand detected! Number of hands: {len(results.multi_hand_landmarks)}")
            hand_landmarks = results.multi_hand_landmarks[0]
            print(f"📏 Hand landmarks: {len(hand_landmarks.landmark)} points")
            
            # Debug: Log first few landmark positions
            for i, landmark in enumerate(hand_landmarks.landmark[:5]):
                print(f"   Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
        else:
            print("❌ No hand detected in image")
            return jsonify({"gesture": "none", "confidence": 0.0, "message": "No hand detected"})
        
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Use your exact feature extraction method
        def to_feature_vec(hand_landmarks):
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            vec = []
            for lm in hand_landmarks.landmark:
                vec.append(lm.x - base_x)
                vec.append(lm.y - base_y)
            return np.array(vec, dtype=np.float32).reshape(1, -1)  # (1,42)
        
        features = to_feature_vec(hand_landmarks)
        
        # Debug: Log feature extraction
        print(f"🔢 Features extracted: {features.shape}")
        print(f"   First 5 features: {features[0][:5]}")
        print(f"   Last 5 features: {features[0][-5:]}")
        
        features_scaled = gesture_scaler.transform(features)
        
        # Debug: Log scaling results
        print(f"⚖️ Features scaled: {features_scaled.shape}")
        print(f"   First 5 scaled: {features_scaled[0][:5]}")
        
        if hasattr(gesture_model, 'predict_proba'):
            probabilities = gesture_model.predict_proba(features_scaled)[0]
            predicted_class = gesture_model.classes_[np.argmax(probabilities)]
            confidence = float(np.max(probabilities))
            
            # Debug: Log all class probabilities
            print(f"🎯 Model prediction results:")
            print(f"   Predicted class: {predicted_class}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   All probabilities:")
            for i, (cls, prob) in enumerate(zip(gesture_model.classes_, probabilities)):
                print(f"     {cls}: {prob:.3f}")
        else:
            predicted_class = gesture_model.predict(features_scaled)[0]
            confidence = 1.0
        
        threshold = getattr(Config, 'GESTURE_CONFIDENCE_THRESHOLD', 0.3)
        
        # Debug: Log threshold comparison
        print(f"🎚️ Confidence threshold: {threshold}")
        print(f"   Confidence {confidence:.3f} {'>=' if confidence >= threshold else '<'} threshold {threshold}")
        
        if confidence < threshold:
            predicted_class = "none"
            confidence = 0.0
            print(f"   ⚠️ Below threshold, setting to 'none'")
        else:
            print(f"   ✅ Above threshold, keeping prediction: {predicted_class}")
        
        return jsonify({
            "gesture": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.tolist() if hasattr(gesture_model, 'predict_proba') else None,
            "threshold": threshold
        })
        
    except Exception as e:
        print(f"❌ Gesture prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect/all', methods=['POST'])
def detect_all():
    """Combined endpoint that returns both gesture and emotion detection"""
    if not gesture_model or not gesture_scaler:
        return jsonify({"error": "Gesture models not loaded"}), 500
    
    if not emotion_detector:
        return jsonify({"error": "Emotion detector not available"}), 500
    
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        result = {
            "gesture": {"gesture": "none", "confidence": 0.0},
            "emotion": {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "detected": False}
        }
        
        # ===== GESTURE DETECTION =====
        try:
            hand_results = hands.process(rgb_image)
            
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                
                # Extract features
                def to_feature_vec(hand_landmarks):
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    vec = []
                    for lm in hand_landmarks.landmark:
                        vec.append(lm.x - base_x)
                        vec.append(lm.y - base_y)
                    return np.array(vec, dtype=np.float32).reshape(1, -1)
                
                features = to_feature_vec(hand_landmarks)
                features_scaled = gesture_scaler.transform(features)
                
                if hasattr(gesture_model, 'predict_proba'):
                    probabilities = gesture_model.predict_proba(features_scaled)[0]
                    predicted_class = gesture_model.classes_[np.argmax(probabilities)]
                    confidence = float(np.max(probabilities))
                else:
                    predicted_class = gesture_model.predict(features_scaled)[0]
                    confidence = 1.0
                
                threshold = getattr(Config, 'GESTURE_CONFIDENCE_THRESHOLD', 0.3)
                
                # Store raw prediction before threshold check
                raw_gesture = predicted_class
                raw_confidence = confidence
                
                # Only filter if below threshold (but still return raw data for debugging)
                if confidence < threshold:
                    predicted_class = "none"
                    # Keep confidence at 0 for filtered gestures, but include raw data
                else:
                    # Above threshold - gesture is valid
                    pass
                
                result["gesture"] = {
                    "gesture": predicted_class,
                    "confidence": confidence if confidence >= threshold else 0.0,
                    "raw_gesture": raw_gesture,  # Always include raw prediction
                    "raw_confidence": raw_confidence,  # Always include raw confidence
                    "threshold": threshold,
                    "probabilities": probabilities.tolist() if hasattr(gesture_model, 'predict_proba') else None
                }
        except Exception as e:
            print(f"Gesture detection error: {e}")
        
        # ===== EMOTION DETECTION =====
        try:
            # Detect emotions in the image
            emotions_data = emotion_detector.detect_emotions(rgb_image)
            
            if emotions_data and len(emotions_data) > 0:
                # Get the first (largest) face
                best_face = max(emotions_data, key=lambda r: r["box"][2] * r["box"][3])
                raw_emotions = best_face["emotions"]
                
                # Map to trio: neutral, happy, sad
                # FER gives us: angry, disgust, fear, happy, neutral, sad, surprise
                sad_raw = float(raw_emotions.get("sad", 0.0))
                angry_disgust_fear = float(raw_emotions.get("angry", 0.0)) + \
                                    float(raw_emotions.get("disgust", 0.0)) + \
                                    float(raw_emotions.get("fear", 0.0))
                sad_total = sad_raw + 0.25 * angry_disgust_fear
                
                trio_values = {
                    "neutral": float(raw_emotions.get("neutral", 0.0)),
                    "happy": float(raw_emotions.get("happy", 0.0)),
                    "sad": float(sad_total)
                }
                
                # Apply gains (bias happy up, sad down)
                trio_values["neutral"] *= 1.0
                trio_values["happy"] *= 1.25
                trio_values["sad"] *= 0.80
                
                # Softmax
                vals = np.array([trio_values["neutral"], trio_values["happy"], trio_values["sad"]], dtype=np.float32)
                ex = np.exp(vals - vals.max())
                probs = ex / (ex.sum() + 1e-9)
                
                result["emotion"] = {
                    "neutral": float(probs[0]),
                    "happy": float(probs[1]),
                    "sad": float(probs[2]),
                    "detected": True,
                    "raw": raw_emotions  # Include raw for debugging
                }
        except Exception as e:
            print(f"Emotion detection error: {e}")
        
        # ===== SEND TO ORCHESTRATOR =====
        # Send emotions and gestures to orchestrator if it's running
        if orchestrator_running and orchestrator_event_queue:
            try:
                # Send gesture
                if result["gesture"]["gesture"] != "none":
                    orchestrator_event_queue.put({
                        "type": "gesture",
                        "gesture": result["gesture"]["gesture"],
                        "prob": result["gesture"]["confidence"]
                    })
                
                # Send emotion if detected
                if result["emotion"]["detected"]:
                    # Find dominant emotion
                    emotion_vals = {
                        "neutral": result["emotion"]["neutral"],
                        "happy": result["emotion"]["happy"],
                        "sad": result["emotion"]["sad"]
                    }
                    dominant = max(emotion_vals, key=emotion_vals.get)
                    
                    orchestrator_event_queue.put({
                        "type": "emotion",
                        "label": dominant
                    })
            except Exception as e:
                print(f"Error sending to orchestrator: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ===== NEW ARTIST MIX ENDPOINTS =====

@app.route('/api/artist-mix/search', methods=['POST'])
def search_artist_mix():
    """
    Search for artist songs in a specific genre and play on Spotify
    
    Request Body:
    {
        "artist": "Artist Name",
        "genre": "Genre Name",
        "limit": 20
    }
    """
    try:
        data = request.json
        artist_name = data.get('artist', '').strip()
        genre = data.get('genre', '').strip()
        try:
            limit = int(data.get('limit', 20) or 20)
        except (TypeError, ValueError):
            limit = 20
        if limit < 1:
            limit = 1
        if not genre:
            limit = max(limit, 50)
        
        if not artist_name:
            return jsonify({
                'ok': False,
                'error': 'Artist name is required'
            }), 400
        
        print(f"Searching for artist: {artist_name}, genre: {genre or 'all'}")
        
        # Get Spotify client
        sp = get_spotify_client()
        
        # Check if Spotify is active
        devices = sp.devices()
        if not devices.get('devices'):
            return jsonify({
                'ok': False,
                'error': 'No active Spotify device found. Please open Spotify and start playing something.'
            }), 400
        
        # Search for the artist
        artist_results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
        
        if not artist_results['artists']['items']:
            return jsonify({
                'ok': False,
                'error': f'Artist "{artist_name}" not found'
            }), 404
        
        artist = artist_results['artists']['items'][0]
        artist_id = artist['id']
        artist_full_name = artist['name']
        
        print(f"Found artist: {artist_full_name} (ID: {artist_id})")
        
        # Build search query
        if genre:
            # Search with both artist and genre
            search_query = f'artist:{artist_name} genre:{genre}'
        else:
            # Search just by artist
            search_query = f'artist:{artist_name}'
        
        print(f"Search query: {search_query}")
        
        # Search for tracks (accumulate across multiple pages if needed)
        tracks = []
        offset = 0
        while len(tracks) < limit:
            batch_size = min(50, limit - len(tracks))
            track_results = sp.search(q=search_query, type='track', limit=batch_size, offset=offset)
            items = track_results['tracks']['items']
            if not items:
                break
            tracks.extend(items)
            if batch_size < 50:
                break
            offset += batch_size
        
        if not tracks:
            # Fallback: Get artist's top tracks
            print("No tracks found with genre, falling back to top tracks")
            tracks = sp.artist_top_tracks(artist_id)['tracks'][:limit]

        if not genre and len(tracks) < limit:
            # Attempt to supplement with album tracks to reach the minimum quota
            print(f"Only {len(tracks)} tracks found; fetching albums to reach {limit}")
            seen_uris = {t.get('uri') for t in tracks}
            album_offset = 0
            while len(tracks) < limit:
                albums = sp.artist_albums(artist_id, album_type='album,single,compilation', limit=20, offset=album_offset)
                album_items = albums.get('items', [])
                if not album_items:
                    break
                for album in album_items:
                    album_tracks = sp.album_tracks(album['id']).get('items', [])
                    for al_track in album_tracks:
                        al_track = dict(al_track)
                        al_track['album'] = album
                        if not al_track.get('uri') or al_track['uri'] in seen_uris:
                            continue
                        tracks.append(al_track)
                        seen_uris.add(al_track['uri'])
                        if len(tracks) >= limit:
                            break
                    if len(tracks) >= limit:
                        break
                if not albums.get('next'):
                    break
                album_offset += len(album_items)
        tracks = tracks[:limit]
        
        if not tracks:
            return jsonify({
                'ok': False,
                'error': f'No tracks found for {artist_full_name}' + (f' in {genre} genre' if genre else '')
            }), 404
        
        print(f"Found {len(tracks)} tracks")
        
        # Extract track URIs
        track_uris = [track['uri'] for track in tracks]
        
        # Get active device
        active_device = None
        for device in devices['devices']:
            if device['is_active']:
                active_device = device['id']
                break
        
        if not active_device and devices['devices']:
            active_device = devices['devices'][0]['id']
        
        print(f"Using device: {active_device}")
        
        # Start playing the first track
        sp.start_playback(device_id=active_device, uris=[track_uris[0]])
        print(f"Started playing: {tracks[0]['name']}")
        
        # Add remaining tracks to queue
        for uri in track_uris[1:]:
            sp.add_to_queue(uri, device_id=active_device)
        
        print(f"Added {len(track_uris) - 1} tracks to queue")

        auto_profile = genre if genre in TAG_PROFILES else None
        auto_started, auto_info = _start_artgig_background(
            'artist',
            auto_profile,
            [artist_full_name],
            initial_batch=50,
            topup_batch=50,
            changes_per_topup=10,
            seed_uris=track_uris,
            skip_initial_queue=True,
            sp_client=sp,
            device_id=active_device
        )
        
        # Prepare track info for response
        track_info = []
        for track in tracks:
            track_info.append({
                'name': track['name'],
                'artist': ', '.join([a['name'] for a in track['artists']]),
                'album': track['album']['name'],
                'uri': track['uri'],
                'duration_ms': track['duration_ms'],
                'image': track['album']['images'][0]['url'] if track['album']['images'] else None
            })
        
        return jsonify({
            'ok': True,
            'message': f'Playing {len(tracks)} tracks by {artist_full_name}' + (f' in {genre} genre' if genre else ''),
            'artist': artist_full_name,
            'genre': genre,
            'tracks_count': len(tracks),
            'tracks': track_info,
            'now_playing': track_info[0] if track_info else None,
            'auto_topup_started': auto_started,
            'auto_topup_info': auto_info if auto_started else {'error': auto_info}
        })
        
    except Exception as e:
        print(f"❌ Artist mix search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'ok': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/artist-mix/search-artists', methods=['GET'])
def search_artists():
    """
    Search for artists by name (for autocomplete/suggestions)
    
    Query Parameters:
    - q: Search query
    - limit: Number of results (default 10)
    """
    try:
        query = request.args.get('q', '').strip()
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({
                'ok': False,
                'error': 'Search query is required'
            }), 400
        
        sp = get_spotify_client()
        
        # Search for artists
        results = sp.search(q=f'artist:{query}', type='artist', limit=limit)
        
        artists = []
        for artist in results['artists']['items']:
            artists.append({
                'id': artist['id'],
                'name': artist['name'],
                'genres': artist['genres'],
                'popularity': artist['popularity'],
                'image': artist['images'][0]['url'] if artist['images'] else None,
                'followers': artist['followers']['total']
            })
        
        return jsonify({
            'ok': True,
            'artists': artists,
            'count': len(artists)
        })
    
    except Exception as e:
        print(f"❌ Artist search error: {e}")
        return jsonify({
            'ok': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/artist-mix/genres', methods=['GET'])
def get_genres():
    """Get available genres from Spotify"""
    try:
        sp = get_spotify_client()
        
        # Get available genre seeds
        genres = sp.recommendation_genre_seeds()
        
        return jsonify({
            'ok': True,
            'genres': sorted(genres['genres']),
            'count': len(genres['genres'])
        })
    
    except Exception as e:
        print(f"❌ Genre fetch error: {e}")
        # Return fallback genres if Spotify API fails
        fallback_genres = [
            'pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical', 
            'country', 'r&b', 'reggae', 'blues', 'folk', 'indie', 
            'alternative', 'metal', 'punk', 'funk', 'soul', 'gospel',
            'latin', 'world', 'ambient', 'dance', 'house', 'techno',
            'trance', 'dubstep', 'trap', 'lo-fi', 'chill', 'acoustic'
        ]
        
        return jsonify({
            'ok': True,
            'genres': sorted(fallback_genres),
            'count': len(fallback_genres),
            'fallback': True,
            'message': 'Using fallback genres due to Spotify API error'
        })

@app.route('/api/artist-mix/play-multiple', methods=['POST'])
def play_multiple_artists():
    """
    Play tracks from multiple artists
    
    Request Body:
    {
        "artists": ["Artist 1", "Artist 2", "Artist 3"],
        "genre": "Genre Name",  # optional
        "tracks_per_artist": 5,  # optional, default 5
        "shuffle": true  # optional, default true
    }
    """
    try:
        data = request.json
        artists = data.get('artists', [])
        topup_batch = int(data.get('topup_batch', 50) or 50)
        changes_per_topup = int(data.get('changes_per_topup', 10) or 10)
        if topup_batch < 10:
            topup_batch = 10
        if changes_per_topup < 1:
            changes_per_topup = 1
        genre = data.get('genre', '').strip()
        requested_tracks = int(data.get('tracks_per_artist', 5) or 5)
        tracks_per_artist = requested_tracks if genre else max(requested_tracks, 50)
        shuffle = data.get('shuffle', True)
        
        if not artists:
            return jsonify({
                'ok': False,
                'error': 'Artists list is required'
            }), 400
        
        print(f"Playing multiple artists: {artists}, genre: {genre or 'all'}")
        
        # Get Spotify client
        sp = get_spotify_client()
        
        # Check if Spotify is active
        devices = sp.devices()
        if not devices.get('devices'):
            return jsonify({
                'ok': False,
                'error': 'No active Spotify device found. Please open Spotify and start playing something.'
            }), 400
        
        all_tracks = []
        artist_info = {}
        
        # Process each artist
        for artist_name in artists:
            print(f"Processing artist: {artist_name}")
            
            # Search for the artist
            artist_results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
            
            if not artist_results['artists']['items']:
                print(f"Artist '{artist_name}' not found, skipping")
                continue
            
            artist = artist_results['artists']['items'][0]
            artist_id = artist['id']
            artist_full_name = artist['name']
            
            # Build search query
            if genre:
                search_query = f'artist:{artist_name} genre:{genre}'
            else:
                search_query = f'artist:{artist_name}'
            
            # Search for tracks
            track_results = sp.search(q=search_query, type='track', limit=tracks_per_artist)
            tracks = track_results['tracks']['items']
            
            if not tracks:
                # Fallback: Get artist's top tracks
                print(f"No tracks found with genre for {artist_full_name}, using top tracks")
                tracks = sp.artist_top_tracks(artist_id)['tracks'][:tracks_per_artist]
            
            if tracks:
                # Add artist info to tracks
                for track in tracks:
                    track['artist_name'] = artist_full_name
                    track['artist_id'] = artist_id
                
                all_tracks.extend(tracks)
                artist_info[artist_full_name] = {
                    'id': artist_id,
                    'tracks_found': len(tracks),
                    'target_per_artist': tracks_per_artist
                }
                print(f"Found {len(tracks)} tracks for {artist_full_name}")
            else:
                print(f"No tracks found for {artist_full_name}")
        
        if not all_tracks:
            return jsonify({
                'ok': False,
                'error': 'No tracks found for any of the specified artists'
            }), 404
        
        # Shuffle tracks if requested
        if shuffle:
            import random
            random.shuffle(all_tracks)
            print("Shuffled tracks")
        
        # Extract track URIs
        track_uris = [track['uri'] for track in all_tracks]
        
        # Get active device
        active_device = None
        for device in devices['devices']:
            if device['is_active']:
                active_device = device['id']
                break
        
        if not active_device and devices['devices']:
            active_device = devices['devices'][0]['id']
        
        print(f"Using device: {active_device}")
        
        # Start playing the first track
        sp.start_playback(device_id=active_device, uris=[track_uris[0]])
        print(f"Started playing: {all_tracks[0]['name']} by {all_tracks[0]['artist_name']}")
        
        # Add remaining tracks to queue
        for uri in track_uris[1:]:
            sp.add_to_queue(uri, device_id=active_device)
        
        print(f"Added {len(track_uris) - 1} tracks to queue")

        auto_profile = genre if genre in TAG_PROFILES else None
        auto_started, auto_info = _start_artgig_background(
            'artist',
            auto_profile,
            list(artist_info.keys()),
            initial_batch=tracks_per_artist,
            topup_batch=topup_batch,
            changes_per_topup=changes_per_topup,
            seed_uris=track_uris,
            skip_initial_queue=True,
            sp_client=sp,
            device_id=active_device
        )
        
        # Prepare response
        track_info = []
        for track in all_tracks:
            track_info.append({
                'name': track['name'],
                'artist': track['artist_name'],
                'duration_ms': track['duration_ms'],
                'image': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'uri': track['uri']
            })
        
        return jsonify({
            'ok': True,
            'message': f'Playing {len(track_uris)} tracks from {len(artist_info)} artists',
            'artists': list(artist_info.keys()),
            'artist_info': artist_info,
            'tracks': track_info,
            'tracks_count': len(track_uris),
            'genre': genre,
            'shuffled': shuffle,
            'auto_topup_started': auto_started,
            'auto_topup_info': auto_info if auto_started else {'error': auto_info}
        })
        
    except Exception as e:
        print(f"Error in play_multiple_artists: {e}")
        traceback.print_exc()
        return jsonify({
            'ok': False,
            'error': f'Server error: {str(e)}'
        }), 500

# ===== ARTGIG INTEGRATION ENDPOINTS =====

@app.route('/api/artgig/start', methods=['POST'])
def start_artgig_session():
    """
    Start an artgig DJ session with tag-based music discovery
    
    Request Body:
    {
        "mode": "artist" | "random",
        "tag_profile": "1" | "2" | "3" | "4",  # Tag profile selection
        "artists": ["Artist 1", "Artist 2"],  # Required for artist mode
        "initial_batch": 50,  # optional
        "topup_batch": 50,    # optional
        "changes_per_topup": 10  # optional
    }
    """
    if not spotify_client:
        return jsonify({"error": "Artgig functionality not available"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        mode = data.get('mode', 'random')
        tag_profile = data.get('tag_profile', '1')
        artists = data.get('artists', [])
        initial_batch = int(data.get('initial_batch', 50) or 50)
        topup_batch = int(data.get('topup_batch', 50) or 50)
        changes_per_topup = int(data.get('changes_per_topup', 10) or 10)
        if initial_batch < 10:
            initial_batch = 10
        if topup_batch < 10:
            topup_batch = 10
        if changes_per_topup < 1:
            changes_per_topup = 1
        
        # Validate inputs
        if mode not in ['random', 'artist']:
            return jsonify({"error": "Invalid mode. Must be 'random' or 'artist'"}), 400
            
        if tag_profile not in TAG_PROFILES:
            return jsonify({"error": f"Invalid tag profile. Must be one of: {list(TAG_PROFILES.keys())}"}), 400
            
        if mode == 'artist' and not artists:
            return jsonify({"error": "Artists list required for artist mode"}), 400
        
        ok, info = _start_artgig_background(
            mode,
            tag_profile,
            artists,
            initial_batch=initial_batch,
            topup_batch=topup_batch,
            changes_per_topup=changes_per_topup,
        )
        if not ok:
            return jsonify({'ok': False, 'error': info}), 500
        
        return jsonify({
            'ok': True,
            'message': f'Started artgig session with {mode} mode and tag profile {tag_profile}',
            'mode': mode,
            'tag_profile': tag_profile,
            'tags': info.get('tags', []),
            'artists': info.get('artists', []),
            'device_id': info.get('device_id'),
            'initial_batch': info.get('initial_batch'),
            'topup_batch': info.get('topup_batch'),
            'changes_per_topup': info.get('changes_per_topup')
        })
        
    except Exception as e:
        print(f"Artgig session error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/artgig/tag-profiles', methods=['GET'])
def get_tag_profiles():
    """Get available tag profiles for artgig"""
    try:
        if not TAG_PROFILES:
            return jsonify({"error": "Tag profiles not available"}), 500
            
        # Format profiles for frontend
        profiles = {}
        for key, tags in TAG_PROFILES.items():
            profiles[key] = {
                'tags': tags,
                'name': {
                    '1': 'Remix / Edits',
                    '2': 'Mashup',
                    '3': 'Lofi',
                    '4': 'Slowed + Reverb'
                }.get(key, f'Profile {key}')
            }
        
        return jsonify({
            'ok': True,
            'profiles': profiles
        })
        
    except Exception as e:
        print(f"Tag profiles error: {e}")
        return jsonify({"error": str(e)}), 500

# ===== END ARTGIG INTEGRATION ENDPOINTS =====

# ===== FINA RECOM INTEGRATION ENDPOINTS =====

@app.route('/api/fina-recom/status', methods=['GET'])
def fina_recom_status():
    """Check if Fina Recom system is available and user profile exists"""
    try:
        import os
        import json
        
        # Check if metadata.json exists
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'fina recom', 'metadata.json')
        profile_exists = os.path.exists(metadata_path)
        
        status = {
            "available": True,
            "profile_exists": profile_exists,
            "metadata_path": metadata_path
        }
        
        if profile_exists:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    status["profile_info"] = {
                        "user_name": profile.get("user", {}).get("name", "Unknown"),
                        "top_artists_count": len(profile.get("top_artists", {}).get("long_term", [])),
                        "top_tracks_count": len(profile.get("top_tracks", {}).get("long_term", [])),
                        "genres_count": len(profile.get("top_genres_all_time", [])),
                        "playlists_summary": bool(profile.get("playlists_summary", {}).get("top_artists"))
                    }
            except Exception as e:
                status["profile_error"] = str(e)
        
        return jsonify({
            "ok": True,
            "status": status
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Failed to check Fina Recom status: {str(e)}"
        }), 500

@app.route('/api/fina-recom/generate-profile', methods=['POST'])
def fina_recom_generate_profile():
    """Generate user profile using combocode.py"""
    try:
        cache_info, profile_info = run_full_spotify_bootstrap()
        return jsonify({
            "ok": True,
            "message": "Profile generated successfully",
            "cache_sync": cache_info,
            "profile_summary": profile_info["summary"],
            "metadata_path": profile_info["metadata_path"]
        })
    except FileNotFoundError as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "hint": "Authenticate via /api/spotify/login and retry."
        }), 400
    except ProfileGenerationError as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except RuntimeError as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Failed to generate profile: {str(e)}"
        }), 500

@app.route('/api/setup/bootstrap', methods=['POST'])
def spotify_full_bootstrap():
    """
    Convenience endpoint that performs the full Spotify setup:
    1. Ensure OAuth token exists
    2. Sync backend cache to fina recom cache
    3. Run combocode.py to regenerate metadata.json
    """
    try:
        cache_info, profile_info = run_full_spotify_bootstrap()
        return jsonify({
            "ok": True,
            "steps": {
                "token_verified": True,
                "cache_synced": cache_info,
                "metadata": profile_info["metadata_path"]
            },
            "profile_summary": profile_info["summary"]
        })
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except ProfileGenerationError as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Bootstrap failed: {str(e)}"}), 500

@app.route('/api/fina-recom/start-recommendations', methods=['POST'])
def fina_recom_start_recommendations():
    """Start Fina Recom recommendation system"""
    try:
        import subprocess
        import os
        import threading
        import time
        
        data = request.get_json() or {}
        mode = data.get('mode', 'balanced')
        
        # Validate mode
        valid_modes = ['comfort', 'balanced', 'explorer']
        if mode not in valid_modes:
            return jsonify({
                "ok": False,
                "error": f"Invalid mode. Must be one of: {valid_modes}"
            }), 400
        
        # Path to newmodel.py
        newmodel_path = os.path.join(os.path.dirname(__file__), '..', 'fina recom', 'newmodel.py')
        
        if not os.path.exists(newmodel_path):
            return jsonify({
                "ok": False,
                "error": "Fina Recom system not found"
            }), 404
        
        # Check if metadata.json exists
        metadata_path = os.path.join(os.path.dirname(newmodel_path), 'metadata.json')
        if not os.path.exists(metadata_path):
            return jsonify({
                "ok": False,
                "error": "User profile not found. Please generate profile first."
            }), 400
        
        # Start the recommendation system in a separate thread
        def run_recommendations():
            try:
                # Create a subprocess with the mode as input
                process = subprocess.Popen(
                    [sys.executable, newmodel_path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.dirname(newmodel_path)
                )
                
                # Send the mode selection
                mode_input = "1" if mode == "comfort" else "2" if mode == "balanced" else "3"
                stdout, stderr = process.communicate(input=mode_input)
                
                if process.returncode != 0:
                    print(f"Fina Recom error: {stderr}")
                else:
                    print(f"Fina Recom completed: {stdout}")
                    
            except Exception as e:
                print(f"Fina Recom thread error: {e}")
        
        # Start the thread
        thread = threading.Thread(target=run_recommendations, daemon=True)
        thread.start()
        
        return jsonify({
            "ok": True,
            "message": f"Fina Recom started in {mode} mode",
            "mode": mode,
            "status": "running"
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Failed to start recommendations: {str(e)}"
        }), 500

@app.route('/api/fina-recom/profile', methods=['GET'])
def fina_recom_get_profile():
    """Get user profile data"""
    try:
        import os
        import json
        
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'fina recom', 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return jsonify({
                "ok": False,
                "error": "Profile not found. Please generate profile first."
            }), 404
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        
        # Return a cleaned version of the profile (remove sensitive data)
        cleaned_profile = {
            "user": profile.get("user", {}),
            "top_artists": profile.get("top_artists", {}),
            "top_tracks": profile.get("top_tracks", {}),
            "top_albums": profile.get("top_albums", []),
            "top_genres_all_time": profile.get("top_genres_all_time", []),
            "playlists_summary": profile.get("playlists_summary", {}),
            "eras": profile.get("eras", {})
        }
        
        return jsonify({
            "ok": True,
            "profile": cleaned_profile
        })
        
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Failed to load profile: {str(e)}"
        }), 500

# ===== END FINA RECOM INTEGRATION ENDPOINTS =====

# ===== MOOD MIXER INTEGRATION ENDPOINTS =====

@app.route('/api/mood-mixer/start', methods=['POST'])
def start_mood_mixer():
    """
    Start the Spotify Mood Mixer with custom mood weights and language selection
    
    Request Body:
    {
        "language": "hi" | "en" | "mix",
        "mood_weights": {
            "happy": 20,
            "sad_breakup": 10,
            "motivational": 15,
            "chill_lofi": 15,
            "hiphop": 10,
            "rap": 10,
            "old_classic": 10,
            "romantic": 5,
            "party": 5
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        language = data.get('language', 'mix')
        mood_weights = data.get('mood_weights', {})
        
        # Validate language
        if language not in ['hi', 'en', 'mix']:
            return jsonify({"error": "Invalid language. Must be 'hi', 'en', or 'mix'"}), 400
        
        # Validate mood weights
        valid_moods = ['happy', 'sad_breakup', 'motivational', 'chill_lofi', 'hiphop', 'rap', 'old_classic', 'romantic', 'party']
        total_weight = 0
        
        for mood in valid_moods:
            weight = mood_weights.get(mood, 0)
            if weight < 0 or weight > 100 or weight % 5 != 0:
                return jsonify({"error": f"Invalid weight for {mood}. Must be 0-100, multiple of 5"}), 400
            total_weight += weight
        
        if total_weight != 100:
            return jsonify({"error": f"Total weight must equal 100, got {total_weight}"}), 400
        
        # Get Spotify client
        sp = get_spotify_client()
        
        # Check if Spotify is active
        devices = sp.devices()
        if not devices.get('devices'):
            return jsonify({
                'ok': False,
                'error': 'No active Spotify device found. Please open Spotify and start playing something.'
            }), 400
        
        # Get active device
        active_device = None
        for device in devices['devices']:
            if device['is_active']:
                active_device = device['id']
                break
        
        if not active_device and devices['devices']:
            active_device = devices['devices'][0]['id']
        
        # Start the mood mixer in a separate thread
        import threading
        import time
        
        def run_mood_mixer():
            try:
                # Import the mood mixer functions
                import sys
                import os
                
                # Add the Models directory to Python path to find mood_mixer.py
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Models')
                if models_dir not in sys.path:
                    sys.path.insert(0, models_dir)
                
                print(f"🔍 Looking for mood_mixer.py in: {models_dir}")
                print(f"🔍 Current working directory: {os.getcwd()}")
                print(f"🔍 Python path includes: {sys.path[:3]}...")
                
                # Check if mood_mixer.py exists
                mood_mixer_path = os.path.join(models_dir, 'mood_mixer.py')
                if os.path.exists(mood_mixer_path):
                    print(f"✅ Found mood_mixer.py at: {mood_mixer_path}")
                else:
                    print(f"❌ mood_mixer.py not found at: {mood_mixer_path}")
                    raise FileNotFoundError(f"mood_mixer.py not found at {mood_mixer_path}")
                
                # Import the mood mixer functions from mood_mixer.py
                from mood_mixer import (
                    build_batch, start_with_seed_then_queue, monitor_and_topup,
                    SEED_START, INITIAL_BATCH, TOP_UP_EVERY, TOP_UP_BATCH
                )
                print("✅ Successfully imported mood_mixer.py functions")
                
                # Build initial batch
                seen_uris = set()
                seed = build_batch(sp, language, mood_weights, SEED_START, seen_uris, first_page_only=True)
                for tr in seed: 
                    seen_uris.add(tr["uri"])
                
                # Build the remainder of the initial batch
                remaining_needed = max(0, INITIAL_BATCH - len(seed))
                rest = []
                if remaining_needed > 0:
                    rest = build_batch(sp, language, mood_weights, remaining_needed, seen_uris, first_page_only=False)
                    for tr in rest: 
                        seen_uris.add(tr["uri"])
                
                print(f"Starting mood mixer with {len(seed)} seed tracks, then queuing {len(rest)} more")
                start_with_seed_then_queue(sp, active_device, seed, rest)
                
                # Monitor and top-up
                monitor_and_topup(sp, active_device, language, mood_weights, seen_uris, next_topup_after=TOP_UP_EVERY)
                
            except Exception as e:
                print(f"Mood mixer error: {e}")
                import traceback
                traceback.print_exc()
        
        # Start the mood mixer thread
        mixer_thread = threading.Thread(target=run_mood_mixer, daemon=True)
        mixer_thread.start()
        
        return jsonify({
            'ok': True,
            'message': f'Started mood mixer with {language} language',
            'language': language,
            'mood_weights': mood_weights,
            'device_id': active_device
        })
        
    except Exception as e:
        print(f"Mood mixer error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/mood-mixer/moods', methods=['GET'])
def get_mood_categories():
    """Get available mood categories and their display names"""
    try:
        moods = [
            {"key": "happy", "name": "Happy ☀", "description": "Upbeat, cheerful music"},
            {"key": "sad_breakup", "name": "Sad / Breakup 💔", "description": "Emotional, melancholic tracks"},
            {"key": "motivational", "name": "Motivational ⚡", "description": "Energetic, inspiring music"},
            {"key": "chill_lofi", "name": "Chill / Lo-Fi 🌿", "description": "Relaxed, ambient tracks"},
            {"key": "hiphop", "name": "Hip Hop 🎧", "description": "Hip hop genre"},
            {"key": "rap", "name": "Rap 🔥", "description": "Rap music"},
            {"key": "old_classic", "name": "Old / Classic 📼", "description": "Vintage and classic songs"},
            {"key": "romantic", "name": "Romantic ❤", "description": "Love songs and romantic tracks"},
            {"key": "party", "name": "Party / Dance 🎉", "description": "High-energy party music"}
        ]
        
        return jsonify({
            'ok': True,
            'moods': moods,
            'total_moods': len(moods)
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Failed to get mood categories: {str(e)}'
        }), 500

@app.route('/api/mood-mixer/languages', methods=['GET'])
def get_language_options():
    """Get available language options"""
    try:
        languages = [
            {"key": "hi", "name": "Hindi 🇮🇳", "description": "Hindi music only"},
            {"key": "en", "name": "English 🇬🇧", "description": "English music only"},
            {"key": "mix", "name": "Mixed 🌍", "description": "Both Hindi and English music"}
        ]
        
        return jsonify({
            'ok': True,
            'languages': languages
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Failed to get language options: {str(e)}'
        }), 500

# ===== END MOOD MIXER INTEGRATION ENDPOINTS =====

# ===== END ARTIST MIX ENDPOINTS =====

@app.route('/api/spotify/dj/start', methods=['POST'])
def start_dj_session():
    """Start a DJ session with the specified parameters"""
    if not dj_run_once:
        return jsonify({"error": "DJ functionality not available"}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        mode = data.get('mode', 'random')
        genre = data.get('genre', 'Remix')
        artists = data.get('artists', [])
        batch_size = data.get('batch_size', getattr(Config, 'DJ_DEFAULT_BATCH_SIZE', 150))
        strict_primary = data.get('strict_primary', getattr(Config, 'DJ_STRICT_PRIMARY', True))
        
        # Validate inputs
        if mode not in ['random', 'artist']:
            return jsonify({"error": "Invalid mode. Must be 'random' or 'artist'"}), 400
            
        if genre not in ['Remix', 'LOFI', 'Mashup']:
            return jsonify({"error": "Invalid genre. Must be 'Remix', 'LOFI', or 'Mashup'"}), 400
            
        if mode == 'artist' and not artists:
            return jsonify({"error": "Artists list required for artist mode"}), 400
        
        # Call the DJ function
        result = dj_run_once(
            mode=mode,
            genre=genre,
            artists=artists,
            batch_size=batch_size,
            strict_primary=strict_primary
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"DJ session error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": str(datetime.datetime.now()),
        "gesture_models": gesture_model is not None,
        "dj_module": dj_run_once is not None,
        "artist_mix": True,
        "config": {
            "gesture_confidence_threshold": getattr(Config, 'GESTURE_CONFIDENCE_THRESHOLD', 0.8),
            "dj_batch_size": getattr(Config, 'DJ_DEFAULT_BATCH_SIZE', 150)
        }
    })

@app.route('/api/gesture/classes')
def get_gesture_classes():
    """Get available gesture classes"""
    if not gesture_model:
        return jsonify({"error": "Gesture model not loaded"}), 500
    
    classes = gesture_model.classes_.tolist() if hasattr(gesture_model, 'classes_') else []
    return jsonify({
        "classes": classes,
        "total_classes": len(classes),
        "confidence_threshold": getattr(Config, 'GESTURE_CONFIDENCE_THRESHOLD', 0.8)
    })

@app.route('/api/config')
def get_config():
    """Get current configuration (non-sensitive)"""
    return jsonify({
        "gesture_recognition": {
            "confidence_threshold": getattr(Config, 'GESTURE_CONFIDENCE_THRESHOLD', 0.8),
            "stable_frames": getattr(Config, 'GESTURE_STABLE_FRAMES', 5),
            "action_cooldown": getattr(Config, 'GESTURE_ACTION_COOLDOWN', 1.0)
        },
        "dj": {
            "default_batch_size": getattr(Config, 'DJ_DEFAULT_BATCH_SIZE', 150),
            "strict_primary": getattr(Config, 'DJ_STRICT_PRIMARY', True)
        },
        "server": {
            "host": getattr(Config, 'HOST', '0.0.0.0'),
            "port": getattr(Config, 'PORT', 5000),
            "debug": getattr(Config, 'DEBUG', True)
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ===== ORCHESTRATOR SERVICE (Background Thread) =====

# Global orchestrator state
orchestrator_running = False
orchestrator_threads = []
orchestrator_event_queue = queue.Queue()

def get_or_create_playlist(sp, name):
    """Get or create a playlist by name"""
    try:
        user = sp.current_user()["id"]
        offset = 0
        while True:
            playlists = sp.current_user_playlists(limit=50, offset=offset)["items"]
            for p in playlists:
                if p["name"] == name:
                    return p["id"]
            if len(playlists) < 50:
                break
            offset += 50
        return sp.user_playlist_create(user, name)["id"]
    except Exception as e:
        print(f"Error with playlist: {e}")
        return None

def orchestrator_listener():
    """Listen for detector connections on port 5050"""
    global orchestrator_running, orchestrator_event_queue
    
    HOST, PORT = "127.0.0.1", 5050
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[orchestrator] Listening on {HOST}:{PORT}")
        
        while orchestrator_running:
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
                print(f"[orchestrator] Connected to detector at {addr}")
                
                with conn:
                    buf = ""
                    while orchestrator_running:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk.decode("utf-8")
                        
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            try:
                                msg = json.loads(line)
                                
                                # Send gestures to event queue
                                gesture = msg.get("gesture", "None")
                                prob = msg.get("prob", 0.0)
                                
                                if gesture != "None":
                                    orchestrator_event_queue.put({
                                        "type": "gesture",
                                        "gesture": gesture,
                                        "prob": prob
                                    })
                                
                                # Send emotions to event queue
                                emo = msg.get("emotion", {})
                                if emo:
                                    label = max(emo, key=emo.get)
                                    orchestrator_event_queue.put({
                                        "type": "emotion",
                                        "label": label
                                    })
                                    
                            except Exception as e:
                                print(f"⚠️ Message parse error: {e}")
                                
            except socket.timeout:
                continue
            except Exception as e:
                if orchestrator_running:
                    print(f"[orchestrator] Connection error: {e}")

def weighted_with_neutral_cap(raw, WEIGHTS):
    """Apply weights, then cap neutral so it can't drown out short happy/sad bursts."""
    w_h = raw.get("happy", 0) * WEIGHTS["happy"]
    w_s = raw.get("sad", 0) * WEIGHTS["sad"]
    w_n = raw.get("neutral", 0) * WEIGHTS["neutral"]
    cap = max(w_h, w_s) * 0.6
    if cap > 0 and w_n > cap:
        w_n = cap
    return {"happy": w_h, "neutral": w_n, "sad": w_s}

def pct_triplet(weighted_dict):
    """Calculate percentage for each emotion"""
    total = sum(weighted_dict.values()) or 1e-9
    return {k: 100.0 * weighted_dict[k] / total for k in weighted_dict}

def dominant_label(weighted_dict):
    """Get dominant emotion label"""
    return max(weighted_dict, key=weighted_dict.get)

def print_phase_summary(title, raw_counts, WEIGHTS):
    """Print phase summary"""
    w = weighted_with_neutral_cap(raw_counts, WEIGHTS)
    p = pct_triplet(w)
    dom = dominant_label(w) if sum(raw_counts.values()) > 0 else "-"
    print(f"\n[{title}]  Dominant: {dom.upper()}  | "
          f"Happy: {p['happy']:.0f}%  Neutral: {p['neutral']:.0f}%  Sad: {p['sad']:.0f}%")

def should_skip_on_sad(raw_counts, WEIGHTS, min_events=2):
    """Check if should skip based on sad emotion share"""
    total_events = sum(raw_counts.values())
    if total_events < min_events:
        return False
    w = weighted_with_neutral_cap(raw_counts, WEIGHTS)
    denom = (w["happy"] + w["neutral"] + w["sad"]) or 1e-9
    sad_share = w["sad"] / denom
    print(f"  → SAD share={sad_share:.2%} (events={total_events})")
    return sad_share >= 0.60

def sum_counts(*phase_dicts):
    """Sum multiple phase dictionaries"""
    out = {"happy": 0, "neutral": 0, "sad": 0}
    for d in phase_dicts:
        for k in out:
            out[k] += d.get(k, 0)
    return out

def orchestrator_main_loop():
    """Main orchestrator loop that monitors playback and makes decisions"""
    global orchestrator_running, orchestrator_event_queue
    
    # Get Spotify OAuth manager (handles token refresh automatically)
    oauth = _spotify_oauth()
    token = oauth.get_cached_token()
    if not token:
        print("[orchestrator] Not authenticated with Spotify")
        return
    
    # Use auth_manager for automatic token refresh
    # This ensures tokens are refreshed automatically when they expire
    sp = spotipy.Spotify(auth_manager=oauth)
    
    # Get or create playlist
    playlist_id = get_or_create_playlist(sp, "found gems")
    
    # Orchestrator configuration
    INTRO_END = 0.20
    MIDDLE_END = 0.80
    WEIGHTS = {"happy": 1.0, "sad": 1.0, "neutral": 0.25}
    HAPPY_ADD_THRESHOLD = 0.30
    COOLDOWN_SEC = 6
    GESTURE_COOLDOWN = 2.0
    
    last_gesture_time = 0
    last_track_id = None
    last_uri = None
    phase_counts = {
        "intro": {"happy": 0, "neutral": 0, "sad": 0},
        "middle": {"happy": 0, "neutral": 0, "sad": 0},
        "outro": {"happy": 0, "neutral": 0, "sad": 0},
    }
    intro_closed = False
    middle_closed = False
    cooldown_until = 0
    
    def finalize_and_maybe_add():
        """At song end (or when new song starts), decide on add-to-playlist."""
        nonlocal last_uri, phase_counts, middle_closed
        
        if not last_uri:
            return
        
        # Only add if we actually reached phase 3 (outro) → means we crossed 80%
        if not middle_closed:
            print("ℹ️ Not adding: song didn't reach phase 3 (outro).")
            return
        
        overall_raw = sum_counts(
            phase_counts["intro"],
            phase_counts["middle"],
            phase_counts["outro"]
        )
        print_phase_summary("FINAL (All Phases)", overall_raw, WEIGHTS)
        
        # Overall happy share with neutral cap applied
        w_all = weighted_with_neutral_cap(overall_raw, WEIGHTS)
        denom = (w_all["happy"] + w_all["neutral"] + w_all["sad"]) or 1e-9
        happy_share = w_all["happy"] / denom
        
        if happy_share >= HAPPY_ADD_THRESHOLD:
            print(f"💚 Reached phase 3 & HAPPY ≥ {int(HAPPY_ADD_THRESHOLD*100)}% → Adding to playlist: found gems")
            try:
                sp.playlist_add_items(playlist_id, [last_uri])
            except Exception as e:
                print("⚠️ Add failed:", e)
        else:
            print(f"ℹ️ Not adding: HAPPY share {happy_share:.0%} below threshold ({HAPPY_ADD_THRESHOLD:.0%}).")
    
    print("[orchestrator] Main loop started")
    
    while orchestrator_running:
        try:
            # Check for playback (auth_manager handles token refresh automatically)
            pb = sp.current_playback()
            
            if not pb or not pb.get("item"):
                time.sleep(3)
                continue
            
            # Cooldown guard
            if time.time() < cooldown_until:
                time.sleep(0.5)
                continue
            
            # Process current track
            track = pb["item"]
            name = track["name"]
            artist = track["artists"][0]["name"]
            track_id = track["id"]
            uri = track["uri"]
            duration_ms = int(track.get("duration_ms") or 0)
            progress_ms = int(pb.get("progress_ms") or 0)
            
            # Calculate phase
            if not duration_ms:
                phase, pct, r = "unknown", 0, 0.0
            else:
                r = progress_ms / duration_ms
                pct = int(r * 100)
                if r < INTRO_END:
                    phase, pct, r = "intro", pct, r
                elif r < MIDDLE_END:
                    phase, pct, r = "middle", pct, r
                else:
                    phase, pct, r = "outro", pct, r
            
            # New song detection
            if track_id != last_track_id and progress_ms > 2000:
                # Finalize previous song
                finalize_and_maybe_add()
                
                # Reset phase counters
                phase_counts = {
                    "intro": {"happy": 0, "neutral": 0, "sad": 0},
                    "middle": {"happy": 0, "neutral": 0, "sad": 0},
                    "outro": {"happy": 0, "neutral": 0, "sad": 0},
                }
                intro_closed = False
                middle_closed = False
                last_track_id = track_id
                last_uri = uri
                print(f"\n\n🎵 Now playing: {name} — {artist}")
                cooldown_until = time.time() + 1.5
            
            # Process events from queue
            while True:
                try:
                    event = orchestrator_event_queue.get_nowait()
                    
                    if event["type"] == "gesture":
                        gesture = event["gesture"]
                        prob = event["prob"]
                        
                        # Simple gesture handling
                        current_time = time.time()
                        if (current_time - last_gesture_time) >= GESTURE_COOLDOWN and prob >= 0.6:
                            last_gesture_time = current_time
                            
                            try:
                                if gesture == "play_right":
                                    sp.start_playback()
                                elif gesture == "pause_right":
                                    sp.pause_playback()
                                elif gesture == "next_right":
                                    sp.next_track()
                                elif gesture == "previous_right":
                                    sp.previous_track()
                                elif gesture == "volume_up_left":
                                    pb = sp.current_playback()
                                    if pb and 'device' in pb:
                                        current_vol = pb['device']['volume_percent'] or 50
                                        sp.volume(min(100, current_vol + 10))
                                elif gesture == "volume_down_left":
                                    pb = sp.current_playback()
                                    if pb and 'device' in pb:
                                        current_vol = pb['device']['volume_percent'] or 50
                                        sp.volume(max(0, current_vol - 10))
                                elif gesture == "like_left":
                                    pb = sp.current_playback()
                                    if pb and pb.get('item'):
                                        track_id = pb['item']['id']
                                        sp.current_user_saved_tracks_add(tracks=[track_id])
                                elif gesture == "thumbs_up_right" and playlist_id:
                                    pb = sp.current_playback()
                                    if pb and pb.get('item'):
                                        track_uri = pb['item']['uri']
                                        sp.playlist_add_items(playlist_id, [track_uri])
                            except Exception as e:
                                print(f"⚠️ Gesture action failed: {e}")
                    
                    elif event["type"] == "emotion":
                        # Count emotion in current phase
                        if phase in phase_counts and event["label"] in phase_counts[phase]:
                            phase_counts[phase][event["label"]] += 1
                            
                except queue.Empty:
                    break
            
            # Phase boundary checks with skip decisions
            # Close INTRO at 20%
            if not intro_closed and r >= INTRO_END:
                intro_closed = True
                print_phase_summary("INTRO (0–20%)", phase_counts["intro"], WEIGHTS)
                # Skip only if SAD share >= 70% in intro, with min_events=2
                if should_skip_on_sad(phase_counts["intro"], WEIGHTS, min_events=2):
                    print("😞 Intro SAD ≥ 70% → SKIP")
                    try:
                        sp.next_track()
                        print("→ Skipped.")
                    except Exception as e:
                        print("⚠️ Skip failed:", e)
                    cooldown_until = time.time() + COOLDOWN_SEC
                    time.sleep(1.0)
                    continue
            
            # Close MIDDLE at 80% (means OUTRO reached)
            if not middle_closed and r >= MIDDLE_END:
                middle_closed = True
                print_phase_summary("MIDDLE (20–80%)", phase_counts["middle"], WEIGHTS)
                # Show cumulative Intro+Middle
                cum_im_raw = sum_counts(phase_counts["intro"], phase_counts["middle"])
                print_phase_summary("CUMULATIVE (0–80%)", cum_im_raw, WEIGHTS)
                
                # Skip only if SAD share >= 70% in middle, with min_events=3
                if should_skip_on_sad(phase_counts["middle"], WEIGHTS, min_events=3):
                    print("😔 Middle SAD ≥ 70% → SKIP")
                    try:
                        sp.next_track()
                        print("→ Skipped.")
                    except Exception as e:
                        print("⚠️ Skip failed:", e)
                    cooldown_until = time.time() + COOLDOWN_SEC
                    time.sleep(1.0)
                    continue
            
            time.sleep(1.0)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Handle token expiration errors
            if isinstance(e, spotipy.exceptions.SpotifyException) and e.http_status == 401:
                print(f"⚠️ Orchestrator error: Token expired (401). Attempting refresh...")
                try:
                    token = oauth.get_cached_token()
                    if token:
                        sp = spotipy.Spotify(auth_manager=oauth)
                        print("[orchestrator] Token refreshed, continuing...")
                        time.sleep(1)
                        continue
                    else:
                        print("[orchestrator] Failed to refresh token, stopping orchestrator")
                        break
                except Exception as refresh_error:
                    print(f"[orchestrator] Token refresh failed: {refresh_error}")
                    break
            else:
                print(f"⚠️ Orchestrator error: {e}")
                time.sleep(5)
    
    print("[orchestrator] Main loop ended")

# start_orchestrator and stop_orchestrator moved to API endpoints

# ===== ORCHESTRATOR API ENDPOINTS =====

@app.route('/api/orchestrator/start', methods=['POST'])
def api_start_orchestrator():
    """Start the orchestrator service"""
    global orchestrator_running, orchestrator_threads
    
    try:
        if orchestrator_running:
            return jsonify({"ok": False, "error": "Already running"}), 400
        
        # Reset state
        orchestrator_running = True
        orchestrator_threads = []
        
        # Start listener thread
        listener_thread = threading.Thread(target=orchestrator_listener, daemon=True)
        listener_thread.start()
        orchestrator_threads.append(listener_thread)
        
        # Start main loop thread
        main_thread = threading.Thread(target=orchestrator_main_loop, daemon=True)
        main_thread.start()
        orchestrator_threads.append(main_thread)
        
        print("[orchestrator] Service started via API")
        return jsonify({"ok": True, "message": "Orchestrator started"})
    except Exception as e:
        orchestrator_running = False
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/orchestrator/stop', methods=['POST'])
def api_stop_orchestrator():
    """Stop the orchestrator service"""
    global orchestrator_running, orchestrator_threads
    
    try:
        orchestrator_running = False
        
        # Wait for threads to stop
        for thread in orchestrator_threads:
            thread.join(timeout=2.0)
        
        orchestrator_threads = []
        print("[orchestrator] Service stopped via API")
        return jsonify({"ok": True, "message": "Orchestrator stopped"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/orchestrator/status', methods=['GET'])
def api_orchestrator_status():
    """Get orchestrator status"""
    return jsonify({
        "ok": True,
        "running": orchestrator_running
    })

# ===== END ORCHESTRATOR API ENDPOINTS =====

# ===== Spotify OAuth (server-managed) =====

@app.get('/api/spotify/status')
def spotify_status():
    try:
        oauth = _spotify_oauth()
        token_info = oauth.get_cached_token()
        is_authed = bool(token_info)
        status = {"authenticated": is_authed}
        if is_authed:
            sp = spotipy.Spotify(auth=token_info['access_token'])
            try:
                me = sp.current_user()
                status["user"] = {"id": me.get('id'), "name": me.get('display_name') or me.get('id')}
                devices = sp.devices().get('devices', [])
                status["devices"] = [{"id": d.get('id'), "name": d.get('name'), "is_active": d.get('is_active')} for d in devices]
            except Exception:
                pass
        return jsonify(status)
    except Exception as e:
        return jsonify({"authenticated": False, "error": str(e)}), 500

@app.get('/api/spotify/login')
def spotify_login():
    try:
        oauth = _spotify_oauth()
        auth_url = oauth.get_authorize_url()
        return jsonify({"auth_url": auth_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get('/callback')
def spotify_callback():
    try:
        oauth = _spotify_oauth()
        if not oauth:
            return jsonify({"error": "Spotify OAuth not configured"}), 500
            
        code = request.args.get('code')
        state = request.args.get('state')
        frontend_target = getattr(
            Config,
            'FRONTEND_REDIRECT_URI',
            'http://127.0.0.1:5500/profile.html'  # Updated: assumes server root is at frontend/
        )

        if not code:
            # If this is an internal redirect (e.g., auth=success) just bounce to frontend.
            query = request.query_string.decode('utf-8')
            suffix = f"?{query}" if query else ""
            return redirect(f"{frontend_target}{suffix}")
            
        token_info = oauth.get_access_token(code)
        try:
            sync_spotify_token_cache()
        except Exception as sync_err:
            print(f"[WARN] Spotify cache sync skipped: {sync_err}")
        # Persisted via cache_path; redirect back to frontend with success
        return redirect(f"{frontend_target}?auth=success&expires_in={token_info.get('expires_in', 0)}")
    except Exception as e:
        print(f"Spotify callback error: {e}")
        frontend_target = getattr(
            Config,
            'FRONTEND_REDIRECT_URI',
            'http://127.0.0.1:5500/profile.html'  # Updated: assumes server root is at frontend/
        )
        return redirect(f"{frontend_target}?auth=error&message={str(e)}")

@app.post('/api/spotify/play')
def spotify_play_specific():
    """Play a specific track given a Spotify URI or search query."""
    try:
        data = request.get_json(force=True)
        uri = (data or {}).get('uri')
        query = (data or {}).get('query')
        oauth = _spotify_oauth()
        token = oauth.get_cached_token()
        if not token:
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        sp = spotipy.Spotify(auth=token['access_token'])

        # resolve device
        devices = sp.devices().get('devices', [])
        if not devices:
            return jsonify({"ok": False, "error": "No active Spotify device"}), 400
        device_id = None
        for d in devices:
            if d.get('is_active'):
                device_id = d.get('id'); break
        if not device_id:
            device_id = devices[0].get('id')
            try:
                sp.transfer_playback(device_id=device_id, force_play=True)
            except Exception:
                pass

        target_uri = uri
        if not target_uri and query:
            res = sp.search(q=query, type='track', limit=1)
            items = ((res or {}).get('tracks') or {}).get('items') or []
            if not items:
                return jsonify({"ok": False, "error": "Track not found"}), 404
            target_uri = items[0].get('uri')

        if not target_uri:
            return jsonify({"ok": False, "error": "Provide 'uri' or 'query'"}), 400

        sp.start_playback(device_id=device_id, uris=[target_uri])
        return jsonify({"ok": True, "played": target_uri})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post('/api/spotify/control')
def spotify_control():
    """Generic control endpoint for playback actions from gestures/UI.
    Body: { "action": "play|pause|next|previous|volume|seek", "delta": 10| -10 | 30000 }
    """
    try:
        data = request.get_json(force=True) or {}
        action = (data.get('action') or '').lower()
        delta = int(data.get('delta') or 0)

        oauth = _spotify_oauth()
        token = oauth.get_cached_token()
        if not token:
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        sp = spotipy.Spotify(auth=token['access_token'])

        devices = sp.devices().get('devices', [])
        if not devices:
            return jsonify({"ok": False, "error": "No active Spotify device"}), 400
        device = next((d for d in devices if d.get('is_active')), devices[0])
        device_id = device.get('id')

        if action == 'play':
            sp.start_playback(device_id=device_id)
        elif action == 'pause':
            sp.pause_playback(device_id=device_id)
        elif action == 'next':
            sp.next_track(device_id=device_id)
        elif action == 'previous':
            sp.previous_track(device_id=device_id)
        elif action == 'volume':
            cur_v = device.get('volume_percent', 50)
            new_v = max(0, min(100, cur_v + (delta if delta else 0)))
            sp.volume(new_v, device_id=device_id)
        elif action == 'seek':
            pb = sp.current_playback()
            if not pb or not pb.get('item'):
                return jsonify({"ok": False, "error": "No current playback"}), 400
            pos = pb.get('progress_ms', 0)
            dur = pb['item'].get('duration_ms', 0)
            new_pos = min(max(0, pos + (delta if delta else 0)), max(0, dur - 1000))
            sp.seek_track(new_pos, device_id=device_id)
        elif action == 'like':
            # Like/save current track
            pb = sp.current_playback()
            if not pb or not pb.get('item'):
                return jsonify({"ok": False, "error": "No current playback"}), 400
            track_id = pb['item'].get('id')
            if track_id:
                sp.current_user_saved_tracks_add([track_id])
            else:
                return jsonify({"ok": False, "error": "No track ID"}), 400
        else:
            return jsonify({"ok": False, "error": "Unknown action"}), 400

        return jsonify({"ok": True, "action": action})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get('/api/spotify/current')
def spotify_current():
    """Get currently playing track information with metadata and progress"""
    try:
        oauth = _spotify_oauth()
        token = oauth.get_cached_token()
        if not token:
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        
        sp = spotipy.Spotify(auth=token['access_token'])
        playback = sp.current_playback()
        
        if not playback:
            return jsonify({"ok": True, "playing": False, "message": "No active playback"})
        
        track = playback.get('item', {})
        if not track:
            return jsonify({"ok": True, "playing": False, "message": "No track information"})
        
        # Extract track information
        track_info = {
            "id": track.get('id'),
            "name": track.get('name'),
            "artists": [artist.get('name') for artist in track.get('artists', [])],
            "album": track.get('album', {}).get('name'),
            "duration_ms": track.get('duration_ms'),
            "external_urls": track.get('external_urls', {}),
            "preview_url": track.get('preview_url')
        }
        
        # Extract album art
        images = track.get('album', {}).get('images', [])
        if images:
            track_info['album_art'] = images[0].get('url')  # Get largest image
        
        # Extract playback state
        playback_info = {
            "is_playing": playback.get('is_playing', False),
            "progress_ms": playback.get('progress_ms', 0),
            "volume_percent": playback.get('device', {}).get('volume_percent', 0),
            "shuffle_state": playback.get('shuffle_state', False),
            "repeat_state": playback.get('repeat_state', 'off'),
            "device": {
                "id": playback.get('device', {}).get('id'),
                "name": playback.get('device', {}).get('name'),
                "type": playback.get('device', {}).get('type'),
                "is_active": playback.get('device', {}).get('is_active', False)
            }
        }
        
        return jsonify({
            "ok": True,
            "playing": True,
            "track": track_info,
            "playback": playback_info
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get('/api/spotify/devices')
def spotify_devices():
    """Get available Spotify devices"""
    try:
        oauth = _spotify_oauth()
        token = oauth.get_cached_token()
        if not token:
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        
        sp = spotipy.Spotify(auth=token['access_token'])
        devices = sp.devices().get('devices', [])
        
        return jsonify({
            "ok": True,
            "devices": devices
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post('/api/spotify/transfer')
def spotify_transfer():
    """Transfer playback to a specific device"""
    try:
        data = request.get_json(force=True) or {}
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({"ok": False, "error": "Device ID required"}), 400
        
        oauth = _spotify_oauth()
        token = oauth.get_cached_token()
        if not token:
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        
        sp = spotipy.Spotify(auth=token['access_token'])
        sp.transfer_playback(device_id=device_id, force_play=True)
        
        return jsonify({"ok": True, "device_id": device_id})
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Print startup information
    print("🎵 Smart Music Backend Starting...")
    print(f"📍 Server will run on {getattr(Config, 'HOST', '0.0.0.0')}:{getattr(Config, 'PORT', 5000)}")
    print(f"🔧 Gesture models: {'✅ Loaded' if gesture_model else '❌ Not loaded'}")
    print(f"🎧 DJ functionality: {'✅ Available' if dj_run_once else '❌ Not available'}")
    print(f"🎤 Artist Mix: ✅ Available")
    print(f"🎵 Mood Mixer: ✅ Available")
    print("-" * 50)
    
    app.run(
        debug=getattr(Config, 'DEBUG', True),
        host=getattr(Config, 'HOST', '0.0.0.0'),
        port=getattr(Config, 'PORT', 3000)
    )