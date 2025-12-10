#!/usr/bin/env python3
# zorchestrator_live.py — Phase rules: SAD skip @ 75% (per phase) + HAPPY add @ threshold (overall)
# Neutral capped to 60% of stronger expressive emotion; full gesture support

import os, sys, json, time, socket, threading, queue
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ---------------------------------------------
# 👇 HARDCODED SPOTIFY CREDENTIALS START HERE 👇
# ---------------------------------------------
SPOTIPY_CLIENT_ID     = "1dac75a3e61745f7bc7cbc82ff882af3"     # Replace with your Client ID
SPOTIPY_CLIENT_SECRET = "1ebe2efb8edb4d4aabaaecf2de4cbeaa"     # Replace with your Client Secret
SPOTIPY_REDIRECT_URI  = "http://localhost:8888/callback"
# ---------------------------------------------
# 👆 HARDCODED SPOTIFY CREDENTIALS END HERE 👆
# ---------------------------------------------

# ------------- Spotify Auth -------------
SCOPE = "user-read-playback-state user-modify-playback-state playlist-modify-public user-library-modify"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=SCOPE
))

HOST, PORT = "127.0.0.1", 5050
PLAYLIST_NAME = "found gems"

# Phase cutoffs (fractions)
INTRO_END  = 0.20
MIDDLE_END = 0.80

# No bias between happy and sad; neutral low so it doesn't dominate.
WEIGHTS = {"happy": 1.0, "sad": 1.0, "neutral": 0.25}

# Happy threshold (overall, weighted+neutral-capped share). 
HAPPY_ADD_THRESHOLD = 0.30

COOLDOWN_SEC = 6               # prevent rapid re-decisions
GESTURE_COOLDOWN = 2.0         # Prevent gesture spam
event_q = queue.Queue()

# Gesture state tracking
last_gesture_time = 0
current_volume = 50  # Default volume

# ------------- Helpers -------------
def get_or_create_playlist(sp, name):
    user = sp.current_user()["id"]
    offs = 0
    while True:
        pls = sp.current_user_playlists(limit=50, offset=offs)["items"]
        for p in pls:
            if p["name"] == name:
                return p["id"]
        if len(pls) < 50:
            break
        offs += 50
    return sp.user_playlist_create(user, name)["id"]

def song_phase(progress_ms, duration_ms):
    if not duration_ms:
        return "unknown", 0, 0.0
    r = progress_ms / duration_ms
    pct = int(r * 100)
    if r < INTRO_END:  return "intro",  pct, r
    if r < MIDDLE_END: return "middle", pct, r
    return "outro", pct, r

def reset_phase_counts():
    return {
        "intro":  {"happy": 0, "neutral": 0, "sad": 0},
        "middle": {"happy": 0, "neutral": 0, "sad": 0},
        "outro":  {"happy": 0, "neutral": 0, "sad": 0},
    }

def add_event(counts, phase, label):
    if phase in counts and label in counts[phase]:
        counts[phase][label] += 1

def sum_counts(*phase_dicts):
    out = {"happy":0, "neutral":0, "sad":0}
    for d in phase_dicts:
        for k in out:
            out[k] += d.get(k,0)
    return out

def weighted_with_neutral_cap(raw):
    """
    Apply weights, then cap neutral so it can't drown out short happy/sad bursts.
    Neutral cap = 0.6 * max(weighted_happy, weighted_sad)
    """
    w_h = raw.get("happy", 0)   * WEIGHTS["happy"]
    w_s = raw.get("sad", 0)     * WEIGHTS["sad"]
    w_n = raw.get("neutral", 0) * WEIGHTS["neutral"]
    cap = max(w_h, w_s) * 0.6
    if cap > 0 and w_n > cap:
        w_n = cap
    return {"happy": w_h, "neutral": w_n, "sad": w_s}

def pct_triplet(weighted_dict):
    total = sum(weighted_dict.values()) or 1e-9
    return {k: 100.0 * weighted_dict[k] / total for k in weighted_dict}

def dominant_label(weighted_dict):
    return max(weighted_dict, key=weighted_dict.get)

def print_phase_summary(title, raw_counts):
    w = weighted_with_neutral_cap(raw_counts)
    p = pct_triplet(w)
    dom = dominant_label(w) if sum(raw_counts.values())>0 else "-"
    print(f"\n[{title}]  Dominant: {dom.upper()}  | "
          f"Happy: {p['happy']:.0f}%  Neutral: {p['neutral']:.0f}%  Sad: {p['sad']:.0f}%")

# ---- Share-based SAD skip rule (≥ 70% in that phase) ----
def should_skip_on_sad(raw_counts, min_events=2):
    total_events = sum(raw_counts.values())
    if total_events < min_events:
        return False
    w = weighted_with_neutral_cap(raw_counts)
    denom = (w["happy"] + w["neutral"] + w["sad"]) or 1e-9
    sad_share = w["sad"] / denom
    # Debug print to verify behavior; comment out if noisy
    print(f"  → SAD share={sad_share:.2%} (events={total_events})")
    return sad_share >= 0.70

# ------------- Gesture Handlers (UNCHANGED) -------------
def handle_gesture(gesture, prob):
    """Handle all gestures from the detector with proper Spotify controls"""
    global last_gesture_time, current_volume
    
    current_time = time.time()
    if (current_time - last_gesture_time) < GESTURE_COOLDOWN:
        return
    
    # Only process high-confidence gestures
    if prob < 0.6: 
        return
        
    last_gesture_time = current_time
    
    try:
        if gesture == "play_right":
            print("▶️  Gesture: PLAY")
            sp.start_playback()
            print("→ Playback started")
            
        elif gesture == "pause_right":
            print("⏸️  Gesture: PAUSE")
            sp.pause_playback()
            print("→ Playback paused")
                
        elif gesture == "next_right":
            print("⏭️  Gesture: NEXT TRACK")
            sp.next_track()
            print("→ Skipped to next track")
            
        elif gesture == "previous_right":
            print("⏮️  Gesture: PREVIOUS TRACK")
            sp.previous_track()
            print("→ Went back to previous track")

        elif gesture == "volume_down_left":
            print("🔉 Gesture: VOLUME DOWN (-10%)")
            pb = sp.current_playback()
            if pb and 'device' in pb:
                current_vol = pb['device']['volume_percent'] or current_volume
                new_vol = max(0, current_vol - 10) 
                sp.volume(new_vol)
                current_volume = new_vol
                print(f"→ Volume: {current_vol}% → {new_vol}%")

        elif gesture == "volume_up_left":
            print("🔊 Gesture: VOLUME UP (+10%)")
            pb = sp.current_playback()
            if pb and 'device' in pb:
                current_vol = pb['device']['volume_percent'] or current_volume
                new_vol = min(100, current_vol + 10)
                sp.volume(new_vol)
                current_volume = new_vol
                print(f"→ Volume: {current_vol}% → {new_vol}%")
                
        elif gesture == "like_left":
            print("❤️ Gesture: LIKE - Add to Saved Tracks")
            pb = sp.current_playback()
            if pb and pb.get('item'):
                track_id = pb['item']['id']
                sp.current_user_saved_tracks_add(tracks=[track_id])
                print("→ Added current track to your Saved Tracks")
                
        elif gesture == "skip30_left":
            print("⏩ Gesture: SKIP FORWARD (+30s)")
            pb = sp.current_playback()
            if pb and pb.get('progress_ms') is not None and pb.get('item'):
                new_progress = pb['progress_ms'] + 30000 
                duration_ms = pb['item'].get('duration_ms')
                if duration_ms and new_progress < duration_ms:
                    sp.seek_track(new_progress)
                    print("→ Skipped forward 30 seconds")
                else:
                    print("→ Cannot skip, near end of track or no duration data.")

        # Optional legacy gestures:
        elif gesture == "thumbs_up_right":
            print("👍 Gesture: THUMBS UP - Add to 'found gems' playlist")
            pb = sp.current_playback()
            if pb and pb.get('item'):
                track_uri = pb['item']['uri']
                sp.playlist_add_items(playlist_id, [track_uri])
                print("→ Added current track to playlist")
                
        elif gesture == "thumbs_down_left":
            print("👎 Gesture: THUMBS DOWN - Skip track")
            sp.next_track()
            print("→ Skipped track (thumbs down)")
            
        elif gesture == "stop_right":
            print("🛑 Gesture: STOP/SEEK 0")
            sp.pause_playback()
            sp.seek_track(0)
            print("→ Stopped and reset to beginning")
            
        else:
            if gesture == "pause_right_old":
                 print("⏸️  Gesture: PAUSE/RESUME (toggle)")
                 pb = sp.current_playback()
                 if pb and pb.get('is_playing'):
                     sp.pause_playback()
                     print("→ Playback paused")
                 else:
                     sp.start_playback()
                     print("→ Playback resumed")
            elif gesture != "None":
                 print(f"❓ Unmapped gesture received: {gesture}")

    except Exception as e:
        print(f"⚠️ Gesture action failed for {gesture}: {e}")

# ------------- Socket listener -------------
def listen_detector():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[orchestrator] Listening on {HOST}:{PORT}")
        conn, _ = s.accept()
        print("[orchestrator] Connected to detector.")
        with conn:
            buf = ""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk.decode("utf-8")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    try:
                        msg = json.loads(line)
                        
                        # Handle gestures for playback control
                        gesture = msg.get("gesture", "None")
                        prob = msg.get("prob", 0.0)
                        
                        # Process gestures for immediate playback control
                        if gesture != "None":
                            handle_gesture(gesture, prob)
                        
                        # Handle emotions for playlist decisions
                        emo = msg.get("emotion", {})
                        if emo:
                            label = max(emo, key=emo.get)  # 'happy'/'neutral'/'sad'
                            event_q.put({"label": label, "ts": time.time()})
                            
                    except Exception as e:
                        print(f"⚠️ Message parse error: {e}")

threading.Thread(target=listen_detector, daemon=True).start()
playlist_id = get_or_create_playlist(sp, PLAYLIST_NAME)

# ------------- Main loop -------------
print("[orchestrator] Waiting for Spotify playback…")
print("[orchestrator] Gesture controls ready (based on your model):")
print("  ▶️  play_right - Start Playback")
print("  ⏸️  pause_right - Pause Playback")
print("  ⏭️  next_right - Next Track")
print("  ⏮️  previous_right - Previous Track")
print("  🔊 volume_up_left - Volume Up (+10%)")
print("  🔉 volume_down_left - Volume Down (-10%)") 
print("  ❤️ like_left - Add to Saved Tracks")
print("  ⏩ skip30_left - Skip Forward 30s")
print("  👍 thumbs_up_right - Add to 'found gems'")
print("  👎 thumbs_down_left - Skip Track")

last_track_id = None
last_uri = None
phase_counts = reset_phase_counts()
intro_closed = False
middle_closed = False
cooldown_until = 0

def finalize_and_maybe_add():
    """At song end (or when new song starts), decide on add-to-playlist."""
    global last_uri, phase_counts, middle_closed
    if not last_uri:
        return

    # Only add if we actually reached phase 3 (outro) → means we crossed 80%
    if not middle_closed:
        print("ℹ️ Not adding: song didn’t reach phase 3 (outro).")
        return

    overall_raw = sum_counts(
        phase_counts["intro"],
        phase_counts["middle"],
        phase_counts["outro"]
    )
    print_phase_summary("FINAL (All Phases)", overall_raw)

    # Overall happy share with neutral cap applied
    w_all = weighted_with_neutral_cap(overall_raw)
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

while True:
    try:
        pb = sp.current_playback()
        if not pb or not pb.get("item"):
            sys.stdout.write("\r⏸️  Nothing playing. Waiting…   ")
            sys.stdout.flush()
            time.sleep(3)
            continue

        # cooldown guard
        if time.time() < cooldown_until:
            time.sleep(0.5)
            continue

        track = pb["item"]
        name   = track["name"]
        artist = track["artists"][0]["name"]
        track_id = track["id"]
        uri = track["uri"]
        duration_ms = int(track.get("duration_ms") or 0)
        progress_ms = int(pb.get("progress_ms") or 0)
        phase, pct, r = song_phase(progress_ms, duration_ms)

        # New song (only when progress > 2s to avoid flicker)
        if track_id != last_track_id and progress_ms > 2000:
            # finalize previous
            finalize_and_maybe_add()
            # reset counters for new song
            phase_counts = reset_phase_counts()
            intro_closed = False
            middle_closed = False
            last_track_id = track_id
            last_uri = uri
            print(f"\n\n🎵 Now playing: {name} — {artist}")
            cooldown_until = time.time() + 1.5  # small grace

        # Drain stable-emotion events and count them into the CURRENT phase
        drained = 0
        while True:
            try:
                ev = event_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            add_event(phase_counts, phase, ev["label"])

        # Simple progress line (no dict spam)
        bar = "█" * (pct // 5) + "-" * (20 - pct // 5)
        sys.stdout.write(f"\r  [{bar}] {pct:3d}% {phase:6} | new events: {drained:<2} ")
        sys.stdout.flush()

        # ---- Phase boundary checks with clean summaries ----
        # Close INTRO at 20%
        if (not intro_closed) and r >= INTRO_END:
            intro_closed = True
            print_phase_summary("INTRO (0–20%)", phase_counts["intro"])
            # Skip only if SAD share >= 75% in intro, with min_events=2
            if should_skip_on_sad(phase_counts["intro"], min_events=2):
                print("😞 Intro SAD ≥ 75% → SKIP")
                try:
                    sp.next_track()
                    print("→ Skipped.")
                except Exception as e:
                    print("⚠️ Skip failed:", e)
                cooldown_until = time.time() + COOLDOWN_SEC
                time.sleep(1.0)
                continue

        # Close MIDDLE at 80% (means OUTRO reached)
        if (not middle_closed) and r >= MIDDLE_END:
            middle_closed = True
            print_phase_summary("MIDDLE (20–80%)", phase_counts["middle"])
            # Show cumulative Intro+Middle
            cum_im_raw = sum_counts(phase_counts["intro"], phase_counts["middle"])
            print_phase_summary("CUMULATIVE (0–80%)", cum_im_raw)

            # Skip only if SAD share >= 75% in middle, with min_events=3
            if should_skip_on_sad(phase_counts["middle"], min_events=3):
                print("😔 Middle SAD ≥ 75% → SKIP")
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
        print("\nExiting orchestrator...")
        finalize_and_maybe_add()
        break
    except Exception as e:
        print("\n⚠️ Spotify error:", e)
        time.sleep(5)
