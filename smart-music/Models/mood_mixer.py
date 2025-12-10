import os, sys, time, math, random
from typing import Dict, List, Tuple, Optional
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

# ========================= USER SETTINGS =========================

INITIAL_BATCH   = 50   # total initial tracks to queue
SEED_START      = 5    # start playback as soon as this many are ready
TOP_UP_EVERY    = 20   # after this many plays/skips, add TOP_UP_BATCH
TOP_UP_BATCH    = 20
QUEUE_LOW_WATER = 5    # proactive top-up when queue length <= this
POLL_SECONDS    = 3    # how often to poll playback for progress

# Popularity rules
POP_SKIP_BELOW      = 20   # skip tracks with popularity < 20
HIGH_POP_MIN        = 75   # high popularity bin
MID_POP_MIN         = 40   # mid popularity bin
TARGET_HIGH_RATIO   = 0.60 # target ratio from high bin per mood

# Spotify scopes required
SCOPE = "user-read-playback-state user-modify-playback-state user-library-modify user-read-recently-played"

# ---- YOUR SPOTIFY CREDENTIALS ----
SPOTIFY_CLIENT_ID     = "795fe99315d844abaa418cbb7e7bd243"
SPOTIFY_CLIENT_SECRET = "dfd184c296764fc1bb05ae83a25c5415"
SPOTIFY_REDIRECT_URI  = "http://localhost:8888/callback"

# ========================= PLAYLIST MAPPING =========================
# Focus/Study REMOVED. 9 moods remain.
MOOD_BUCKETS: Dict[str, Dict[str, List[str]]] = {
    "happy": {
        "hi":  ["https://open.spotify.com/playlist/79xLMj1KeVMoUi41KqMUQe?si=11080a47019b4382"],
        "en":  ["https://open.spotify.com/playlist/3EB6XyFTpSv8VxruReIdMg?si=7ba6a5dcd92043b8",
                "https://open.spotify.com/playlist/1UQrjtuEtQbDEwTabF6yIO?si=5b6faa97915a4977"],
        "mix": ["https://open.spotify.com/playlist/79xLMj1KeVMoUi41KqMUQe?si=11080a47019b4382",
                "https://open.spotify.com/playlist/3EB6XyFTpSv8VxruReIdMg?si=7ba6a5dcd92043b8",
                "https://open.spotify.com/playlist/1UQrjtuEtQbDEwTabF6yIO?si=5b6faa97915a4977"]
    },
    "sad_breakup": {
        "hi":  ["https://open.spotify.com/playlist/1rTY4QOYmic4b3BHVRiojQ?si=251e79249bce43c2",
                "https://open.spotify.com/playlist/3Ws61jJ4noliGA8Q5h37t8?si=b6ccfe61adba4389"],
        "en":  ["https://open.spotify.com/playlist/6JaPLSSkBptVCFXRXrY0fQ?si=4d9541fde9c54da4"],
        "mix": ["https://open.spotify.com/playlist/1rTY4QOYmic4b3BHVRiojQ?si=251e79249bce43c2",
                "https://open.spotify.com/playlist/3Ws61jJ4noliGA8Q5h37t8?si=b6ccfe61adba4389",
                "https://open.spotify.com/playlist/6JaPLSSkBptVCFXRXrY0fQ?si=4d9541fde9c54da4",
                "https://open.spotify.com/playlist/4Z5dm557CxWggGawm9uLF9?si=aecf96023c3740e2"]
    },
    "motivational": {
        "hi":  ["https://open.spotify.com/playlist/5Qbce0bWB8Y59eZauJJlX8?si=d94fbbb63272436f",
                "https://open.spotify.com/playlist/5JLOzpu3ailj2avsE7Z7Qd?si=59091b84e5f6441d"],
        "en":  ["https://open.spotify.com/playlist/17Av5xOLGhDEJVAEKYNJgl?si=1bf6af83aa404b07",
                "https://open.spotify.com/playlist/78R9LCH5fpKA5c1UxbuozZ?si=997a89ca1e274968"],
        "mix": ["https://open.spotify.com/playlist/5Qbce0bWB8Y59eZauJJlX8?si=d94fbbb63272436f",
                "https://open.spotify.com/playlist/5JLOzpu3ailj2avsE7Z7Qd?si=59091b84e5f6441d",
                "https://open.spotify.com/playlist/17Av5xOLGhDEJVAEKYNJgl?si=1bf6af83aa404b07",
                "https://open.spotify.com/playlist/78R9LCH5fpKA5c1UxbuozZ?si=997a89ca1e274968",
                "https://open.spotify.com/playlist/1kwxVltIevJMrwItTVI36s?si=2b0c822a8f1f4599"]
    },
    "chill_lofi": {
        "hi":  ["https://open.spotify.com/playlist/3TTTD8hn7hdBAp8MF4gRqV?si=df10bdc42d744653",
                "https://open.spotify.com/playlist/36OeXlo8GxxR0w9k7HsEyL?si=c20ccec025aa442f",
                "https://open.spotify.com/playlist/5Hz91rsZYzxub5mj8IO123?si=692e3ba344ca48ca",
                "https://open.spotify.com/playlist/79XqRi1252xrIS89CLoThd?si=09c71d19d69c4521"],
        "en":  ["https://open.spotify.com/playlist/6lFErS9yaCeEJG35wml9vc?si=a6aea8f626104df9",
                "https://open.spotify.com/playlist/4Z5dm557CxWggGawm9uLF9?si=aecf96023c3740e2"],
        "mix": ["https://open.spotify.com/playlist/3TTTD8hn7hdBAp8MF4gRqV?si=df10bdc42d744653",
                "https://open.spotify.com/playlist/6lFErS9yaCeEJG35wml9vc?si=a6aea8f626104df9",
                "https://open.spotify.com/playlist/5Hz91rsZYzxub5mj8IO123?si=692e3ba344ca48ca"]
    },
    "hiphop": {
        "hi":  ["https://open.spotify.com/playlist/2ilgAL3xMoG3C07EZ2VEAy?si=7761bf20700f4522"],
        "en":  ["https://open.spotify.com/playlist/38fSPhhBYyWM77NmpyBOxF?si=74e3b76cb72b429b"],
        "mix": ["https://open.spotify.com/playlist/2ilgAL3xMoG3C07EZ2VEAy?si=7761bf20700f4522",
                "https://open.spotify.com/playlist/38fSPhhBYyWM77NmpyBOxF?si=74e3b76cb72b429b"]
    },
    "rap": {
        "hi":  ["https://open.spotify.com/playlist/4Vhjc2sUetHcH57fIUdHib?si=71840f08f79a44c1"],
        "en":  ["https://open.spotify.com/playlist/561x7wLHO2c44jiJEOy1nA?si=a1829a5d8b3b4183"],
        "mix": ["https://open.spotify.com/playlist/4Vhjc2sUetHcH57fIUdHib?si=71840f08f79a44c1",
                "https://open.spotify.com/playlist/561x7wLHO2c44jiJEOy1nA?si=a1829a5d8b3b4183",
                "https://open.spotify.com/playlist/1kwxVltIevJMrwItTVI36s?si=2b0c822a8f1f4599"]
    },
    "old_classic": {
        "hi":  ["https://open.spotify.com/playlist/121325QlkyTvvLPfTsAt4D?si=8751d0843b3d4b30",
                "https://open.spotify.com/playlist/3Ws61jJ4noliGA8Q5h37t8?si=b6ccfe61adba4389",
                "https://open.spotify.com/playlist/79XqRi1252xrIS89CLoThd?si=09c71d19d69c4521"],
        "en":  ["https://open.spotify.com/playlist/2adNX7k9f1BY4Azm6ZCfwo?si=fee302c7965d4596"],
        "mix": ["https://open.spotify.com/playlist/121325QlkyTvvLPfTsAt4D?si=8751d0843b3d4b30",
                "https://open.spotify.com/playlist/3Ws61jJ4noliGA8Q5h37t8?si=b6ccfe61adba4389",
                "https://open.spotify.com/playlist/2adNX7k9f1BY4Azm6ZCfwo?si=fee302c7965d4596",
                "https://open.spotify.com/playlist/79XqRi1252xrIS89CLoThd?si=09c71d19d69c4521"]
    },
    "romantic": {
        "hi":  ["https://open.spotify.com/playlist/7DmoMtX3dvySvSigzbgNcy?si=4b295f583eb8426f",
                "https://open.spotify.com/playlist/121325QlkyTvvLPfTsAt4D?si=8751d0843b3d4b30"],
        "en":  ["https://open.spotify.com/playlist/0ONA0nz5UyefGKJKobGpz1?si=9c9297f6470941d6"],
        "mix": ["https://open.spotify.com/playlist/7DmoMtX3dvySvSigzbgNcy?si=4b295f583eb8426f",
                "https://open.spotify.com/playlist/0ONA0nz5UyefGKJKobGpz1?si=9c9297f6470941d6",
                "https://open.spotify.com/playlist/121325QlkyTvvLPfTsAt4D?si=8751d0843b3d4b30"]
    },
    "party": {
        "hi":  ["https://open.spotify.com/playlist/6IXKFG4scJV5mi8wcqxcVt?si=dfc359a9df784520"],
        "en":  ["https://open.spotify.com/playlist/1UQrjtuEtQbDEwTabF6yIO?si=5b6faa97915a4977"],
        "mix": ["https://open.spotify.com/playlist/6IXKFG4scJV5mi8wcqxcVt?si=dfc359a9df784520",
                "https://open.spotify.com/playlist/1UQrjtuEtQbDEwTabF6yIO?si=5b6faa97915a4977",
                "https://open.spotify.com/playlist/79xLMj1KeVMoUi41KqMUQe?si=11080a47019b4382"]
    },
}

# Mood order (9 moods)
MOOD_ORDER = [
    ("happy",        "Happy ☀"),
    ("sad_breakup",  "Sad / Breakup 💔"),
    ("motivational", "Motivational (Aura) ⚡"),
    ("chill_lofi",   "Chill / Lo-Fi 🌿"),
    ("hiphop",       "Hip Hop 🎧"),
    ("rap",          "Rap 🔥"),
    ("old_classic",  "Old / Classic 📼"),
    ("romantic",     "Romantic ❤"),
    ("party",        "Party / Dance 🎉"),
]

# ========================= HELPERS =========================

def _id_from_uri(x: str) -> str:
    return x.split("?")[0].split(":")[-1].split("/")[-1]

def ensure_active_device(sp: spotipy.Spotify) -> Optional[str]:
    devices = sp.devices().get("devices", [])
    for d in devices:
        if d.get("is_active"): return d.get("id")
    return devices[0]["id"] if devices else None

_PLAYLIST_NAME_CACHE: Dict[str, str] = {}
def playlist_name(sp: spotipy.Spotify, uri: str) -> str:
    pid = _id_from_uri(uri)
    if pid in _PLAYLIST_NAME_CACHE: return _PLAYLIST_NAME_CACHE[pid]
    try:
        pl = sp.playlist(pid, fields="name")
        _PLAYLIST_NAME_CACHE[pid] = pl.get("name", pid)
        return _PLAYLIST_NAME_CACHE[pid]
    except Exception:
        return pid

def fetch_playlist_tracks(sp: spotipy.Spotify, playlist_uri: str, first_page_only: bool=False) -> List[dict]:
    """
    Safe, defensive fetcher.
    Returns list of dicts:
      {uri,id,name,artists,popularity,source_playlist_uri,source_playlist_name}
    - Skips local tracks
    - Handles None artist names safely
    - Popularity defaults to 0 if missing
    """
    pid = _id_from_uri(playlist_uri)
    out: List[dict] = []
    fields = "items(track(id,uri,name,artists(name),popularity,is_local)),next"
    try:
        results = sp.playlist_items(pid, additional_types=["track"], fields=fields, limit=100)
    except Exception:
        return out  # if playlist fetch fails, return empty

    pl_name = playlist_name(sp, playlist_uri)

    while True:
        items = results.get("items") or []
        for it in items:
            tr = it.get("track") or {}
            if tr.get("is_local"):
                continue
            tid = tr.get("id")
            uri = tr.get("uri")
            if not tid or not uri:
                continue

            # Safely build artists string (drop None)
            raw_artists = tr.get("artists") or []
            artist_names = []
            for a in raw_artists:
                if isinstance(a, dict):
                    nm = a.get("name")
                    if nm:
                        artist_names.append(str(nm))
                elif isinstance(a, str):
                    artist_names.append(a)
            artists_str = ", ".join(artist_names)

            pop = tr.get("popularity")
            try:
                pop_int = int(pop) if pop is not None else 0
            except Exception:
                pop_int = 0

            out.append({
                "uri": uri,
                "id": tid,
                "name": tr.get("name") or "Unknown",
                "artists": artists_str,
                "popularity": pop_int,
                "source_playlist_uri": playlist_uri,
                "source_playlist_name": pl_name,
            })

        if first_page_only: break
        nxt = results.get("next")
        if not nxt: break
        results = sp.next(results)

    random.shuffle(out)
    return out

# ---------- Popularity & allocation helpers ----------

def split_by_popularity(pool: List[dict]) -> Tuple[List[dict], List[dict]]:
    high, mid = [], []
    for tr in pool:
        p = tr["popularity"]
        if p < POP_SKIP_BELOW: continue
        if p >= HIGH_POP_MIN: high.append(tr)
        elif p >= MID_POP_MIN: mid.append(tr)
    random.shuffle(high); random.shuffle(mid)
    return high, mid

def _eligible_bins_for_playlist(sp, pl, first_page_only, seen_uris):
    """Fetch tracks for a playlist, filter by popularity & seen, return {'high': [...], 'mid': [...]}"""
    tracks = fetch_playlist_tracks(sp, pl, first_page_only=first_page_only)
    uniq = []
    used = set()
    for tr in tracks:
        if tr["uri"] in seen_uris: 
            continue
        if tr["uri"] in used:
            continue
        used.add(tr["uri"])
        uniq.append(tr)
    high, mid = split_by_popularity(uniq)
    return {"high": high, "mid": mid}

def _proportional_quota(total_needed, sizes, max_share=0.70):
    """
    sizes: list of ints per playlist (eligible items count)
    returns integer quotas summing to total_needed, roughly proportional to sizes,
    and caps any single playlist to <= max_share of total_needed.
    """
    if not sizes or total_needed <= 0:
        return [0]*len(sizes)
    S = sum(sizes)
    if S == 0:
        return [0]*len(sizes)
    raw = [(s / S) * total_needed for s in sizes]
    cap = int(max(1, round(total_needed * max_share)))
    base = [min(cap, max(0, int(x // 1))) for x in raw]
    leftover = total_needed - sum(base)
    rema = [(i, raw[i] - base[i]) for i in range(len(sizes))]
    for i, _ in sorted(rema, key=lambda t: t[1], reverse=True):
        if leftover == 0:
            break
        if base[i] < cap and base[i] < sizes[i]:
            base[i] += 1
            leftover -= 1
    return base

def _sample_high_mid(bins, k):
    """bins = {'high':[...], 'mid':[...]} with fallback across bins."""
    if k <= 0:
        return []
    kh = int(round(k * TARGET_HIGH_RATIO))
    km = k - kh
    out = []
    take_h = min(kh, len(bins["high"]))
    out += bins["high"][:take_h]
    take_m = min(km, len(bins["mid"]))
    out += bins["mid"][:take_m]
    remaining = k - len(out)
    ih, im = take_h, take_m
    while remaining > 0 and (ih < len(bins["high"]) or im < len(bins["mid"])):
        if ih < len(bins["high"]):
            out.append(bins["high"][ih]); ih += 1; remaining -= 1
            if remaining == 0: break
        if im < len(bins["mid"]):
            out.append(bins["mid"][im]); im += 1; remaining -= 1
    random.shuffle(out)
    return out[:k]

# ---------- Weighting & batching ----------

def prompt_language() -> str:
    print("\nSelect Language:")
    print("  1) Hindi 🇮🇳")
    print("  2) English 🇬🇧")
    print("  3) Both 🌍")
    while True:
        x = input("Enter 1/2/3: ").strip()
        if x == "1": return "hi"
        if x == "2": return "en"
        if x == "3": return "mix"
        print("Invalid choice. Please enter 1, 2, or 3.")

def prompt_weights() -> Dict[str, int]:
    weights = {key: 0 for key, _ in MOOD_ORDER}
    total = 0
    print("\nSet weights for the 9 moods (allowed: 0,5,10,...,100).")
    print("Tip: When total reaches 100, remaining moods default to 0.\n")
    for key, label in MOOD_ORDER:
        remaining = 100 - total
        if remaining == 0:
            print("Total reached 100. Skipping remaining moods.")
            break
        prompt = f"{label} — enter weight (0..{remaining}, step 5): "
        while True:
            raw = input(prompt).strip()
            if not raw.isdigit():
                print("Please enter a number."); continue
            val = int(raw)
            if val % 5 != 0:
                print("Weight must be a multiple of 5."); continue
            if val < 0 or val > remaining:
                print(f"Enter value between 0 and {remaining}."); continue
            weights[key] = val
            total += val
            print(f"Running total = {total}/100\n")
            break
    return weights

def collect_playlists_for_mood(language: str, mood_key: str) -> List[str]:
    return (MOOD_BUCKETS.get(mood_key) or {}).get(language) or (MOOD_BUCKETS.get(mood_key) or {}).get("mix") or []

def proportional_split(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    positives = {k: w for k, w in weights.items() if w > 0}
    W = sum(positives.values())
    if W == 0 or total == 0: return {k: 0 for k in weights}
    alloc, leftover = {}, total
    for k, w in positives.items():
        share = (w / W) * total
        n = max(1, int(math.floor(share)))
        alloc[k] = n; leftover -= n
    if leftover < 0:
        for k in sorted(alloc, key=lambda x: alloc[x], reverse=True):
            if leftover == 0: break
            if alloc[k] > 1:
                dec = min(alloc[k]-1, -leftover)
                alloc[k] -= dec; leftover += dec
    if leftover > 0:
        rema = []
        for k, w in positives.items():
            exact = (w / W) * total
            rema.append((k, exact - alloc[k]))
        for k, _r in sorted(rema, key=lambda x: x[1], reverse=True):
            if leftover == 0: break
            alloc[k] += 1; leftover -= 1
    return {k: alloc.get(k, 0) for k in weights.keys()}

def build_batch(sp: spotipy.Spotify, language: str,
                mood_weights: Dict[str, int], batch_size: int,
                seen_uris: set, first_page_only: bool=False) -> List[dict]:
    """
    Build a batch with per-mood allocation and per-playlist proportional sampling,
    popularity-aware, excluding seen URIs.
    """
    chosen: List[dict] = []
    per_mood_counts = proportional_split(batch_size, mood_weights)

    for mood_key, count in per_mood_counts.items():
        if count <= 0:
            continue
        playlists = collect_playlists_for_mood(language, mood_key)
        if not playlists:
            continue

        # Build eligible bins per playlist for this mood
        bins_per_pl = []
        for pl in playlists:
            bins_per_pl.append(_eligible_bins_for_playlist(sp, pl, first_page_only, seen_uris))

        # Eligible sizes per playlist (high+mid)
        sizes = [len(b["high"]) + len(b["mid"]) for b in bins_per_pl]

        # Allocate this mood's count proportionally; cap any single playlist's share
        quotas = _proportional_quota(count, sizes, max_share=0.70)

        # Sample from each playlist as per its quota, preserving high:mid target
        picks = []
        for b, q in zip(bins_per_pl, quotas):
            if q > 0:
                picks.extend(_sample_high_mid(b, q))

        # If short due to thin bins, fill from any remaining eligible across all
        deficit = count - len(picks)
        if deficit > 0:
            rem_high, rem_mid = [], []
            for b, q in zip(bins_per_pl, quotas):
                rem_high.extend(b["high"])
                rem_mid.extend(b["mid"])
            random.shuffle(rem_high); random.shuffle(rem_mid)
            merged = {"high": rem_high, "mid": rem_mid}
            extra = _sample_high_mid(merged, deficit)
            picks.extend(extra)

        for tr in picks:
            tr["mood_key"] = mood_key
        chosen.extend(picks)

    random.shuffle(chosen)
    return chosen

# ---------- Playback helpers & auto top-up ----------

def log_track(prefix: str, tr: dict):
    print(f"{prefix} [{tr.get('mood_key','?')}] {tr['name']} — {tr['artists']}  "
          f"(pop {tr['popularity']})  from: {tr['source_playlist_name']}")

def queue_only(sp: spotipy.Spotify, device_id: str, tracks: List[dict]):
    for tr in tracks:
        try:
            sp.add_to_queue(tr["uri"], device_id=device_id)
            log_track("➕ Queued     ", tr)
            time.sleep(0.03)
        except SpotifyException as e:
            print("add_to_queue error:", e)

def start_with_seed_then_queue(sp: spotipy.Spotify, device_id: str, seed: List[dict], rest: List[dict]):
    if not seed:
        print("No seed tracks to start."); return
    first = seed[0]
    try:
        sp.start_playback(device_id=device_id, uris=[first["uri"]])
        log_track("▶  Now playing", first)
    except SpotifyException as e:
        print("start_playback error:", e); return
    # queue remaining seed (if any)
    if len(seed) > 1:
        queue_only(sp, device_id, seed[1:])
    # queue the rest as they’re ready
    if rest:
        queue_only(sp, device_id, rest)

def safe_get_queue(sp):
    try:
        return sp.queue()
    except Exception:
        return None

def monitor_and_topup(sp: spotipy.Spotify, device_id: str,
                      language: str, weights: Dict[str, int],
                      seen_uris: set, next_topup_after: int):
    """
    Robust top-up trigger:
    - increments counter when track ID changes OR progress_ms drops notably (skip/next),
    - also tops up when the remaining queue is low (<= QUEUE_LOW_WATER).
    """
    played = 0
    last_track_id = None
    last_progress = None
    threshold = next_topup_after

    print(f"\n🔁 Auto top-up armed: +{TOP_UP_BATCH} after every {TOP_UP_EVERY} plays/skips, "
          f"and also when queue ≤ {QUEUE_LOW_WATER}.\n")

    while True:
        try:
            pb = sp.current_playback()
        except SpotifyException:
            time.sleep(POLL_SECONDS)
            continue

        # detect track change or skip (progress reset)
        changed = False
        if pb and pb.get("item"):
            tid = pb["item"].get("id")
            prog = pb.get("progress_ms")

            if last_track_id is None:
                last_track_id = tid
                last_progress = prog

            if tid and tid != last_track_id:
                changed = True
            elif prog is not None and last_progress is not None:
                # large negative delta = probably skip/next
                if prog + 5000 < last_progress:
                    changed = True

            last_track_id = tid
            last_progress = prog

        if changed:
            played += 1
            print(f"♪ Progress: {played} tracks played/skipped…")

        # queue length check (safety net)
        need_topup = False
        q = safe_get_queue(sp)
        if q and isinstance(q, dict):
            queue_list = q.get("queue") or []
            # This is Spotify's pending queue (not including current track)
            if len(queue_list) <= QUEUE_LOW_WATER:
                print(f"⚠  Queue buffer low ({len(queue_list)}). Will top up.")
                need_topup = True

        # threshold check
        if played >= threshold:
            print(f"✅ Reached {played} plays/skips (threshold {threshold}). Will top up.")
            need_topup = True

        if need_topup:
            print(f"\n⬆  Top-up: building {TOP_UP_BATCH} more (balanced, no duplicates)…")
            add = build_batch(sp, language, weights, TOP_UP_BATCH, seen_uris, first_page_only=False)
            if not add:
                print("No new tracks available to top-up (all seen or empty pools).")
            else:
                queue_only(sp, device_id, add)   # only queue; do not interrupt current song
                for tr in add: seen_uris.add(tr["uri"])
                print(f"✅ Top-up complete: +{len(add)} tracks.\n")
            # move threshold forward to next block
            while threshold <= played:
                threshold += TOP_UP_EVERY

        time.sleep(POLL_SECONDS)

# ========================= MAIN =========================

def main():
    print("\n=== Spotify Mood Mixer (instant start + auto top-up) — 9 moods, playlist-balanced ===\n")

    # Auth
    os.environ["SPOTIPY_CLIENT_ID"] = SPOTIFY_CLIENT_ID
    os.environ["SPOTIPY_CLIENT_SECRET"] = SPOTIFY_CLIENT_SECRET
    os.environ["SPOTIPY_REDIRECT_URI"] = SPOTIFY_REDIRECT_URI
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SCOPE
    ))

    device_id = ensure_active_device(sp)
    if not device_id:
        print("No active device found. Open Spotify, play/pause once, then run again.")
        sys.exit(1)

    # Language + mood weights
    lang = prompt_language()  # "hi" | "en" | "mix"
    print("\nSet weights for the 9 moods below. Use multiples of 5. Total target = 100.\n")
    for key, label in MOOD_ORDER: print(f" - {label}")
    weights = prompt_weights()
    total = sum(weights.values())
    print(f"\nFinal total = {total}/100")

    # Build a small seed fast (first page only) and start immediately
    seen_uris = set()
    seed = build_batch(sp, lang, weights, SEED_START, seen_uris, first_page_only=True)
    for tr in seed: seen_uris.add(tr["uri"])

    # Build the remainder of the initial batch (full fetch)
    remaining_needed = max(0, INITIAL_BATCH - len(seed))
    rest = []
    if remaining_needed > 0:
        rest = build_batch(sp, lang, weights, remaining_needed, seen_uris, first_page_only=False)
        for tr in rest: seen_uris.add(tr["uri"])

    print(f"\n▶  Starting now with {len(seed)} seed tracks, then queuing {len(rest)} more…\n")
    start_with_seed_then_queue(sp, device_id, seed, rest)

    # Monitor playback and auto top-up forever
    monitor_and_topup(sp, device_id, lang, weights, seen_uris, next_topup_after=TOP_UP_EVERY)

if __name__ == "__main__":
    main()