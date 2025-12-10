# modelcode.py
# Queue-based player (no playlist context). Builds an initial queue, then
# keeps appending new recommendations as you listen.

import os, json, random, time, threading
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from dateutil import parser as dateparser

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

# Import configuration
try:
    from config import Config
except ImportError:
    print("Warning: config.py not found, using default values")
    class Config:
        SPOTIPY_CLIENT_ID = None
        SPOTIPY_CLIENT_SECRET = None
        SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:8888/callback'
        SPOTIFY_MARKET = 'IN'
        SPOTIFY_SCOPES = 'user-modify-playback-state user-read-playback-state user-read-currently-playing'
        SPOTIFY_CACHE_PATH = '.cache_spotify_export'
        INITIAL_COUNT = 50
        APPEND_EVERY_N = 15
        APPEND_BATCH = 20
        TOP_N_BOOST = 10
        ALBUM_TOP_N = 3
        RELATED_MAX = 5

# ================== CONFIG ==================
CLIENT_ID     = Config.SPOTIPY_CLIENT_ID
CLIENT_SECRET = Config.SPOTIPY_CLIENT_SECRET
REDIRECT_URI  = Config.SPOTIPY_REDIRECT_URI
MARKET        = Config.SPOTIFY_MARKET

USER_JSON_PATH  = "metadata.json"  # keep beside this script

INITIAL_COUNT   = Config.INITIAL_COUNT
APPEND_EVERY_N  = Config.APPEND_EVERY_N
APPEND_BATCH    = Config.APPEND_BATCH

# Playback scopes (no library modification needed)
SCOPES = Config.SPOTIFY_SCOPES

# ===== Strategy weights per mode (sum to 1.0) =====
# 1) artist_top_tracks      2) artist_album_pick     3) era_year_bias
# 4) genre_explore (ALL-TIME ONLY)                   5) favorite_throwback
# 6) short_term_boost       7) medium_term_boost     8) playlist_artist
# 9) playlist_genre (PLAYLIST-ONLY)
STRATEGY_WEIGHTS = {
    "comfort": {
        "artist_top_tracks": 0.35,
        "artist_album_pick": 0.20,
        "era_year_bias":     0.10,
        "genre_explore":     0.08,  # all-time genres only
        "favorite_throwback":0.12,
        "short_term_boost":  0.10,
        "medium_term_boost": 0.03,
        "playlist_artist":   0.01,
        "playlist_genre":    0.01   # playlist genres only
    },
    "balanced": {
        "artist_top_tracks": 0.22,
        "artist_album_pick": 0.15,
        "era_year_bias":     0.15,
        "genre_explore":     0.18,  # all-time genres only
        "favorite_throwback":0.08,
        "short_term_boost":  0.10,
        "medium_term_boost": 0.06,
        "playlist_artist":   0.03,
        "playlist_genre":    0.03   # playlist genres only
    },
    "explorer": {
        "artist_top_tracks": 0.12,
        "artist_album_pick": 0.08,
        "era_year_bias":     0.22,
        "genre_explore":     0.26,  # all-time genres only
        "favorite_throwback":0.04,
        "short_term_boost":  0.10,
        "medium_term_boost": 0.08,
        "playlist_artist":   0.05,
        "playlist_genre":    0.05   # playlist genres only
    }
}

# Tunables for selection windows
TOP_N_BOOST = Config.TOP_N_BOOST
ALBUM_TOP_N = Config.ALBUM_TOP_N
RELATED_MAX = Config.RELATED_MAX
# ============================================

@dataclass
class Reason:
    strategy: str
    details: str

class QueuePlayer:
    def __init__(self, mode: str = "balanced"):
        self.mode = mode if mode in STRATEGY_WEIGHTS else "balanced"
        self.strategy_weights = STRATEGY_WEIGHTS[self.mode]

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_path=os.path.join(os.path.dirname(__file__), Config.SPOTIFY_CACHE_PATH),
            open_browser=True
        ))

        # ---------- Load profile ----------
        self.profile = self._load_profile(USER_JSON_PATH)

        # Long-term artists are the backbone for core strategies
        self.long_artists = self.profile["top_artists"]["long_term"]
        self.short_artists = self.profile["top_artists"].get("short_term", [])
        self.medium_artists = self.profile["top_artists"].get("medium_term", [])
        self.playlist_artists = self.profile.get("playlists_summary", {}) \
                                         .get("top_artists", [])

        # Weighted artist selection from long-term
        self.artist_weights = self._build_artist_weights(self.long_artists)
        self.top_artist_ids = [a["id"] for a in self.long_artists]

        # Eras (from long-term top 50 tracks)
        self.decade_weights, self.year_weights = self._build_era_weights(self.profile.get("eras", {}))

        # Genres (separate pools)
        self.genres_all_time = {g["genre"]: g["count"] for g in self.profile.get("top_genres_all_time", [])}
        self.genres_playlist = {g["genre"]: g["count"] for g in self.profile.get("playlists_summary", {}).get("top_genres", [])}

        # Favorites: top 50 tracks (long-term)
        self.favs = self.profile.get("top_tracks", {}).get("long_term", [])[:50]

        # State (dedup and flow)
        self.queued_or_played: Set[str] = set()
        self.last_artist_id: Optional[str] = None
        self.last_track_id: Optional[str] = None
        self.tracks_played_since_append = 0

        # Device
        self.device_id = self._pick_device()
        if not self.device_id:
            raise RuntimeError("No active Spotify device found. Open Spotify on any device and try again.")

    # ---------- Profile ----------
    def _load_profile(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_artist_weights(self, top_artists: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        weights = []
        for idx, a in enumerate(top_artists, start=1):
            aid = a["id"]
            if self.mode == "comfort":
                base = max(1, 51 - idx) ** 1.5
                if idx <= TOP_N_BOOST: base *= 2.0
            elif self.mode == "balanced":
                base = max(1, 51 - idx) ** 1.2
                if idx <= TOP_N_BOOST: base *= 1.5
            else:  # explorer
                base = max(1, 51 - idx) ** 0.8
                if idx <= TOP_N_BOOST: base *= 1.2
            weights.append((aid, float(base)))
        total = sum(w for _, w in weights) or 1.0
        return [(aid, w/total) for aid, w in weights]

    def _build_era_weights(self, era: Dict[str, Any]):
        decades = era.get("decades", [])
        years   = era.get("years", [])
        d_weights, y_weights = [], []
        if decades:
            total = sum(d["count"] for d in decades) or 1
            for d in decades:
                # slightly different smoothing per mode
                smooth = 2 if self.mode=="comfort" else 5 if self.mode=="balanced" else 10
                w = (d["count"] + smooth) / (total + smooth*len(decades))
                d_weights.append((d["decade"], float(w)))
        if years:
            total = sum(y["count"] for y in years) or 1
            for y in years:
                smooth = 0.5 if self.mode=="comfort" else 1 if self.mode=="balanced" else 2
                w = (y["count"] + smooth) / (total + smooth*len(years))
                y_weights.append((int(y["year"]), float(w)))
        return d_weights, y_weights

    # ---------- Spotify helpers ----------
    def _devices(self):
        return self.sp.devices().get("devices", [])

    def _pick_device(self) -> Optional[str]:
        devs = self._devices()
        for d in devs:
            if d.get("is_active"): return d["id"]
        return devs[0]["id"] if devs else None

    def _transfer_and_play_single(self, track_id: str):
        try:
            self.sp.transfer_playback(self.device_id, force_play=True)
            time.sleep(0.2)
        except SpotifyException:
            pass
        self.sp.start_playback(device_id=self.device_id, uris=[f"spotify:track:{track_id}"], position_ms=0)

    def _add_to_queue(self, track_id: str):
        self.sp.add_to_queue(uri=f"spotify:track:{track_id}", device_id=self.device_id)

    def _current_playback(self):
        try:
            return self.sp.current_playback()
        except SpotifyException:
            return None

    # ---------- Utility ----------
    def _pick_weighted(self, items_with_weights: List[Tuple[Any, float]]):
        xs, ws = zip(*items_with_weights)
        return random.choices(xs, weights=ws, k=1)[0]

    def _choose_strategy(self) -> str:
        keys = list(self.strategy_weights.keys())
        vals = [self.strategy_weights[k] for k in keys]
        return random.choices(keys, weights=vals, k=1)[0]

    # ---------- Core Strategies ----------
    def strategy_artist_top_tracks(self):
        aid = self._pick_weighted(self.artist_weights)
        art = self.sp.artist(aid)
        tops = self.sp.artist_top_tracks(aid, country=MARKET).get("tracks", [])
        if not tops:
            res = self.sp.search(q=f'artist:"{art["name"]}"', type="track", market=MARKET, limit=20)
            tops = res.get("tracks", {}).get("items", [])
        # mode-dependent window
        win = ALBUM_TOP_N + (0 if self.mode=="comfort" else 2 if self.mode=="balanced" else 5)
        picks = tops[:max(1, min(win, len(tops)))]
        t = random.choice(picks)
        return t["id"], Reason("artist_top_tracks", f"Weighted long-term artist -> {art['name']}")

    def strategy_artist_album_pick(self):
        aid = self._pick_weighted(self.artist_weights)
        art = self.sp.artist(aid)
        albums = self.sp.artist_albums(aid, album_type="album,single", country=MARKET, limit=50).get("items", [])
        if not albums: return self.strategy_artist_top_tracks()
        def parse_date(d):
            try: return dateparser.parse(d).date()
            except: return datetime(1970,1,1).date()
        albums.sort(key=lambda a: parse_date(a.get("release_date", "1970-01-01")), reverse=True)
        chosen = albums[0] if self.mode=="comfort" else random.choice(albums[:3 if self.mode=="balanced" else 8])
        tracks = self.sp.album_tracks(chosen["id"], limit=50).get("items", [])
        if not tracks: return self.strategy_artist_top_tracks()
        win = ALBUM_TOP_N + (0 if self.mode=="comfort" else 3 if self.mode=="balanced" else 7)
        t = random.choice(tracks[:max(1, min(win, len(tracks)))])
        return t["id"], Reason("artist_album_pick", f"{art['name']} -> album {chosen['name']}")

    def strategy_era_year_bias(self):
        if self.decade_weights:
            decade = self._pick_weighted(self.decade_weights)
            base = int(decade[:4]); year = base + random.randint(0,9)
        elif self.year_weights:
            year = self._pick_weighted(self.year_weights)
        else:
            year = random.randint(2000, datetime.now().year)
        results = self.sp.search(q=f"year:{year}", type="track", market=MARKET, limit=50)
        tracks = results.get("tracks", {}).get("items", [])
        if not tracks:
            return self.strategy_artist_top_tracks()
        exp = 1.5 if self.mode=="comfort" else 1.2 if self.mode=="balanced" else 0.8
        pops = [max(1, (t.get("popularity") or 50) ** exp) for t in tracks]
        t = random.choices(tracks, weights=pops, k=1)[0]
        return t["id"], Reason("era_year_bias", f"Favored year {year} ({(year//10)*10}s)")

    def strategy_genre_explore(self):
        """ALL-TIME genres only (top_genres_all_time)."""
        if not self.genres_all_time:
            return self.strategy_artist_top_tracks()
        genres = list(self.genres_all_time.items())
        # gentle smoothing by mode
        smooth = 2 if self.mode=="comfort" else 5 if self.mode=="balanced" else 10
        gweights = [(g, c + smooth) for g, c in genres]
        genre = self._pick_weighted(gweights)

        # Prefer long-term artists whose genres contain the picked genre
        candidates = [a for a in self.long_artists if genre.lower() in (a.get("genres","").lower())]
        if not candidates:
            # Fallback: use related artists off a long-term seed, filtered by genre
            seed_aid = random.choice(self.top_artist_ids)
            related = self.sp.artist_related_artists(seed_aid).get("artists", [])
            rel = [r for r in related if genre.lower() in " ".join(r.get("genres", [])).lower()]
            if rel:
                pick_artist = random.choice(rel)
                tops = self.sp.artist_top_tracks(pick_artist["id"], country=MARKET).get("tracks", [])
                if tops:
                    win = RELATED_MAX + (0 if self.mode=="comfort" else 2 if self.mode=="balanced" else 5)
                    t = random.choice(tops[:max(1, min(win, len(tops)))])
                    return t["id"], Reason("genre_explore", f"All-time genre '{genre}' -> {pick_artist['name']}")
            # Final fallback: a top track from a long-term artist
            return self.strategy_artist_top_tracks()

        # Choose a matching long-term artist and one of their top tracks
        pick_a = random.choice(candidates)
        tops = self.sp.artist_top_tracks(pick_a["id"], country=MARKET).get("tracks", [])
        if not tops:
            return self.strategy_artist_top_tracks()
        win = RELATED_MAX + (0 if self.mode=="comfort" else 2 if self.mode=="balanced" else 5)
        t = random.choice(tops[:max(1, min(win, len(tops)))])
        return t["id"], Reason("genre_explore", f"All-time genre '{genre}' -> {pick_a['name']}")

    def strategy_favorite_throwback(self):
        if not self.favs:
            return self.strategy_artist_top_tracks()
        # mode window for how deep we pick from favs
        if self.mode == "comfort":
            pool = self.favs[:10]
        elif self.mode == "balanced":
            pool = self.favs[:25]
        else:
            pool = self.favs
        pick = random.choice(pool)
        tid = pick["id"]
        tr = self.sp.track(tid)
        return tid, Reason("favorite_throwback", f"Top-tracks throwback -> {tr['name']}")

    # ---------- New Strategies (short/medium & playlist-driven) ----------
    def strategy_short_term_boost(self):
        if not self.short_artists:
            return self.strategy_artist_top_tracks()
        a = random.choice(self.short_artists)
        tops = self.sp.artist_top_tracks(a["id"], country=MARKET).get("tracks", [])
        if not tops:
            return self.strategy_artist_top_tracks()
        t = random.choice(tops[:max(1, min(ALBUM_TOP_N+2, len(tops)))])
        return t["id"], Reason("short_term_boost", f"Recent favorite artist -> {a['name']}")

    def strategy_medium_term_boost(self):
        if not self.medium_artists:
            return self.strategy_artist_top_tracks()
        a = random.choice(self.medium_artists)
        tops = self.sp.artist_top_tracks(a["id"], country=MARKET).get("tracks", [])
        if not tops:
            return self.strategy_artist_top_tracks()
        t = random.choice(tops[:max(1, min(ALBUM_TOP_N+3, len(tops)))])
        return t["id"], Reason("medium_term_boost", f"Mid-term favorite artist -> {a['name']}")

    def strategy_playlist_artist(self):
        if not self.playlist_artists:
            return self.strategy_artist_top_tracks()
        a = random.choice(self.playlist_artists)  # {'id','name', ...}
        if not a.get("id"):
            return self.strategy_artist_top_tracks()
        tops = self.sp.artist_top_tracks(a["id"], country=MARKET).get("tracks", [])
        if not tops:
            return self.strategy_artist_top_tracks()
        t = random.choice(tops[:max(1, min(RELATED_MAX, len(tops)))])
        return t["id"], Reason("playlist_artist", f"Playlist-heavy artist -> {a.get('name','(unknown)')}")

    def strategy_playlist_genre(self):
        """PLAYLIST genres only (from playlists_summary.top_genres)."""
        if not self.genres_playlist:
            return self.strategy_artist_top_tracks()
        genres = list(self.genres_playlist.items())
        smooth = 2 if self.mode=="comfort" else 5 if self.mode=="balanced" else 10
        gweights = [(g, c + smooth) for g, c in genres]
        genre = self._pick_weighted(gweights)

        # Try to find a seed artist with this playlist-genre (long-term or playlist artists)
        candidates = [a for a in self.long_artists if genre.lower() in (a.get("genres","").lower())]
        if not candidates:
            candidates = [a for a in self.playlist_artists if genre.lower() in " ".join(a.get("genres", [])).lower()]
        if candidates:
            a = random.choice(candidates)
            aid = a["id"]
            tops = self.sp.artist_top_tracks(aid, country=MARKET).get("tracks", [])
            if tops:
                t = random.choice(tops[:max(1, min(RELATED_MAX+2, len(tops)))])
                return t["id"], Reason("playlist_genre", f"Playlist genre '{genre}' -> {a.get('name','(artist)')}")

        # Fallback: search tracks loosely by text (not perfect, but works broadly)
        res = self.sp.search(q=genre, type="track", market=MARKET, limit=20)
        picks = res.get("tracks", {}).get("items", [])
        if picks:
            t = random.choice(picks)
            return t["id"], Reason("playlist_genre", f"Playlist genre '{genre}' -> text search pick")
        return self.strategy_artist_top_tracks()

    # ---------- Orchestrator ----------
    def _generate_one(self) -> Tuple[str, Reason]:
        strat = self._choose_strategy()
        if strat == "artist_top_tracks":   return self.strategy_artist_top_tracks()
        if strat == "artist_album_pick":   return self.strategy_artist_album_pick()
        if strat == "era_year_bias":       return self.strategy_era_year_bias()
        if strat == "genre_explore":       return self.strategy_genre_explore()      # ALL-TIME ONLY
        if strat == "favorite_throwback":  return self.strategy_favorite_throwback()
        if strat == "short_term_boost":    return self.strategy_short_term_boost()
        if strat == "medium_term_boost":   return self.strategy_medium_term_boost()
        if strat == "playlist_artist":     return self.strategy_playlist_artist()
        if strat == "playlist_genre":      return self.strategy_playlist_genre()     # PLAYLIST-ONLY
        # safety
        return self.strategy_artist_top_tracks()

    def _build_batch(self, n: int) -> List[Tuple[str, Reason]]:
        out: List[Tuple[str, Reason]] = []
        attempts = 0
        while len(out) < n and attempts < n * 15:
            attempts += 1
            try:
                tid, reason = self._generate_one()
                if tid in self.queued_or_played:
                    continue
                tr = self.sp.track(tid)
                artist_id = tr["artists"][0]["id"] if tr.get("artists") else None
                # avoid consecutive same artist
                if (out and artist_id == self._get_artist_of_track(out[-1][0])) or (not out and artist_id == self.last_artist_id):
                    continue
                out.append((tid, reason))
            except Exception:
                continue
        return out

    def _get_artist_of_track(self, track_id: str) -> Optional[str]:
        try:
            tr = self.sp.track(track_id)
            return tr["artists"][0]["id"] if tr.get("artists") else None
        except Exception:
            return None

    # ---------- Public flow ----------
    def start_session(self):
        mode_names = {
            "comfort":  "Comfort Zone (top artists + favorites, gentle explore)",
            "balanced": "Balanced Explore (mix of top, recent, playlist & genres)",
            "explorer": "Full Explorer (eras + genres + wider discovery)"
        }
        print(f"Building initial {INITIAL_COUNT} tracks in {mode_names[self.mode]}...")
        first_batch = self._build_batch(INITIAL_COUNT)
        if not first_batch:
            raise RuntimeError("Could not build initial queue.")
        # play 1st, queue rest
        first_track, first_reason = first_batch[0]
        self._transfer_and_play_single(first_track)
        self.queued_or_played.add(first_track)
        self.last_artist_id = self._get_artist_of_track(first_track)
        print("\nNow playing:")
        self._print_entry(first_track, first_reason)

        for tid, r in first_batch[1:]:
            self._add_to_queue(tid)
            self.queued_or_played.add(tid)
            self._print_entry(tid, r, queued=True)

        print(f"\nWatcher: will append {APPEND_BATCH} new tracks after every {APPEND_EVERY_N} plays.")
        t = threading.Thread(target=self._watch_and_append, daemon=True)
        t.start()

        print("\nUse your normal Spotify app/keys. Leave this running. Ctrl+C to stop.\n")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("\nBye.")

    def _watch_and_append(self):
        while True:
            try:
                cur = self._current_playback()
                if not cur or not cur.get("item"):
                    time.sleep(3); continue
                tid = cur["item"]["id"]
                if tid and tid != self.last_track_id:
                    # track advanced
                    self.tracks_played_since_append += 1
                    self.last_track_id = tid
                    self.queued_or_played.add(tid)
                    self.last_artist_id = self._get_artist_of_track(tid)
                    print(f"(Played: {self.tracks_played_since_append}/{APPEND_EVERY_N})")
                    if self.tracks_played_since_append >= APPEND_EVERY_N:
                        more = self._build_batch(APPEND_BATCH)
                        for tid2, r2 in more:
                            self._add_to_queue(tid2)
                            self.queued_or_played.add(tid2)
                            self._print_entry(tid2, r2, queued=True)
                        self.tracks_played_since_append = 0
                time.sleep(3)
            except Exception:
                time.sleep(5)

    def _print_entry(self, tid: str, reason: Reason, queued: bool=False):
        tr = self.sp.track(tid)
        name = tr["name"]
        artist = ", ".join([a["name"] for a in tr.get("artists", [])])
        album = tr.get("album", {}).get("name", "")
        prefix = "Queued" if queued else "Playing"
        print(f"  - {prefix}: {name} - {artist} | {album}")
        print(f"    Reason: [{reason.strategy}] {reason.details}")

def get_user_mode():
    print("="*64)
    print("SPOTIFY QUEUE PLAYER - SELECT MODE")
    print("="*64)
    print("1. Comfort Zone\n2. Balanced Explore\n3. Full Explorer\n")
    choice = input("> ").strip()
    return "comfort" if choice=="1" else "balanced" if choice=="2" else "explorer"

def main():
    # Validate configuration
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease check your .env file or run 'python config.py' to create a template.")
        return
    
    print("Configuration loaded successfully")
    Config.print_config()
    print()
    
    try:
        mode = get_user_mode()
        qp = QueuePlayer(mode=mode)
        qp.start_session()
    except Exception as e:
        print(f"Failed to start recommendation system: {e}")
        print("Please check your Spotify credentials in the .env file")
        return

if __name__ == "__main__":
    main()
