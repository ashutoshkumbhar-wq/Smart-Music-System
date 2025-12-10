# pip install spotipy pandas tqdm numpy python-dotenv

import os, re, sys, json
from urllib.parse import urlparse
from typing import List, Optional

import pandas as pd
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Import configuration
try:
    from config import Config
except ImportError:
    print("Warning: config.py not found, using default values")
    class Config:
        SPOTIPY_CLIENT_ID = None
        SPOTIPY_CLIENT_SECRET = None
        SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'
        SPOTIFY_MARKET = 'IN'
        SPOTIFY_SCOPES = 'playlist-read-private playlist-read-collaborative user-top-read'
        SPOTIFY_CACHE_PATH = '.cache_spotify_export'
        PLAYLIST_URLS_FILE = os.path.join(os.path.dirname(__file__), "playlist_urls.txt")
        OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")

# ================== CONFIG ==================
HERE                = os.path.abspath(os.path.dirname(__file__))
PLAYLIST_URLS_FILE  = Config.PLAYLIST_URLS_FILE
OUTPUT_JSON_PATH    = Config.OUTPUT_JSON_PATH

CLIENT_ID     = Config.SPOTIPY_CLIENT_ID
CLIENT_SECRET = Config.SPOTIPY_CLIENT_SECRET
REDIRECT_URI  = Config.SPOTIPY_REDIRECT_URI
MARKET        = Config.SPOTIFY_MARKET
SCOPES        = Config.SPOTIFY_SCOPES
# =============================================


# ================== Helpers ==================
def spotify_client() -> spotipy.Spotify:
    auth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=os.path.join(HERE, Config.SPOTIFY_CACHE_PATH)
    )
    return spotipy.Spotify(auth_manager=auth, requests_timeout=30, retries=3)


def parse_playlist_id(url: str) -> Optional[str]:
    m = re.search(r"(playlist/|playlist:)([A-Za-z0-9]+)", url)
    if m:
        return m.group(2)
    try:
        last = urlparse(url).path.strip("/").split("/")[-1]
        return last if last else None
    except Exception:
        return None


def read_playlist_ids(path: str) -> List[str]:
    """Read playlist IDs from file. If file is missing/empty, return []."""
    if not os.path.exists(path):
        print(f"[INFO] {path} not found. Skipping playlist summary.")
        return []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            pid = parse_playlist_id(url)
            if pid:
                ids.append(pid)
    # de-dup, keep order
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    if not out:
        print(f"[INFO] No valid playlist URLs in {path}. Skipping playlist summary.")
    return out


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]
# =============================================


# ================== Playlist Summary ==================
def summarize_playlists(sp: spotipy.Spotify, playlist_ids: List[str]):
    """Return aggregated top artists + genres across playlists."""
    if not playlist_ids:
        return [], []

    artist_counter = {}
    genre_counter  = {}

    for pid in tqdm(playlist_ids, desc="Summarizing playlists"):
        try:
            items = sp.playlist_items(pid, market=MARKET, additional_types=["track"], limit=100)
            while items:
                for it in items.get("items", []):
                    tr = it.get("track") or {}
                    artists = tr.get("artists") or []
                    for a in artists:
                        aid, aname = a.get("id"), a.get("name")
                        if not aid:
                            continue
                        artist_counter[(aid, aname)] = artist_counter.get((aid, aname), 0) + 1
                if items.get("next"):
                    items = sp.next(items)
                else:
                    break
        except spotipy.exceptions.SpotifyException as e:
            print(f"[WARN] Failed playlist {pid}: {e}")
            continue

    # fetch genres from artist API
    artist_meta = {}
    if artist_counter:
        for batch in chunked([aid for (aid, _) in artist_counter.keys()], 50):
            arts = sp.artists(batch).get("artists", [])
            for a in arts:
                artist_meta[a["id"]] = {
                    "genres": a.get("genres", []),
                    "popularity": a.get("popularity"),
                    "followers": (a.get("followers") or {}).get("total"),
                }

    # build summary
    top_artists = []
    for (aid, aname), count in sorted(artist_counter.items(), key=lambda x: x[1], reverse=True)[:25]:
        meta = artist_meta.get(aid, {})
        top_artists.append({
            "id": aid, "name": aname, "count": count,
            "genres": meta.get("genres", []),
            "popularity": meta.get("popularity"),
            "followers": meta.get("followers")
        })
        for g in meta.get("genres", []):
            genre_counter[g] = genre_counter.get(g, 0) + 1

    top_genres = [{"genre": g, "count": c} for g, c in sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)[:25]]
    return top_artists, top_genres
# ======================================================


# ================== Top Artists/Tracks ==================
def get_top_dfs(sp: spotipy.Spotify):
    """Get top artists and tracks with limits."""
    def top_artists(time_range, limit):
        items = sp.current_user_top_artists(limit=limit, time_range=time_range).get("items", [])
        return pd.DataFrame([{
            "id": a.get("id"),
            "name": a.get("name"),
            "popularity": a.get("popularity"),
            "followers": (a.get("followers") or {}).get("total"),
            "genres": ", ".join(a.get("genres", []))
        } for a in items])

    def top_tracks(time_range, limit):
        items = sp.current_user_top_tracks(limit=limit, time_range=time_range).get("items", [])
        return pd.DataFrame([{
            "id": t.get("id"),
            "name": t.get("name"),
            "album": (t.get("album") or {}).get("name"),
            "album_id": (t.get("album") or {}).get("id"),
            "release_date": (t.get("album") or {}).get("release_date"),
            "artist_names": ", ".join([a.get("name") for a in (t.get("artists") or [])]),
            "popularity": t.get("popularity"),
            "duration_ms": t.get("duration_ms"),
        } for t in items])

    return (
        top_artists("short_term", 10),
        top_artists("medium_term", 10),
        top_artists("long_term", 50),
        top_tracks("long_term", 50)
    )
# ========================================================


# ================== Main ==================
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
        sp = spotify_client()
        user = sp.current_user()
    except Exception as e:
        print(f"Failed to authenticate with Spotify: {e}")
        print("Please check your Spotify credentials in the .env file")
        return

    # 1) Read playlists and summarize (skip if none)
    pids = read_playlist_ids(PLAYLIST_URLS_FILE)
    if pids:
        pl_top_artists, pl_top_genres = summarize_playlists(sp, pids)
    else:
        pl_top_artists, pl_top_genres = [], []

    # 2) Get top artists & tracks
    ta_4w, ta_6m, ta_all, tt_all = get_top_dfs(sp)

    # 3) Summarize top albums (from top 50 long-term tracks)
    tt_all["release_date"] = pd.to_datetime(tt_all["release_date"], errors="coerce")
    top_albums = tt_all["album"].value_counts().head(30).to_dict()

    # convert release_date to string for JSON safety
    tt_all["release_date"] = tt_all["release_date"].dt.date.astype(str)

    # 4) Top genres all-time (from top 50 long-term artists)
    genre_counter = {}
    for g in ta_all["genres"].dropna():
        for genre in [x.strip() for x in g.split(",") if x.strip()]:
            genre_counter[genre] = genre_counter.get(genre, 0) + 1
    top_genres_all_time = [{"genre": g, "count": c} for g, c in sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)[:50]]

    # 5) Eras from long-term tracks
    years = pd.to_datetime(tt_all["release_date"], errors="coerce").dropna().astype("datetime64[ns]").dt.year.value_counts().sort_index()
    eras = {
        "years": [{"year": int(y), "count": int(c)} for y, c in years.items()],
        "decades": [{"decade": f"{(y//10)*10}s", "count": int(c)} for y, c in years.groupby(years//10*10).sum().items()]
    }

    # 6) Build JSON
    payload = {
        "user": {
            "id": user.get("id"),
            "name": user.get("display_name"),
            "country": user.get("country"),
            "product": user.get("product")
        },
        "top_artists": {
            "short_term": ta_4w.to_dict(orient="records"),
            "medium_term": ta_6m.to_dict(orient="records"),
            "long_term": ta_all.to_dict(orient="records")
        },
        "top_tracks": {
            "long_term": tt_all.to_dict(orient="records")
        },
        "top_albums": [{"album": k, "count": v} for k, v in top_albums.items()],
        "top_genres_all_time": top_genres_all_time,
        "playlists_summary": {
            "top_artists": pl_top_artists,
            "top_genres": pl_top_genres
        },
        "eras": eras
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"JSON saved to {OUTPUT_JSON_PATH}")
    if not pids:
        print("Note: No playlists were provided; playlists_summary is empty.")


if __name__ == "__main__":
    main()
