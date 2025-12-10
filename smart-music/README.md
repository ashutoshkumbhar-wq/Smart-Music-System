# Smart Music

An end-to-end gesture-driven Spotify companion that blends live playback control, auto-DJ queues, mood-based queues, and personal recommendation tooling. The project consists of:

- **Backend (Flask)**: gesture recognition API, Spotify OAuth/session manager, Artist Mix/Mood Mixer/Fina Recom endpoints, and auto-artgig queue logic.
- **Frontend (vanilla JS + HTML/CSS)**: landing/home experience, profile/auth dashboard, artist mix studio, mood mixer UI, and gesture camera surface.
- **Models**: MediaPipe gesture models, Artgig auto-DJ helpers, mood mixer recipes, Fina Recom CLI tooling.

---

## Features

### 1. Spotify Bootstrap + Profile Generation
- One-click Connect button triggers the backend-managed OAuth flow (`/api/spotify/login` + `/callback`).
- Backend auto-syncs `.cache-dj-session` → `fina recom/.cache_spotify_export` and runs `fina recom/combocode.py`.
- `/api/setup/bootstrap` and `/api/fina-recom/generate-profile` expose the same flow via API.

### 2. Gesture + Emotion Services
- `/api/gesture/predict` and `/api/detect/all` run MediaPipe hands + FER (if installed) with configurable thresholds.
- Unified gesture controller for the frontend to send play/pause/next/volume events.

### 3. Artist Mix & Auto Artgig Top-ups
- `POST /api/artist-mix/search` queues at least 50 tracks per artist (auto-fall back to albums if needed).
- `POST /api/artist-mix/play-multiple` enforces ≥50 tracks per artist whenever no genre filter is used.
- Both endpoints automatically start the `Models/artgig.py` pump loop in “monitor mode”—the loop watches the current queue and injects `topup_batch` tracks after `changes_per_topup` song changes (defaults 50/10). Tags follow the selected profile; without a profile, it runs “original mix” mode with dedupe only.

### 4. Manual Artgig Sessions
- `POST /api/artgig/start` launches the classic artgig pump loop (seed queue + auto top-ups) with per-request overrides for:
  - `initial_batch`
  - `topup_batch`
  - `changes_per_topup`
- Works for both random discovery and artist-driven sessions.

### 5. Mood Mixer & DJ Controls
- `POST /api/mood-mixer/start` seeds + tops up a Spotify queue based on weighted moods and language filters (HI/EN/Mix).
- DJ session endpoints, orchestrator service, and unified gesture control integrate with Spotify playback.

### 6. Fina Recom Toolkit
- `/api/fina-recom/status` / `.../profile` surface the metadata produced by `fina recom/combocode.py`.
- `/api/fina-recom/start-recommendations` spawns `newmodel.py` in the selected mode (Comfort / Balanced / Explorer).

---

## Project Layout

```
smart-music/
├── backend/
├── frontend/
├── Models/
├── Gesture final/
├── fina recom/
├── G-E/
├── spotify-backend/
└── tmp_proto/
```

### Folder Walkthrough

| Folder | Highlights |
|--------|------------|
| `backend/` | **Primary control plane.** `app.py` exposes every REST endpoint (Spotify OAuth, bootstrap, artist/mood mix, gesture APIs, orchestrator control, Fina Recom automation). `config.py` loads `.env`, and `generate_token.py` lets you run the original OAuth helper if desired. `.cache-dj-session` is the canonical token cache that gets synced into other modules. |
| `frontend/` | **All client experiences.** `index.html` is the hero page; `profile.html/js` manages Connect with Spotify and triggers `/api/setup/bootstrap`; `Cards/artist` hosts the Style Radio studio; `mood2` is Mood Radio; `fina` shows the recommendation profile; `camera.html/js` demos gestures. Shared assets/scripts live alongside. |
| `Models/` | **Runtime engines imported by the backend.** `artgig.py` implements the auto-DJ pump loop (seed queue + top-ups or “monitor mode” when a queue already exists). `mood_mixer.py` computes mood-weighted batches and keeps topping up the Spotify queue. |
| `Gesture final/` | **Gesture R&D lab.** Contains the MediaPipe training scripts (`collect_gestures.py`, `train_model_strong.py`), testing harnesses (`testing.py`, `maintesting_spotify.py`), and trained artifacts (`gesture_model*.pkl`, `scaler*.pkl`). The backend loads these pickles on startup. |
| `G-E/` | **Legacy integrated gesture engines.** `integrated1 copy 3/` bundles older stand-alone Python apps (`integrated1.py`, `zorchestrator_live.py`) that wired webcam gestures straight to Spotify. Useful for regression tests or understanding how current logic evolved. |
| `fina recom/` | **Profile + recommendation toolkit.** `combocode.py` exports playlists/top artists/eras into `metadata.json`; `newmodel.py` runs the Comfort/Balanced/Explorer CLI; `config.py` defines markets/scopes. The backend now runs these scripts automatically (no manual `.env` needed). `.cache_spotify_export` holds the Spotipy cache specific to these scripts. |
| `spotify-backend/` | A legacy Node-based backend prototype retained for posterity. Not part of the current stack but handy as reference. |
| `tmp_proto/` | Holds a pinned protobuf wheel and extracted `runtime_version.py` used while debugging the Mediapipe/TensorFlow dependency chain. Safe to delete when not required. |

Each folder intentionally stays in the repo so documentation and automation reference the exact files the backend/ frontend use.

---

### How the Major Components Interact

#### Backend (Flask)
1. **OAuth & Bootstrap**: `/api/spotify/login` redirects to Spotify, `/callback` stores tokens in `.cache-dj-session`, syncs Fina Recom cache, and can immediately run `combocode.py`.
2. **Artist Mix / Auto DJ**: When the frontend hits `/api/artist-mix/search` or `play-multiple`, the backend queues ≥50 tracks and launches `Models/artgig.pump_loop` in monitor mode so the queue stays full (default: add 50 tracks every 10 song changes). `/api/artgig/start` exposes the classic artgig session with seed/top-up overrides.
3. **Mood Mixer**: `/api/mood-mixer/start` passes mood weights + language to `Models/mood_mixer`, which seeds a queue and keeps adding batches to match the weight distribution.
4. **Gestures**: Loads the pickled model/scaler from `Gesture final/` and exposes `/api/gesture/*`. Gesture events can trigger Spotify actions or feed into the orchestrator.
5. **Fina Recom**: `/api/setup/bootstrap` / `/api/fina-recom/generate-profile` run `combocode.py`; `/api/fina-recom/profile` serves `metadata.json`; `/api/fina-recom/start-recommendations` spawns `newmodel.py` and feeds it the Comfort/Balanced/Explorer selection.

#### Frontend
- Uses fetch calls (default `http://localhost:3000`) to hit the endpoints above:
  - **Profile page** triggers the OAuth bootstrap and shows session status.
  - **Style Radio** calls artist mix endpoints and renders the track list while the backend handles auto top-ups.
  - **Mood Radio** sends mood weights, then relies on backend queue monitoring.
  - **Recommendation Model** polls status and pulls `metadata.json` for the analytics dashboard.
  - **Gesture Camera** streams frames for classification/testing.

#### Models, Gesture final, G-E
- `Gesture final/` is where you collect/train/test new gesture models. When you produce new `gesture_model.pkl` / `scaler.pkl`, simply replace the files and restart the backend.
- `Models/artgig.py` and `mood_mixer.py` are not stand-alone apps; they’re imported by the backend so the same logic is used whether a mix is started automatically or manually.
- `G-E/` stays around for standalone demos—handy if you want to run gestures without the Flask backend.

#### Fina Recom
- Even though the backend automates the export, you can still run `python combocode.py` or `python newmodel.py` manually if you want CLI control. The shared token cache means the backend and CLI use the same Spotify session.

This architecture lets you tweak any area (gesture model, mood recipes, artgig thresholds, recommendation heuristics) without touching the others, while the README and automation stay aligned with the actual folder structure.

---

## Prerequisites

- Python 3.11+ (backend, models, Fina Recom tooling)
- Node/npm (optional if you want to serve frontend with something other than Live Server)
- Spotify Developer account with a registered app:
  - Redirect URI: `http://127.0.0.1:3000/callback`
  - Frontend redirect (for the profile page): `http://127.0.0.1:5500/frontend/profile.html`

---

## Quick Start

### 1. Environment Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create `backend/.env` (copy from the template) with:

```
SPOTIPY_CLIENT_ID=...
SPOTIPY_CLIENT_SECRET=...
SPOTIPY_REDIRECT_URI=http://127.0.0.1:3000/callback
FRONTEND_REDIRECT_URI=http://127.0.0.1:5500/frontend/profile.html
```

### 2. Run the Backend

```bash
cd backend
python app.py
```

### 3. Serve the Frontend

Use VS Code Live Server, `python -m http.server`, or your favorite static server pointing to the `frontend/` folder (port 5500 expected by default).

### 4. Authenticate with Spotify

1. Open `http://127.0.0.1:5500/frontend/profile.html`.
2. Click **Connect with Spotify**.
3. Approve the OAuth prompt. The backend receives the code, caches tokens, and runs the bootstrap (copy cache + combocode) automatically.

### 5. Explore Features

- **Artist Mix Studio** (`frontend/Cards/artist/index.html`):
  - Add artists, optionally choose a remix/lofi/mashup/slowed tag profile, click Play.
  - Backend queues ≥50 tracks per artist and automatically starts the artgig top-up loop.
- **Mood Mixer** (`frontend/mood2/index.html`):
  - Choose language + mood weights; backend seeds the queue (`Models/mood_mixer.py`).
- **Gesture Control** (`frontend/camera.html`):
  - Capture frames or stream gestures; backend handles recognition and playback actions.
- **Fina Recom** (`frontend/fina/index.html`):
  - Generates metadata via `/api/setup/bootstrap`; view summaries or start recommendation modes.

---

## Key API Endpoints (Backend)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gesture/predict` | POST | Predict a gesture from a base64 image. |
| `/api/detect/all` | POST | Combined gesture + emotion detection. |
| `/api/spotify/status` | GET | Check auth state + devices. |
| `/api/spotify/login` | GET | Returns Spotify auth URL. |
| `/api/setup/bootstrap` | POST | Runs cache sync + combocode. |
| `/api/artist-mix/search` | POST | Queue ≥50 tracks for one artist (auto top-up). |
| `/api/artist-mix/play-multiple` | POST | Queue ≥50 per artist (auto top-up). |
| `/api/artgig/start` | POST | Manual artgig session with custom top-up thresholds. |
| `/api/mood-mixer/start` | POST | Language + mood weighted queue builder. |
| `/api/fina-recom/*` | GET/POST | Status, generate profile, run recommender, fetch metadata. |
| `/api/orchestrator/*` | REST | Start/stop status for the playback orchestrator service. |

Each endpoint returns structured JSON (`ok`, `message`, etc.). Auto artgig responses include `auto_topup_started` and the parameters being used.

---

## Auto DJ (Artgig) Behavior

- Works whether you click “Play Mix” or start a dedicated DJ session.
- Keeps track of played URIs and deduplicates new queue entries.
- Supports both tag-enforced searches (Remix/Lofi/etc.) and tag-less “original mix” sessions.
- Parameters:
  - `initial_batch`: number of tracks seeded on session start (ignored when attached to an existing queue).
  - `topup_batch`: number of tracks to add after every `changes_per_topup` song changes.
  - `changes_per_topup`: how many track changes before queue refill.

Defaults are 50/50/10 but can be changed per request.

---

## Development Tips

- Gesture FER requires `tensorflow` + `fer`. If you don’t need facial emotion detection, you can omit those installs; the backend logs a warning but continues with gestures only.
- Keep Spotify open on a device; many endpoints require an active player to queue tracks.
- `tmp_proto/` contains a pinned protobuf wheel used for inspection; not required at runtime.

---

## License

Project code is owned by the Smart Music authors. Refer to Spotify’s developer TOS before distributing.


