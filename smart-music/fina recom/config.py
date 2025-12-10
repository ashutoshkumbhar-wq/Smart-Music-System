import os
from dotenv import load_dotenv

# Load environment variables from fina recom/.env explicitly (robust on Windows)
_DOTENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
# Force override so .env values win over empty or stale system env vars
load_dotenv(_DOTENV_PATH, override=True)

class Config:
    """Configuration class for the Fina Recom Music Recommendation System"""
    
    # Spotify API credentials
    # Workaround for UTF-8 BOM on first line of .env on Windows
    _cid = os.environ.get('SPOTIPY_CLIENT_ID')
    if not _cid:
        _cid_bom = os.environ.get('\ufeffSPOTIPY_CLIENT_ID')
        if _cid_bom:
            os.environ['SPOTIPY_CLIENT_ID'] = _cid_bom
            _cid = _cid_bom
    SPOTIPY_CLIENT_ID = _cid
    SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')
    SPOTIPY_REDIRECT_URI = os.environ.get('SPOTIPY_REDIRECT_URI', 'http://127.0.0.1:8888/callback')
    
    # Clean up any potential formatting issues from environment variables
    if SPOTIPY_CLIENT_ID and '=' in SPOTIPY_CLIENT_ID:
        SPOTIPY_CLIENT_ID = SPOTIPY_CLIENT_ID.split('=')[-1]
    if SPOTIPY_CLIENT_SECRET and '=' in SPOTIPY_CLIENT_SECRET:
        SPOTIPY_CLIENT_SECRET = SPOTIPY_CLIENT_SECRET.split('=')[-1]
    if SPOTIPY_REDIRECT_URI and '=' in SPOTIPY_REDIRECT_URI:
        SPOTIPY_REDIRECT_URI = SPOTIPY_REDIRECT_URI.split('=')[-1]
    
    # Market and scopes
    SPOTIFY_MARKET = os.environ.get('SPOTIFY_MARKET', 'IN')
    SPOTIFY_SCOPES = os.environ.get(
        'SPOTIFY_SCOPES',
        'playlist-read-private playlist-read-collaborative user-top-read user-modify-playback-state user-read-playback-state user-read-currently-playing'
    )
    
    # Cache path for OAuth tokens
    SPOTIFY_CACHE_PATH = os.environ.get('SPOTIFY_CACHE_PATH', '.cache_spotify_export')
    
    # Recommendation system settings
    INITIAL_COUNT = int(os.environ.get('INITIAL_COUNT', '50'))
    APPEND_EVERY_N = int(os.environ.get('APPEND_EVERY_N', '15'))
    APPEND_BATCH = int(os.environ.get('APPEND_BATCH', '20'))
    
    # Tunables for selection windows
    TOP_N_BOOST = int(os.environ.get('TOP_N_BOOST', '10'))
    ALBUM_TOP_N = int(os.environ.get('ALBUM_TOP_N', '3'))
    RELATED_MAX = int(os.environ.get('RELATED_MAX', '5'))
    
    # File paths
    PLAYLIST_URLS_FILE = os.path.join(os.path.dirname(__file__), "playlist_urls.txt")
    OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "metadata.json")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.SPOTIPY_CLIENT_ID:
            errors.append("SPOTIPY_CLIENT_ID is required")
        
        if not cls.SPOTIPY_CLIENT_SECRET:
            errors.append("SPOTIPY_CLIENT_SECRET is required")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)"""
        print("Fina Recom Configuration:")
        print(f"   Market: {cls.SPOTIFY_MARKET}")
        print(f"   Redirect URI: {cls.SPOTIPY_REDIRECT_URI}")
        print(f"   Spotify Client ID: {'Set' if cls.SPOTIPY_CLIENT_ID else 'Not Set'}")
        print(f"   Spotify Client Secret: {'Set' if cls.SPOTIPY_CLIENT_SECRET else 'Not Set'}")
        print(f"   Initial Count: {cls.INITIAL_COUNT}")
        print(f"   Append Every N: {cls.APPEND_EVERY_N}")
        print(f"   Append Batch: {cls.APPEND_BATCH}")
        print(f"   Playlist URLs File: {cls.PLAYLIST_URLS_FILE}")
        print(f"   Output JSON Path: {cls.OUTPUT_JSON_PATH}")

# Create a .env template if it doesn't exist
def create_env_template():
    """Create a .env template file"""
    env_template = """# Fina Recom Music Recommendation System Environment Variables
# Copy this file to .env and fill in your values

# Spotify API Credentials
SPOTIPY_CLIENT_ID=your_spotify_client_id_here
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback

# Spotify Settings
SPOTIFY_MARKET=IN
SPOTIFY_SCOPES=playlist-read-private playlist-read-collaborative user-top-read user-modify-playback-state user-read-playback-state user-read-currently-playing
SPOTIFY_CACHE_PATH=.cache_spotify_export

# Recommendation System Settings
INITIAL_COUNT=50
APPEND_EVERY_N=15
APPEND_BATCH=20

# Selection Windows
TOP_N_BOOST=10
ALBUM_TOP_N=3
RELATED_MAX=5
"""
    
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(env_template)
        print(f"Created .env template at {env_path}")
        print("   Please edit this file with your actual values")

if __name__ == "__main__":
    create_env_template()
    Config.print_config()
    
    errors = Config.validate()
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("\nConfiguration is valid")
