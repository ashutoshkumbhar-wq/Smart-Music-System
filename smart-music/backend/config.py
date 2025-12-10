import os
from dotenv import load_dotenv

# Force reload .env every time
_DOTENV_PATH = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(_DOTENV_PATH, override=True)  # override=True is CRITICAL

class Config:
    """Configuration class for the Fina Recom Music Recommendation System"""
    
    # Spotify API credentials - with BOM handling
    _cid = os.environ.get('SPOTIPY_CLIENT_ID')
    if not _cid:
        _cid_bom = os.environ.get('\ufeffSPOTIPY_CLIENT_ID')
        if _cid_bom:
            os.environ['SPOTIPY_CLIENT_ID'] = _cid_bom
            _cid = _cid_bom
    
    SPOTIPY_CLIENT_ID = _cid
    SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')
    SPOTIPY_REDIRECT_URI = os.environ.get('SPOTIPY_REDIRECT_URI', 'http://127.0.0.1:3000/callback')
    
    # Clean up any potential formatting issues
    if SPOTIPY_CLIENT_ID and '=' in SPOTIPY_CLIENT_ID:
        SPOTIPY_CLIENT_ID = SPOTIPY_CLIENT_ID.split('=')[-1].strip()
    if SPOTIPY_CLIENT_SECRET and '=' in SPOTIPY_CLIENT_SECRET:
        SPOTIPY_CLIENT_SECRET = SPOTIPY_CLIENT_SECRET.split('=')[-1].strip()
    if SPOTIPY_REDIRECT_URI and '=' in SPOTIPY_REDIRECT_URI:
        SPOTIPY_REDIRECT_URI = SPOTIPY_REDIRECT_URI.split('=')[-1].strip()
    
    # Rest of config...
    SPOTIFY_MARKET = os.environ.get('SPOTIFY_MARKET', 'IN')
    SPOTIFY_SCOPES = os.environ.get(
        'SPOTIFY_SCOPES',
        'playlist-read-private playlist-read-collaborative user-top-read'
    )
    SPOTIFY_CACHE_PATH = os.environ.get('SPOTIFY_CACHE_PATH', '.cache_spotify_export')
    FRONTEND_REDIRECT_URI = os.environ.get(
        'FRONTEND_REDIRECT_URI',
        'http://127.0.0.1:5500/profile.html'  # Updated: assumes server root is at frontend/
    )
    
    # Flask server configuration
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 3000))  # Default to 3000 to match frontend expectations
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Gesture recognition settings
    GESTURE_CONFIDENCE_THRESHOLD = float(os.environ.get('GESTURE_CONFIDENCE_THRESHOLD', '0.3'))  # Lowered from 0.8 to 0.3 for better detection
    GESTURE_STABLE_FRAMES = int(os.environ.get('GESTURE_STABLE_FRAMES', '3'))  # Reduced from 5 to 3 for faster detection
    GESTURE_ACTION_COOLDOWN = float(os.environ.get('GESTURE_ACTION_COOLDOWN', '0.3'))  # Reduced from 1.0 to 0.3 for faster response
    
    # DJ settings
    DJ_DEFAULT_BATCH_SIZE = int(os.environ.get('DJ_DEFAULT_BATCH_SIZE', '150'))
    DJ_STRICT_PRIMARY = int(os.environ.get('DJ_STRICT_PRIMARY', '1'))
    
    # CORS origins
    CORS_ORIGINS = os.environ.get(
        'CORS_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000,http://127.0.0.1:5000,http://localhost:5500,http://127.0.0.1:5500'
    ).split(',')
    
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
        print("🔧 Smart Music Backend Configuration:")
        print(f"   Host: {cls.HOST}")
        print(f"   Port: {cls.PORT}")
        print(f"   Debug: {cls.DEBUG}")
        print(f"   Spotify Client ID: {'Set' if cls.SPOTIPY_CLIENT_ID else 'Not Set'}")
        print(f"   Spotify Client Secret: {'Set' if cls.SPOTIPY_CLIENT_SECRET else 'Not Set'}")
        print(f"   Gesture Confidence Threshold: {cls.GESTURE_CONFIDENCE_THRESHOLD}")
        print(f"   DJ Default Batch Size: {cls.DJ_DEFAULT_BATCH_SIZE}")
        print(f"   CORS Origins: {cls.CORS_ORIGINS}")

# Create a .env template if it doesn't exist
def create_env_template():
    """Create a .env template file"""
    env_template = """# Smart Music Backend Environment Variables
# Copy this file to .env and fill in your values

# Flask Settings
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
HOST=0.0.0.0
PORT=5000

# Spotify API Credentials
SPOTIPY_CLIENT_ID=your_spotify_client_id_here
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIPY_REDIRECT_URI=http://localhost:5000/callback
FRONTEND_REDIRECT_URI=http://127.0.0.1:5500/profile.html

# Gesture Recognition Settings
GESTURE_CONFIDENCE_THRESHOLD=0.8
GESTURE_STABLE_FRAMES=3  # Reduced from 5 to 3 for faster detection
GESTURE_ACTION_COOLDOWN=0.3  # Reduced from 1.0 to 0.3 seconds for faster response

# DJ Settings
DJ_DEFAULT_BATCH_SIZE=150
DJ_STRICT_PRIMARY=1

# Model Paths (relative to backend directory)
GESTURE_MODEL_PATH=../Gesture final/gesture_model.pkl
GESTURE_SCALER_PATH=../Gesture final/scaler.pkl

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5000
"""
    
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(env_template)
        print(f"📝 Created .env template at {env_path}")
        print("   Please edit this file with your actual values")

if __name__ == "__main__":
    create_env_template()
    Config.print_config()
    
    errors = Config.validate()
    if errors:
        print("\n❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("\n✅ Configuration is valid")
