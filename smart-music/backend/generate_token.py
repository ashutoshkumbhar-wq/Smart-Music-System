#!/usr/bin/env python3
"""
Standalone script to generate Spotify access token
This script helps you get an access_token through the OAuth flow
"""

import os
import sys
import json
import time
from urllib.parse import urlparse, parse_qs
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import Config
except ImportError:
    print("⚠️  Warning: config.py not found, using environment variables")
    class Config:
        SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID')
        SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')
        SPOTIPY_REDIRECT_URI = os.environ.get('SPOTIPY_REDIRECT_URI', 'http://localhost:8888/callback')

import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Configuration
CLIENT_ID = Config.SPOTIPY_CLIENT_ID
CLIENT_SECRET = Config.SPOTIPY_CLIENT_SECRET
REDIRECT_URI = Config.SPOTIPY_REDIRECT_URI or 'http://localhost:8888/callback'
SCOPES = os.environ.get(
    'SPOTIFY_SCOPES',
    'user-modify-playback-state user-read-playback-state user-read-currently-playing user-library-modify'
)
CACHE_PATH = os.environ.get('SPOTIFY_CACHE_PATH', '.cache-dj-session')

# Global variable to store the authorization code
auth_code = None

class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP server to handle OAuth callback"""
    
    def do_GET(self):
        global auth_code
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        
        if 'code' in query_params:
            auth_code = query_params['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_content = """
            <html>
            <head><title>Authorization Successful</title></head>
            <body>
                <h1>✅ Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                <script>setTimeout(function(){window.close();}, 2000);</script>
            </body>
            </html>
            """
            self.wfile.write(html_content.encode('utf-8'))
        elif 'error' in query_params:
            error = query_params['error'][0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html_content = f"""
            <html>
            <head><title>Authorization Failed</title></head>
            <body>
                <h1>❌ Authorization Failed</h1>
                <p>Error: {error}</p>
            </body>
            </html>
            """
            self.wfile.write(html_content.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress server logs
        pass

def start_callback_server(port=8888):
    """Start a local HTTP server to receive the OAuth callback"""
    server = HTTPServer(('localhost', port), CallbackHandler)
    return server

def generate_token_interactive():
    """Interactive method to generate token"""
    print("=" * 60)
    print("🎵 Spotify Access Token Generator")
    print("=" * 60)
    print()
    
    # Check credentials
    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ Error: Spotify credentials not configured!")
        print("   Please set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET")
        print("   in your .env file or environment variables.")
        return None
    
    print(f"✅ Client ID: {CLIENT_ID[:10]}...")
    print(f"✅ Redirect URI: {REDIRECT_URI}")
    print(f"✅ Scopes: {SCOPES}")
    print()
    
    # Create OAuth manager
    oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=CACHE_PATH,
        open_browser=False  # We'll handle browser opening manually
    )
    
    # Check if token already exists
    cached_token = oauth.get_cached_token()
    if cached_token:
        print("✅ Found cached token!")
        expires_at = cached_token.get('expires_at', 0)
        current_time = int(time.time())
        
        if current_time < expires_at:
            print(f"✅ Token is valid (expires in {expires_at - current_time} seconds)")
            print()
            print("Token Information:")
            print(f"  Access Token: {cached_token.get('access_token', '')[:50]}...")
            print(f"  Expires At: {time.ctime(expires_at)}")
            print(f"  Scopes: {cached_token.get('scope', 'N/A')}")
            print()
            
            response = input("Use existing token? (y/n): ").strip().lower()
            if response == 'y':
                return cached_token
        else:
            print("⚠️  Cached token is expired, will refresh...")
    
    # Get authorization URL
    print("🔗 Generating authorization URL...")
    auth_url = oauth.get_authorize_url()
    
    print()
    print("📋 Please authorize the application:")
    print(f"   {auth_url}")
    print()
    
    # Extract port from redirect URI
    parsed_uri = urlparse(REDIRECT_URI)
    port = parsed_uri.port or 8888
    
    # Start callback server
    print(f"🌐 Starting callback server on port {port}...")
    server = start_callback_server(port)
    
    # Open browser
    print("🌍 Opening browser...")
    try:
        webbrowser.open(auth_url)
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"   Please open this URL manually: {auth_url}")
    
    # Wait for callback
    print("⏳ Waiting for authorization...")
    print("   (This will timeout after 120 seconds)")
    
    server.timeout = 120
    server.handle_request()
    
    if not auth_code:
        print("❌ No authorization code received. Please try again.")
        return None
    
    print("✅ Authorization code received!")
    print("🔄 Exchanging code for tokens...")
    
    # Exchange code for token
    try:
        token_info = oauth.get_access_token(auth_code)
        
        if token_info:
            print()
            print("=" * 60)
            print("✅ SUCCESS! Token generated and saved!")
            print("=" * 60)
            print()
            print("Token Information:")
            print(f"  Access Token: {token_info.get('access_token', '')[:50]}...")
            print(f"  Token Type: {token_info.get('token_type', 'N/A')}")
            print(f"  Expires In: {token_info.get('expires_in', 0)} seconds")
            print(f"  Expires At: {time.ctime(token_info.get('expires_at', 0))}")
            print(f"  Scopes: {token_info.get('scope', 'N/A')}")
            print(f"  Refresh Token: {'Yes' if token_info.get('refresh_token') else 'No'}")
            print()
            print(f"💾 Token saved to: {os.path.abspath(CACHE_PATH)}")
            print()
            
            return token_info
        else:
            print("❌ Failed to get token information")
            return None
            
    except Exception as e:
        print(f"❌ Error exchanging code for token: {e}")
        return None

def generate_token_from_code(auth_code):
    """Generate token from an existing authorization code"""
    oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=CACHE_PATH
    )
    
    try:
        token_info = oauth.get_access_token(auth_code)
        print("✅ Token generated successfully!")
        return token_info
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def refresh_existing_token():
    """Refresh an existing token using refresh_token"""
    oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        cache_path=CACHE_PATH
    )
    
    cached_token = oauth.get_cached_token()
    if cached_token:
        print("✅ Token refreshed successfully!")
        return cached_token
    else:
        print("❌ No cached token found to refresh")
        return None

def print_token_info(token_info):
    """Print token information in a readable format"""
    if not token_info:
        print("❌ No token information available")
        return
    
    print()
    print("=" * 60)
    print("📋 Token Information")
    print("=" * 60)
    print(f"Access Token: {token_info.get('access_token', 'N/A')}")
    print(f"Token Type: {token_info.get('token_type', 'N/A')}")
    print(f"Expires In: {token_info.get('expires_in', 0)} seconds")
    print(f"Expires At: {time.ctime(token_info.get('expires_at', 0))}")
    print(f"Scopes: {token_info.get('scope', 'N/A')}")
    if token_info.get('refresh_token'):
        print(f"Refresh Token: {token_info.get('refresh_token')[:50]}...")
    print("=" * 60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Spotify access token')
    parser.add_argument('--code', type=str, help='Authorization code (if you already have it)')
    parser.add_argument('--refresh', action='store_true', help='Refresh existing token')
    parser.add_argument('--info', action='store_true', help='Show current token info')
    parser.add_argument('--output', type=str, help='Output file to save token JSON')
    
    args = parser.parse_args()
    
    if args.info:
        # Just show current token info
        oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope=SCOPES,
            cache_path=CACHE_PATH
        )
        token_info = oauth.get_cached_token()
        print_token_info(token_info)
        if args.output and token_info:
            with open(args.output, 'w') as f:
                json.dump(token_info, f, indent=2)
            print(f"\n💾 Token saved to: {args.output}")
        return
    
    if args.refresh:
        # Refresh existing token
        token_info = refresh_existing_token()
        if args.output and token_info:
            with open(args.output, 'w') as f:
                json.dump(token_info, f, indent=2)
            print(f"\n💾 Token saved to: {args.output}")
        return
    
    if args.code:
        # Use provided authorization code
        token_info = generate_token_from_code(args.code)
    else:
        # Interactive flow
        token_info = generate_token_interactive()
    
    # Save to output file if specified
    if args.output and token_info:
        with open(args.output, 'w') as f:
            json.dump(token_info, f, indent=2)
        print(f"\n💾 Token saved to: {args.output}")
    
    # Print token info
    if token_info:
        print_token_info(token_info)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

