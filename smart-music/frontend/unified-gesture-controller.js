/**
 * Unified Gesture Controller for Smart Music
 * Integrates existing camera-based gestures with touch/mouse gestures
 */

class UnifiedGestureController {
    constructor(options = {}) {
        this.options = {
            backendUrl: options.backendUrl || 'http://localhost:3000',
            enableTouchGestures: options.enableTouchGestures !== false,
            enableCameraGestures: options.enableCameraGestures || false,
            gestureThreshold: options.gestureThreshold || 0.3,
            cooldownMs: options.cooldownMs || 300, // Reduced from 1000ms to 300ms for faster response
            ...options
        };
        
        this.isAuthenticated = false;
        this.currentTrack = null;
        this.lastGestureTime = 0;
        this.hammer = null;
        this.cameraStream = null;
        this.gestureRecognitionInterval = null;
        
        // Your exact gesture mappings from maintesting_spotify.py
        this.GESTURE_TO_ACTION = {
            // Right hand gestures (from your trained model)
            'play_right':       { action: 'play' },
            'pause_right':      { action: 'pause' },
            'next_right':       { action: 'next' },
            'previous_right':   { action: 'previous' },
            // Left hand gestures (from your trained model)
            'volume_up_left':   { action: 'volume', delta: +10 },
            'volume_down_left': { action: 'volume', delta: -10 },
            'like_left':        { action: 'like' },
            'skip30_left':      { action: 'seek',   delta: +30000 },
            // Touch gesture mappings (additional)
            'swipe_left':       { action: 'previous' },
            'swipe_right':      { action: 'next' },
            'swipe_up':         { action: 'play_pause' },
            'swipe_down':       { action: 'volume_down' },
            'tap':              { action: 'play_pause' },
            'double_tap':       { action: 'next' }
        };
        
        this.init();
    }
    
    async init() {
        console.log('🎭 Initializing Unified Gesture Controller...');
        
        // Check Spotify authentication
        await this.checkSpotifyAuth();
        
        // Initialize touch/mouse gestures
        if (this.options.enableTouchGestures) {
            this.initTouchGestures();
        }
        
        // Initialize camera gestures
        if (this.options.enableCameraGestures) {
            await this.initCameraGestures();
        }
        
        // Start periodic track updates
        this.startTrackUpdates();
        
        console.log('✅ Unified Gesture Controller initialized');
    }
    
    async checkSpotifyAuth() {
        try {
            const response = await fetch(`${this.options.backendUrl}/api/spotify/status`);
            const data = await response.json();
            this.isAuthenticated = data.authenticated;
            
            if (this.isAuthenticated) {
                console.log('🎵 Spotify authenticated:', data.user?.name || data.user?.id);
                this.updateAuthUI(true, data.user);
            } else {
                console.log('❌ Spotify not authenticated');
                this.updateAuthUI(false);
            }
        } catch (error) {
            console.error('Failed to check Spotify auth:', error);
            this.updateAuthUI(false);
        }
    }
    
    updateAuthUI(authenticated, user = null) {
        const authStatus = document.getElementById('auth-status');
        if (authStatus) {
            if (authenticated) {
                authStatus.innerHTML = `✅ Connected to Spotify${user ? ` as ${user.name || user.id}` : ''}`;
                authStatus.className = 'auth-status connected';
            } else {
                authStatus.innerHTML = '❌ Not connected to Spotify';
                authStatus.className = 'auth-status disconnected';
            }
        }
    }
    
    initTouchGestures() {
        // Initialize Hammer.js for touch/mouse gestures
        if (typeof Hammer === 'undefined') {
            console.error('Hammer.js not loaded. Please include Hammer.js library.');
            return;
        }
        
        console.log('🔧 Initializing Hammer.js...');
        const container = document.body;
        
        // Create Hammer instance - use Manager for better control
        this.hammer = new Hammer.Manager(container, {
            touchAction: 'pan-y', // Allow vertical scrolling but capture gestures
        });
        
        // Add recognizers
        const swipe = new Hammer.Swipe({ 
            direction: Hammer.DIRECTION_ALL, 
            threshold: 10, 
            velocity: 0.3 
        });
        
        const tap = new Hammer.Tap({ 
            taps: 1, 
            interval: 250, 
            time: 250 
        });
        
        const doubletap = new Hammer.Tap({ 
            taps: 2, 
            interval: 300, 
            time: 300 
        });
        
        // Add recognizers to manager
        this.hammer.add(swipe);
        this.hammer.add(tap);
        this.hammer.add(doubletap);
        
        // Make doubletap require double tap (not single)
        tap.recognizeWith(doubletap);
        doubletap.requireFailure(tap);
        
        // Add gesture event listeners with debugging
        this.hammer.on('swipeleft', (e) => {
            if (e.srcEvent) e.srcEvent.stopPropagation(); // Prevent section click
            console.log('👈 Swipe left detected');
            this.handleGesture('swipe_left');
        });
        this.hammer.on('swiperight', (e) => {
            if (e.srcEvent) e.srcEvent.stopPropagation();
            console.log('👉 Swipe right detected');
            this.handleGesture('swipe_right');
        });
        this.hammer.on('swipeup', (e) => {
            if (e.srcEvent) e.srcEvent.stopPropagation();
            console.log('👆 Swipe up detected');
            this.handleGesture('swipe_up');
        });
        this.hammer.on('swipedown', (e) => {
            if (e.srcEvent) e.srcEvent.stopPropagation();
            console.log('👇 Swipe down detected');
            this.handleGesture('swipe_down');
        });
        this.hammer.on('tap', (e) => {
            // Only handle tap if it's not on a clickable element
            const target = e.target || (e.srcEvent && e.srcEvent.target);
            if (target && !target.closest('a, button, .clickable, .profile-link')) {
                if (e.srcEvent) e.srcEvent.stopPropagation();
                console.log('👆 Tap detected');
                this.handleGesture('tap');
            }
        });
        this.hammer.on('doubletap', (e) => {
            if (e.srcEvent) e.srcEvent.stopPropagation();
            console.log('👆👆 Double tap detected');
            this.handleGesture('double_tap');
        });
        
        console.log('👆 Touch gestures initialized');
    }
    
    async initCameraGestures() {
        try {
            // Request camera access
            this.cameraStream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 640, 
                    height: 480,
                    facingMode: 'user'
                } 
            });
            
            // Create video element for camera feed
            const video = document.createElement('video');
            video.srcObject = this.cameraStream;
            video.autoplay = true;
            video.muted = true;
            video.style.position = 'fixed';
            video.style.top = '10px';
            video.style.right = '10px';
            video.style.width = '200px';
            video.style.height = '150px';
            video.style.border = '2px solid #1db954';
            video.style.borderRadius = '10px';
            video.style.zIndex = '1000';
            video.style.display = 'none'; // Hidden by default
            
            document.body.appendChild(video);
            
            // Start gesture recognition
            this.startCameraGestureRecognition(video);
            
            console.log('📹 Camera gestures initialized');
        } catch (error) {
            console.error('Failed to initialize camera gestures:', error);
        }
    }
    
    startCameraGestureRecognition(video) {
        if (this.gestureRecognitionInterval) return;
        
        // Use requestAnimationFrame for better performance and reduce lag
        let lastFrameTime = 0;
        const frameInterval = 300; // Process every 500ms (2 times per second) for balanced gesture detection
        
        const processFrame = async (currentTime) => {
            if (currentTime - lastFrameTime >= frameInterval) {
                lastFrameTime = currentTime;
                await this.predictCameraGesture(video);
            }
            if (this.gestureRecognitionInterval) {
                requestAnimationFrame(processFrame);
            }
        };
        
        this.gestureRecognitionInterval = true; // Use boolean flag instead of interval ID
        requestAnimationFrame(processFrame);
        
        console.log('📹 Camera gesture recognition started (optimized)');
    }
    
    async predictCameraGesture(video) {
        if (!video.srcObject || !video.videoWidth || !video.videoHeight) return;
        
        try {
            // Reuse canvas to reduce memory allocation and improve performance
            if (!this.gestureCanvas) {
                this.gestureCanvas = document.createElement('canvas');
                this.gestureCanvas.width = 320; // Reduced resolution for faster processing
                this.gestureCanvas.height = 240;
                this.gestureCanvasCtx = this.gestureCanvas.getContext('2d');
            }
            
            // Draw video frame to canvas at reduced resolution
            this.gestureCanvasCtx.drawImage(
                video, 
                0, 0, 
                this.gestureCanvas.width, 
                this.gestureCanvas.height
            );
            
            // Convert to base64 with lower quality for faster processing
            const imageData = this.gestureCanvas.toDataURL('image/jpeg', 0.6); // Reduced quality
            
            // Send to backend for gesture recognition (non-blocking)
            fetch(`${this.options.backendUrl}/api/gesture/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            })
            .then(response => response.json())
            .then(result => {
                if (result.gesture && result.gesture !== 'none' && result.confidence >= this.options.gestureThreshold) {
                    console.log('📹 Camera gesture detected:', result.gesture, result.confidence);
                    this.handleGesture(result.gesture);
                }
            })
            .catch(error => {
                // Silently handle errors to avoid console spam
                if (error.message !== 'NetworkError') {
                    console.error('Camera gesture recognition error:', error);
                }
            });
        } catch (error) {
            // Only log non-network errors
            if (error.name !== 'NetworkError') {
                console.error('Camera gesture recognition error:', error);
            }
        }
    }
    
    async handleGesture(gesture) {
        console.log('🎭 Gesture handler called with:', gesture);
        
        // Dispatch event to notify other handlers that a gesture was detected
        document.dispatchEvent(new CustomEvent('gesture-detected', { detail: { gesture } }));
        
        // Check cooldown
        const now = Date.now();
        if (now - this.lastGestureTime < this.options.cooldownMs) {
            console.log('⏳ Gesture ignored (cooldown)');
            return;
        }
        
        if (!this.isAuthenticated) {
            console.log('❌ Gesture ignored (not authenticated)');
            return;
        }
        
        this.lastGestureTime = now;
        console.log('🎭 Gesture detected:', gesture);
        
        // Map gesture to action
        const action = this.GESTURE_TO_ACTION[gesture];
        if (!action) {
            console.log('❓ Unknown gesture:', gesture);
            return;
        }
        
        await this.executeSpotifyAction(action, gesture);
    }
    
    async executeSpotifyAction(action, gesture) {
        console.log('🎵 Executing Spotify action:', action);
        try {
            let endpoint = '/api/spotify/control';
            let body = { ...action };
            
            // Handle special actions
            if (action.action === 'play_pause') {
                // Toggle play/pause based on current state
                if (this.currentTrack && this.currentTrack.playback?.is_playing) {
                    body.action = 'pause';
                } else {
                    body.action = 'play';
                }
            } else if (action.action === 'volume_up') {
                body.action = 'volume';
                body.delta = 10;
            } else if (action.action === 'volume_down') {
                body.action = 'volume';
                body.delta = -10;
            }
            
            console.log('📤 Sending request to:', `${this.options.backendUrl}${endpoint}`);
            console.log('📤 Request body:', body);
            
            const response = await fetch(`${this.options.backendUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });
            
            const result = await response.json();
            console.log('📥 Response:', result);
            
            if (result.ok) {
                console.log('✅ Spotify action executed:', body.action);
                this.showGestureFeedback(gesture, true);
                
                // Update track info after action
                setTimeout(() => this.updateCurrentTrack(), 500);
            } else {
                console.error('❌ Spotify action failed:', result.error);
                this.showGestureFeedback(gesture, false);
            }
        } catch (error) {
            console.error('❌ Failed to execute Spotify action:', error);
            this.showGestureFeedback(gesture, false);
        }
    }
    
    showGestureFeedback(gesture, success) {
        // Create visual feedback
        const feedback = document.createElement('div');
        feedback.className = `gesture-feedback ${success ? 'success' : 'error'}`;
        feedback.textContent = `${success ? '✅' : '❌'} ${gesture}`;
        feedback.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: ${success ? '#1db954' : '#e22134'};
            color: white;
            padding: 15px 25px;
            border-radius: 30px;
            font-weight: bold;
            z-index: 10000;
            pointer-events: none;
            animation: gestureFeedback 1.5s ease-out forwards;
            box-shadow: 0 10px 30px rgba(29, 185, 84, 0.3);
        `;
        
        // Add animation keyframes
        if (!document.getElementById('gesture-feedback-styles')) {
            const style = document.createElement('style');
            style.id = 'gesture-feedback-styles';
            style.textContent = `
                @keyframes gestureFeedback {
                    0% { 
                        opacity: 0; 
                        transform: translate(-50%, -50%) scale(0.5); 
                    }
                    20% { 
                        opacity: 1; 
                        transform: translate(-50%, -50%) scale(1.1); 
                    }
                    100% { 
                        opacity: 0; 
                        transform: translate(-50%, -50%) scale(1); 
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(feedback);
        
        // Remove feedback after animation
        setTimeout(() => {
            if (feedback.parentNode) {
                feedback.parentNode.removeChild(feedback);
            }
        }, 1500);
    }
    
    async updateCurrentTrack() {
        try {
            const response = await fetch(`${this.options.backendUrl}/api/spotify/current`);
            const data = await response.json();
            
            if (data.ok && data.playing) {
                this.currentTrack = data;
                this.updateTrackUI(data);
            } else {
                this.currentTrack = null;
                this.updateTrackUI(null);
            }
        } catch (error) {
            console.error('Failed to update current track:', error);
        }
    }
    
    updateTrackUI(trackData) {
        // Update track info display
        const trackTitle = document.getElementById('current-track-title');
        const trackArtist = document.getElementById('current-track-artist');
        const trackArt = document.getElementById('current-track-art');
        const playPauseBtn = document.getElementById('play-pause-btn');
        const progressBar = document.getElementById('progress-bar');
        
        if (trackData && trackData.track) {
            const track = trackData.track;
            const playback = trackData.playback;
            
            if (trackTitle) trackTitle.textContent = track.name;
            if (trackArtist) trackArtist.textContent = track.artists.join(', ');
            if (trackArt) {
                trackArt.src = track.album_art || '';
                trackArt.alt = `${track.name} - ${track.artists.join(', ')}`;
            }
            if (playPauseBtn) {
                playPauseBtn.innerHTML = playback.is_playing ? 
                    '<i class="fas fa-pause"></i>' : 
                    '<i class="fas fa-play"></i>';
            }
            if (progressBar) {
                const progress = (playback.progress_ms / track.duration_ms) * 100;
                progressBar.style.width = `${progress}%`;
            }
        } else {
            if (trackTitle) trackTitle.textContent = 'No track playing';
            if (trackArtist) trackArtist.textContent = '';
            if (trackArt) trackArt.src = '';
            if (playPauseBtn) playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            if (progressBar) progressBar.style.width = '0%';
        }
    }
    
    startTrackUpdates() {
        // Update track info every 2 seconds
        setInterval(() => {
            this.updateCurrentTrack();
        }, 2000);
        
        // Initial update
        this.updateCurrentTrack();
    }
    
    // Public methods for manual control
    async togglePlayPause() {
        await this.executeSpotifyAction({ action: 'play_pause' }, 'manual');
    }
    
    async nextTrack() {
        await this.executeSpotifyAction({ action: 'next' }, 'manual');
    }
    
    async previousTrack() {
        await this.executeSpotifyAction({ action: 'previous' }, 'manual');
    }
    
    async volumeUp() {
        await this.executeSpotifyAction({ action: 'volume_up' }, 'manual');
    }
    
    async volumeDown() {
        await this.executeSpotifyAction({ action: 'volume_down' }, 'manual');
    }
    
    // Cleanup method
    destroy() {
        if (this.hammer) {
            this.hammer.destroy();
        }
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
        }
        if (this.gestureRecognitionInterval) {
            this.gestureRecognitionInterval = false; // Stop the animation frame loop
        }
        // Clean up canvas
        if (this.gestureCanvas) {
            this.gestureCanvas = null;
            this.gestureCanvasCtx = null;
        }
    }
}

// Export for use in other modules
window.UnifiedGestureController = UnifiedGestureController;
