document.addEventListener('DOMContentLoaded', () => {
  // ==============================
  // Video Performance Optimization (FROM CODE 2)
  // ==============================
  const sections = document.querySelectorAll('.section');
  const videos = document.querySelectorAll('.video-bg');
  let currentSectionIndex = 0;
  let isScrolling = false;
  const scrollCooldown = 1000; // 1 second cooldown after scroll animation

  // Intersection Observer for video optimization
  const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.8
  };

  const videoObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      const video = entry.target.querySelector('.video-bg');
      if (!video) return;

      if (entry.isIntersecting) {
        video.play().catch(error => {
          console.warn("Video autoplay blocked.", error);
        });
      } else {
        video.pause();
        video.currentTime = 0;
      }
    });
  }, observerOptions);

  sections.forEach(section => {
    videoObserver.observe(section);
  });

  // ==============================
  // Scroll Jacking Logic (FROM CODE 2)
  // ==============================
  function scrollToSection(index) {
    const targetPosition = sections[index].offsetTop;
    window.scrollTo({
      top: targetPosition,
      behavior: 'smooth'
    });
  }

  // MAIN SCROLL EVENT - Prevents default and handles section scrolling
  window.addEventListener('wheel', (event) => {
    if (isScrolling) {
      return;
    }
    event.preventDefault();
    isScrolling = true;

    const scrollDirection = event.deltaY;

    if (scrollDirection > 0) { // Scroll Down
      if (currentSectionIndex < sections.length - 1) {
        currentSectionIndex++;
      }
    } else { // Scroll Up
      if (currentSectionIndex > 0) {
        currentSectionIndex--;
      }
    }

    scrollToSection(currentSectionIndex);

    setTimeout(() => {
      isScrolling = false;
    }, scrollCooldown);

  }, { passive: false });

  // ==============================
  // Section Click Navigation Logic (FROM CODE 2)
  // ==============================
  const clickableSections = [
    { id: '#gesture', url: 'ge.html' },
        { id: '#mood-radio', url: 'Fina/index.html' },
    { id: '#data-powered', url: 'Cards/artist/index.html' },
    { id: '#style-radio', url: 'mood2/index.html' }
  ];

  clickableSections.forEach(item => {
    const section = document.querySelector(item.id);
    if (section) {
      let gestureDetected = false;
      let clickTimeout = null;
      
      // Listen for gesture events to prevent navigation
      document.addEventListener('gesture-detected', () => {
        gestureDetected = true;
        if (clickTimeout) {
          clearTimeout(clickTimeout);
          clickTimeout = null;
        }
        // Reset after a short delay
        setTimeout(() => { gestureDetected = false; }, 500);
      });
      
      section.addEventListener('click', (e) => {
        // Only navigate if no gesture was detected
        if (!gestureDetected) {
          // Small delay to allow gesture recognition to process first
          clickTimeout = setTimeout(() => {
            if (!gestureDetected) {
              window.location.href = item.url;
            }
          }, 200);
        }
      });
    }
  });

  // ==============================
  // Sidebar + Toggle Button Logic (FROM CODE 1)
  // ==============================
  const sidebar = document.getElementById('sidebar');
  const mainContainer = document.getElementById('main-container');
  const toggleBtn = document.getElementById('sidebarToggle');

  let sidebarVisible = false;
  let toggleVisible = false;

  // Initial state
  if (sidebar) {
    sidebar.classList.remove('active');
  }
  if (mainContainer) {
    mainContainer.classList.remove('shifted');
  }
  if (toggleBtn) {
    toggleBtn.textContent = '≡';
    toggleBtn.classList.remove('visible');
  }

  // Manual toggle
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      if (sidebarVisible) {
        sidebar.classList.remove('active');
        mainContainer.classList.remove('shifted');
        toggleBtn.textContent = '≡';
        sidebarVisible = false;
      } else {
        sidebar.classList.add('active');
        mainContainer.classList.add('shifted');
        toggleBtn.textContent = '⟨';
        sidebarVisible = true;
      }
    });
  }

  // ==============================
  // Gesture Controller Setup (FROM CODE 1)
  // ==============================
  if (window.UnifiedGestureController) {
    window.gestureController = new UnifiedGestureController({
      backendUrl: 'http://localhost:3000',
      enableTouchGestures: true,
      enableCameraGestures: false,
      gestureThreshold: 0.3,
      cooldownMs: 300, // Reduced for faster gesture response
    });

    const playPauseBtn = document.getElementById('play-pause-btn');
    const prevBtn = document.getElementById('prev-track-btn');
    const nextBtn = document.getElementById('next-track-btn');

    if (playPauseBtn) playPauseBtn.addEventListener('click', () => window.gestureController.togglePlayPause());
    if (prevBtn) prevBtn.addEventListener('click', () => window.gestureController.previousTrack());
    if (nextBtn) nextBtn.addEventListener('click', () => window.gestureController.nextTrack());

    const toggleGestureCamera = document.getElementById('toggle-gesture-camera');
    if (toggleGestureCamera) {
      toggleGestureCamera.addEventListener('click', async () => {
        const touchEnabled = window.gestureController.options.enableTouchGestures;
        const camEnabled = window.gestureController.options.enableCameraGestures;
        const isEnabled = touchEnabled && camEnabled;

        if (isEnabled) {
          window.gestureController.options.enableTouchGestures = false;
          window.gestureController.options.enableCameraGestures = false;
          if (window.gestureController.gestureRecognition) {
            window.gestureController.gestureRecognition.destroy();
          }
          console.log('Gestures + Camera disabled');
        } else {
          window.gestureController.options.enableTouchGestures = true;
          window.gestureController.options.enableCameraGestures = true;
          await window.gestureController.initCameraGestures();
          console.log('Gestures + Camera enabled');
        }

        toggleGestureCamera.classList.toggle('active', !isEnabled);
      });
    }
  }

  // ==============================
  // Search Toggle (FROM CODE 1)
  // ==============================
  const searchToggle = document.getElementById('searchToggle');
  const searchContainer = document.getElementById('searchContainer');

  if (searchToggle && searchContainer) {
    searchToggle.addEventListener('click', () => {
      searchContainer.classList.toggle('active');
      const input = document.getElementById('searchInput');
      setTimeout(() => {
        if (searchContainer.classList.contains('active')) input.focus();
      }, 300);
    });

    document.addEventListener('click', (e) => {
      if (!searchContainer.contains(e.target) && e.target !== searchToggle) {
        searchContainer.classList.remove('active');
      }
    });
  }

  // ==============================
  // Audio Player Logic (FROM CODE 1)
  // ==============================
  const songs = [
    { title: "Track 1", artist: "Artist 1", src: "track1.mp3" },
    { title: "Track 2", artist: "Artist 2", src: "track2.mp3" },
    { title: "Track 3", artist: "Artist 3", src: "track3.mp3" }
  ];

  const audio = document.getElementById('audio');
  const playBtn = document.getElementById('play-btn');
  const nextBtn = document.getElementById('next-btn');
  const prevBtn = document.getElementById('prev-btn');
  const seek = document.getElementById('seek');
  const currentTime = document.getElementById('current-time');
  const duration = document.getElementById('duration');
  const muteBtn = document.getElementById('mute-btn');
  const volumeSlider = document.getElementById('volume');
  const maximizeBtn = document.getElementById('maximize-btn');
  const player = document.getElementById('audio-player');
  const container = player ? player.querySelector('.player-container') : null;

  let currentTrack = 0;
  let isPlaying = false;

  function loadTrack(index) {
    const song = songs[index];
    audio.src = song.src;
    currentTrack = index;
    audio.load();
  }

  function togglePlay() {
    if (audio.paused) {
      audio.play();
      isPlaying = true;
      playBtn.innerHTML = '<i class="fas fa-pause"></i>';
    } else {
      audio.pause();
      isPlaying = false;
      playBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
  }

  function playTrack(index) {
    loadTrack(index);
    if (player) player.classList.remove('hidden');
    audio.play();
    isPlaying = true;
    playBtn.innerHTML = '<i class="fas fa-pause"></i>';
  }

  if (playBtn) playBtn.addEventListener('click', togglePlay);
  if (nextBtn) nextBtn.addEventListener('click', () => {
    currentTrack = (currentTrack + 1) % songs.length;
    playTrack(currentTrack);
  });
  if (prevBtn) prevBtn.addEventListener('click', () => {
    currentTrack = (currentTrack - 1 + songs.length) % songs.length;
    playTrack(currentTrack);
  });

  if (audio) {
    audio.addEventListener('timeupdate', () => {
      seek.value = (audio.currentTime / audio.duration) * 100 || 0;
      currentTime.textContent = formatTime(audio.currentTime);
      duration.textContent = formatTime(audio.duration);
    });
  }

  if (seek) {
    seek.addEventListener('input', () => {
      audio.currentTime = (seek.value / 100) * audio.duration;
    });
  }

  if (volumeSlider) {
    volumeSlider.addEventListener('input', () => {
      audio.volume = volumeSlider.value;
    });
  }

  if (muteBtn) {
    muteBtn.addEventListener('click', () => {
      audio.muted = !audio.muted;
      muteBtn.innerHTML = audio.muted
        ? '<i class="fas fa-volume-mute"></i>'
        : '<i class="fas fa-volume-up"></i>';
    });
  }

  if (maximizeBtn && container) {
    maximizeBtn.addEventListener('click', () => {
      container.classList.toggle('maximized');
      container.classList.toggle('mini');
    });
  }

  document.querySelectorAll('.music-card').forEach((card) => {
    card.addEventListener('click', () => {
      const index = parseInt(card.dataset.index);
      if (!isNaN(index)) playTrack(index);
    });
  });

  function formatTime(sec) {
    if (isNaN(sec)) return '0:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  // ==============================
  // Playlist Modal (FROM CODE 1)
  // ==============================
  const addPlaylistBtn = document.getElementById('addPlaylistBtn');
  const playlistModal = document.getElementById('playlistModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const addAnotherLinkBtn = document.getElementById('addAnotherLinkBtn');
  const playlistLinksContainer = document.getElementById('playlistLinksContainer');
  const savePlaylistsBtn = document.getElementById('savePlaylistsBtn');

  if (addPlaylistBtn && playlistModal) {
    addPlaylistBtn.addEventListener('click', () => playlistModal.classList.add('active'));
  }
  if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => playlistModal.classList.remove('active'));
  }
  const modalOverlay = document.querySelector('.modal-overlay');
  if (modalOverlay) {
    modalOverlay.addEventListener('click', () => playlistModal.classList.remove('active'));
  }
  if (addAnotherLinkBtn && playlistLinksContainer) {
    addAnotherLinkBtn.addEventListener('click', () => {
      const newInput = document.createElement('input');
      newInput.type = 'text';
      newInput.classList.add('playlist-link');
      newInput.placeholder = 'Enter playlist link';
      playlistLinksContainer.appendChild(newInput);
    });
  }
  if (savePlaylistsBtn) {
    savePlaylistsBtn.addEventListener('click', () => {
      const links = Array.from(document.querySelectorAll('.playlist-link'))
        .map(input => input.value.trim())
        .filter(Boolean);
      console.log("User Playlist Links:", links);
      playlistModal.classList.remove('active');
    });
  }

  // ==============================
  // Camera Toggle (FROM CODE 1)
  // ==============================
  const toggle = document.getElementById('toggleSwitch');
  const video = document.getElementById('cameraPreview');
  let stream = null;

  async function startCamera() {
    try {
      if (!video) {
        console.error('Error accessing camera: Camera preview element not found');
        return;
      }
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      video.style.display = 'block';
      console.log('Camera started');
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
      if (video) {
        video.style.display = 'none';
      }
      console.log('Camera stopped');
    }
  }

  if (toggle) {
    toggle.addEventListener('click', async () => {
      toggle.classList.toggle('active');

      if (toggle.classList.contains('active')) {
        await startCamera();
      } else {
        stopCamera();
      }
    });
  }

  // ==============================
  // Spotify Status (FROM CODE 1)
  // ==============================
  const spotifyStatus = document.getElementById('spotifyStatus');

  if (spotifyStatus) {
    spotifyStatus.addEventListener('click', () => {
      if (spotifyStatus.classList.contains('not-connected')) {
        window.location.href = 'profile.html';
      }
    });

    function setSpotifyStatus(isConnected) {
      if (isConnected) {
        spotifyStatus.classList.add('connected');
        spotifyStatus.classList.remove('not-connected');
        spotifyStatus.innerHTML = `
          <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" alt="Spotify" class="spotify-logo" />
          <span class="status-text" style="color: green;">connected</span>
        `;
        spotifyStatus.style.cursor = 'default';
      } else {
        spotifyStatus.classList.add('not-connected');
        spotifyStatus.classList.remove('connected');
        spotifyStatus.innerHTML = `
          <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" alt="Spotify" class="spotify-logo" />
          <span class="status-text" style="color: white;">not connected</span>
        `;
        spotifyStatus.style.cursor = 'pointer';
      }
    }

    setSpotifyStatus(false);
  }

});
