// ===============================================
// --- ARTIST MIX API INTEGRATION ---
// ===============================================

const API_BASE_URL = 'http://localhost:3000';

// ===============================================
// --- Loading Manager (modern, robust) ---
// ===============================================

/**
 * loadingManager:
 * - supports nested/concurrent loads (counter)
 * - persists count in sessionStorage so a refresh can restore the loader
 * - provides a watchdog to auto-clear after a timeout to avoid stuck loader
 */
const loadingManager = (function () {
    const STORAGE_KEY = 'artistmix_loading_count_v1';
    const WATCHDOG_MS = 30000; // 30 seconds safety
    let count = 0;
    let watchdog = null;

    // initialize from sessionStorage
    try {
        const saved = sessionStorage.getItem(STORAGE_KEY);
        if (saved) count = parseInt(saved, 10) || 0;
    } catch (e) {
        // ignore storage errors
    }

    const updateStorage = () => {
        try {
            if (count > 0) sessionStorage.setItem(STORAGE_KEY, String(count));
            else sessionStorage.removeItem(STORAGE_KEY);
        } catch (e) {
            // ignore
        }
    };

    const showElement = () => {
        const loader = document.getElementById('loader');
        if (!loader) return;
        loader.classList.add('active');
        loader.setAttribute('aria-hidden', 'false');
    };

    const hideElement = () => {
        const loader = document.getElementById('loader');
        if (!loader) return;
        loader.classList.remove('active');
        loader.setAttribute('aria-hidden', 'true');
        // let CSS transition run then set display none (CSS handles display via .active)
        setTimeout(() => {
            // if still active, do nothing
            const l = document.getElementById('loader');
            if (!l) return;
            if (!l.classList.contains('active')) {
                l.style.display = 'none';
            }
        }, 300);
    };

    const startWatchdog = () => {
        clearWatchdog();
        watchdog = setTimeout(() => {
            // force-reset to safe state
            count = 0;
            updateStorage();
            hideElement();
            console.warn('loadingManager watchdog cleared the loader to avoid stuck state.');
        }, WATCHDOG_MS);
    };

    const clearWatchdog = () => {
        if (watchdog) {
            clearTimeout(watchdog);
            watchdog = null;
        }
    };

    // Public API
    return {
        increment() {
            count = Math.max(0, count) + 1;
            updateStorage();
            // Ensure loader display logic runs after DOM ready
            if (document.readyState === 'complete' || document.readyState === 'interactive') {
                const loader = document.getElementById('loader');
                if (loader) {
                    loader.style.display = ''; // let CSS handle it
                    showElement();
                }
            } else {
                // wait for DOMContentLoaded
                document.addEventListener('DOMContentLoaded', () => {
                    const loader = document.getElementById('loader');
                    if (loader) {
                        loader.style.display = '';
                        showElement();
                    }
                }, { once: true });
            }
            startWatchdog();
        },
        decrement() {
            count = Math.max(0, count - 1);
            updateStorage();
            if (count <= 0) {
                clearWatchdog();
                count = 0;
                updateStorage();
                hideElement();
            }
        },
        forceClear() {
            count = 0;
            updateStorage();
            clearWatchdog();
            hideElement();
        },
        getCount() {
            return count;
        },
        // used on page load to re-show if a request was in progress before refresh
        restoreIfNeeded() {
            if (count > 0) {
                // show loader and restart watchdog
                const loader = document.getElementById('loader');
                if (loader) {
                    loader.style.display = '';
                    loader.classList.add('active');
                    loader.setAttribute('aria-hidden', 'false');
                }
                startWatchdog();
            } else {
                // ensure hidden state
                const loader = document.getElementById('loader');
                if (loader) {
                    loader.classList.remove('active');
                    loader.setAttribute('aria-hidden', 'true');
                    setTimeout(() => {
                        if (!loader.classList.contains('active')) loader.style.display = 'none';
                    }, 300);
                }
            }
        }
    };
})();

// Small wrapper to keep existing code compatible: call showLoading(true/false)
function showLoading(show) {
    if (show) {
        loadingManager.increment();
    } else {
        loadingManager.decrement();
    }
}

// Optional helper: change loader text dynamically
function setLoaderText(text) {
    const t = document.getElementById('loader-text');
    if (t) t.textContent = text;
}

// Restore loader state on page load (in case of refresh)
document.addEventListener('DOMContentLoaded', () => {
    loadingManager.restoreIfNeeded();
});

// ===============================================
// --- ARTGIG INTEGRATION ---
// ===============================================

// Start artgig session with tag-based music discovery
async function startArtgigSession(mode, tagProfile, artists = []) {
    try {
        console.log(`🎵 Starting artgig session: mode=${mode}, tagProfile=${tagProfile}, artists=${artists.join(', ')}`);
        
        // Show loading indicator
        showLoading(true);
        setLoaderText('Starting artgig session…');

        // Make API call to backend
        const response = await fetch(`${API_BASE_URL}/api/artgig/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                mode: mode,
                tag_profile: tagProfile,
                artists: artists
            })
        });
        
        const data = await response.json();
        
        if (data.ok) {
            // Success! Artgig session started
            console.log('✅ Artgig session started:', data.message);
            const modeText = mode === 'artist' ? `artists: ${artists.join(', ')}` : 'random discovery';
            showNotification(`✅ Started artgig session with ${modeText} and ${data.tags.join(', ')} tags!`, 'success');
            return data;
        } else {
            console.error('❌ Artgig error:', data.error);
            showNotification(`❌ ${data.error}`, 'error');
            return null;
        }
    } catch (error) {
        console.error('❌ Network error:', error);
        showNotification('❌ Failed to connect to backend. Make sure the server is running on port 3000.', 'error');
        return null;
    } finally {
        showLoading(false);
    }
}

// Get available tag profiles
async function getTagProfiles() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/artgig/tag-profiles`);
        const data = await response.json();
        
        if (data.ok) {
            return data.profiles;
        } else {
            console.error('Error fetching tag profiles:', data.error);
            return {};
        }
    } catch (error) {
        console.error('Network error:', error);
        return {};
    }
}

// Main function to search and play artist songs
async function searchAndPlayArtist(artistName, genre = '') {
    try {
        console.log(`🔍 Searching for ${artistName} in ${genre || 'all genres'}...`);
        
        // Show loading indicator if you have one
        showLoading(true);
        setLoaderText(`Building mix for ${artistName}…`);

        // Make API call to backend
        const response = await fetch(`${API_BASE_URL}/api/artist-mix/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                artist: artistName,
                genre: genre,
                limit: 20  // Number of songs to queue
            })
        });
        
        const data = await response.json();
        
        if (data.ok) {
            // Success! Songs are now playing on Spotify
            console.log('✅ Success:', data.message);
            console.log('🎵 Tracks:', data.tracks);
            
            // Show success message
            showNotification(`✅ Playing ${data.tracks_count} tracks by ${data.artist}!`, 'success');
            
            // Display results if you have a results container
            displayResults(data);
            
            return data;
        } else {
            // Error occurred
            console.error('❌ Error:', data.error);
            showNotification(`❌ ${data.error}`, 'error');
            return null;
        }
    } catch (error) {
        console.error('❌ Network error:', error);
        showNotification('❌ Failed to connect to backend. Make sure the server is running on port 5000.', 'error');
        return null;
    } finally {
        showLoading(false);
    }
}

// New function to play multiple artists
async function playMultipleArtists(artists, genre = '', options = {}) {
    try {
        console.log(`🎵 Playing multiple artists: ${artists.join(', ')}`);
        
        // Show loading indicator
        showLoading(true);
        setLoaderText(`Mixing ${artists.slice(0,3).join(', ')}${artists.length>3? ' + more':''}…`);

        // Make API call to backend
        const response = await fetch(`${API_BASE_URL}/api/artist-mix/play-multiple`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                artists: artists,
                genre: genre,
                tracks_per_artist: options.tracksPerArtist || 5,
                shuffle: options.shuffle !== false  // Default to true
            })
        });
        
        const data = await response.json();
        
        if (data.ok) {
            // Success! Songs are now playing on Spotify
            console.log('✅ Success:', data.message);
            console.log('🎵 Tracks:', data.tracks);
            
            // Show success message
            showNotification(`✅ Playing ${data.tracks_count} tracks from ${data.artists.length} artists!`, 'success');
            
            // Display results if you have a results container
            displayMultipleArtistResults(data);
            
            return data;
        } else {
            // Error occurred
            console.error('❌ Error:', data.error);
            showNotification(`❌ ${data.error}`, 'error');
            return null;
        }
    } catch (error) {
        console.error('❌ Network error:', error);
        showNotification('❌ Failed to connect to backend. Make sure the server is running on port 5000.', 'error');
        return null;
    } finally {
        showLoading(false);
    }
}

// Search for artists (for autocomplete/suggestions)
async function searchArtists(query, limit = 10) {
    try {
        const response = await fetch(
            `${API_BASE_URL}/api/artist-mix/search-artists?q=${encodeURIComponent(query)}&limit=${limit}`
        );
        
        const data = await response.json();
        
        if (data.ok) {
            return data.artists;
        } else {
            console.error('Error searching artists:', data.error);
            return [];
        }
    } catch (error) {
        console.error('Network error:', error);
        return [];
    }
}

// Get available genres
async function getGenres() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/artist-mix/genres`);
        const data = await response.json();
        
        if (data.ok) {
            return data.genres;
        } else {
            console.error('Error fetching genres:', data.error);
            return [];
        }
    } catch (error) {
        console.error('Network error:', error);
        return [];
    }
}

// Display search results (optional - customize based on your UI)
function displayResults(data) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) {
        console.log('No results container found, skipping display');
        return;
    }
    
    resultsContainer.innerHTML = '';
    resultsContainer.style.display = 'block';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'results-header';
    header.innerHTML = `
        <h2>Now Playing: ${data.artist}</h2>
        ${data.genre ? `<p class="genre-tag">${data.genre}</p>` : ''}
        <p class="track-count">${data.tracks_count} tracks queued</p>
    `;
    resultsContainer.appendChild(header);
    
    // Create track list
    const trackList = document.createElement('div');
    trackList.className = 'track-list';
    
    data.tracks.forEach((track, index) => {
        const trackItem = document.createElement('div');
        trackItem.className = 'track-item';
        trackItem.innerHTML = `
            <div class="track-number">${index + 1}</div>
            <img src="${track.image || 'placeholder.jpg'}" alt="${track.name}" class="track-image">
            <div class="track-info">
                <div class="track-name">${track.name}</div>
                <div class="track-artist">${track.artist}</div>
            </div>
            <div class="track-duration">${formatDuration(track.duration_ms)}</div>
        `;
        trackList.appendChild(trackItem);
    });
    
    resultsContainer.appendChild(trackList);
}

// Display multiple artist results
function displayMultipleArtistResults(data) {
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) {
        console.log('No results container found, skipping display');
        return;
    }
    
    resultsContainer.innerHTML = '';
    resultsContainer.style.display = 'block';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'results-header';
    header.innerHTML = `
        <h2>Now Playing: Multiple Artists Mix</h2>
        <p class="artists-list">${data.artists.join(', ')}</p>
        ${data.genre ? `<p class="genre-tag">${data.genre}</p>` : ''}
        <p class="track-count">${data.tracks_count} tracks queued from ${data.artists.length} artists</p>
        ${data.shuffled ? '<p class="shuffle-indicator">🔀 Shuffled</p>' : ''}
    `;
    resultsContainer.appendChild(header);
    
    // Create track list
    const trackList = document.createElement('div');
    trackList.className = 'track-list';
    
    data.tracks.forEach((track, index) => {
        const trackItem = document.createElement('div');
        trackItem.className = 'track-item';
        trackItem.innerHTML = `
            <div class="track-number">${index + 1}</div>
            <img src="${track.image || 'placeholder.jpg'}" alt="${track.name}" class="track-image">
            <div class="track-info">
                <div class="track-name">${track.name}</div>
                <div class="track-artist">${track.artist}</div>
            </div>
            <div class="track-duration">${formatDuration(track.duration_ms)}</div>
        `;
        trackList.appendChild(trackItem);
    });
    
    resultsContainer.appendChild(trackList);
}

// Format duration from milliseconds to MM:SS
function formatDuration(ms) {
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(0);
    return `${minutes}:${seconds.padStart(2, '0')}`;
}

// Show notification message
// --- Temporarily disabled notifications ---
function showNotification(message, type = 'info') {
    console.log(`[Notification disabled] ${type.toUpperCase()}: ${message}`);
}


// Display artist suggestions for autocomplete
function displaySuggestions(artists, container, input) {
    container.innerHTML = '';
    
    if (artists.length === 0) {
        container.style.display = 'none';
        return;
    }
    
    artists.forEach(artist => {
        const suggestion = document.createElement('div');
        suggestion.className = 'suggestion-item';
        suggestion.style.cssText = `
            display: flex;
            align-items: center;
            padding: 12px;
            cursor: pointer;
            gap: 12px;
            transition: background 0.2s;
        `;
        suggestion.innerHTML = `
            <img src="${artist.image || 'placeholder.jpg'}" 
                 alt="${artist.name}" 
                 style="width: 50px; height: 50px; border-radius: 8px; object-fit: cover;">
            <div class="suggestion-info">
                <div class="suggestion-name" style="font-weight: 600; color: #333;">${artist.name}</div>
                <div class="suggestion-genres" style="font-size: 0.85rem; color: #666;">${artist.genres.slice(0, 2).join(', ')}</div>
            </div>
        `;
        
        suggestion.addEventListener('mouseenter', () => {
            suggestion.style.background = '#f5f5f5';
        });
        
        suggestion.addEventListener('mouseleave', () => {
            suggestion.style.background = 'transparent';
        });
        
        suggestion.addEventListener('click', () => {
            input.value = artist.name;
            container.innerHTML = '';
            container.style.display = 'none';
        });
        
        container.appendChild(suggestion);
    });
    
    container.style.display = 'block';
}

// Initialize artist search with autocomplete
function initializeArtistMix() {
    const artistInput = document.getElementById('artist-input');
    const genreSelect = document.getElementById('genre-select');
    const searchButton = document.getElementById('search-btn') || document.getElementById('add-artist-btn');
    const suggestionsContainer = document.getElementById('suggestions');
    
    if (!artistInput) {
        console.warn('Artist input not found. Skipping artist mix initialization.');
        return;
    }
    
    console.log('🎵 Artist Mix initialized');
    
    // Ensure genre select is working properly
    if (genreSelect) {
        console.log('🎵 Genre select found with', genreSelect.options.length, 'options');
        
        // Add event listener to show genre info
        genreSelect.addEventListener('change', (e) => {
            const selectedValue = e.target.value;
            const selectedText = e.target.options[e.target.selectedIndex].text;
            console.log('Genre selected:', selectedValue, selectedText);
            
            // Show/hide genre info
            const genreInfo = document.getElementById('genre-info');
            const genreName = document.getElementById('selected-genre-name');
            
            if (selectedValue && genreInfo && genreName) {
                genreName.textContent = selectedText;
                genreInfo.style.display = 'block';
            } else if (genreInfo) {
                genreInfo.style.display = 'none';
            }
        });
        
        // Add click event to ensure dropdown opens
        genreSelect.addEventListener('click', (e) => {
            console.log('Genre select clicked');
        });
        
        // Add focus event
        genreSelect.addEventListener('focus', (e) => {
            console.log('Genre select focused');
        });
    }
    
    let debounceTimer;
    
    // Autocomplete for artist search
    if (suggestionsContainer) {
        artistInput.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            const query = e.target.value.trim();
            
            if (query.length < 2) {
                suggestionsContainer.innerHTML = '';
                suggestionsContainer.style.display = 'none';
                return;
            }
            
            debounceTimer = setTimeout(async () => {
                const artists = await searchArtists(query, 5);
                displaySuggestions(artists, suggestionsContainer, artistInput);
            }, 300);
        });
        
        // Close suggestions when clicking outside
        document.addEventListener('click', (e) => {
            if (!artistInput.contains(e.target) && !suggestionsContainer.contains(e.target)) {
                suggestionsContainer.style.display = 'none';
            }
        });
    }
}

// Load genres into select dropdown
async function loadGenres() {
    const genreSelect = document.getElementById('genre-select');
    if (!genreSelect) return;
    
    // Check if dropdown already has options (from HTML)
    if (genreSelect.options.length > 1) {
        console.log('🎵 Genre dropdown already has options, skipping API load');
        return;
    }
    
    try {
        const genres = await getGenres();
        
        // Only clear and reload if we got genres from API
        if (genres && genres.length > 0) {
            console.log('🎵 Loading genres from API:', genres.length);
            
            // Clear existing options
            genreSelect.innerHTML = '';
            
            // Add "All Genres" option
            const allOption = document.createElement('option');
            allOption.value = '';
            allOption.textContent = 'All Genres';
            genreSelect.appendChild(allOption);
            
            // Add genre options
            genres.forEach(genre => {
                const option = document.createElement('option');
                option.value = genre;
                option.textContent = genre.charAt(0).toUpperCase() + genre.slice(1);
                genreSelect.appendChild(option);
            });
        } else {
            console.log('🎵 No genres from API, keeping existing options');
        }
    } catch (error) {
        console.log('🎵 Failed to load genres from API, keeping existing options:', error.message);
        // Keep the existing options from HTML
    }
}

// Function to handle search button click (if you have a dedicated search button)
function handleArtistSearch() {
    const artistInput = document.getElementById('artist-input');
    const genreSelect = document.getElementById('genre-select');
    
    if (!artistInput) {
        console.error('Artist input not found');
        return;
    }
    
    const artistName = artistInput.value.trim();
    const genre = genreSelect ? genreSelect.value : '';
    
    if (!artistName) {
        showNotification('Please enter an artist name', 'error');
        return;
    }
    
    searchAndPlayArtist(artistName, genre);
}

// Debug function to check dropdown
function debugDropdown() {
    const genreSelect = document.getElementById('genre-select');
    if (genreSelect) {
        console.log('Dropdown element:', genreSelect);
        console.log('Number of options:', genreSelect.options.length);
        console.log('Current value:', genreSelect.value);
        console.log('All options:');
        for (let i = 0; i < genreSelect.options.length; i++) {
            console.log(`  ${i}: ${genreSelect.options[i].value} - ${genreSelect.options[i].text}`);
        }
    } else {
        console.log('Genre select element not found');
    }
}

// Make functions available globally
window.searchAndPlayArtist = searchAndPlayArtist;
window.playMultipleArtists = playMultipleArtists;
window.handleArtistSearch = handleArtistSearch;
window.startArtgigSession = startArtgigSession;
window.getTagProfiles = getTagProfiles;
window.debugDropdown = debugDropdown;


// ===============================================
 // --- UI INTERACTIVITY LOGIC ---
 // ===============================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Artist Mix functionality
    initializeArtistMix();
    loadGenres();
    
    // Get references to the necessary HTML elements
    const artistInput = document.getElementById('artist-input');
    const addArtistBtn = document.getElementById('add-artist-btn');
    const selectedArtistsBox = document.getElementById('selected-artists-box');
    const placeholderText = document.getElementById('placeholder-text');
    const playBtn = document.querySelector(".btn-play");

    // Function to update the visibility of the placeholder text
    const updatePlaceholderVisibility = () => {
        // If there are any elements besides the placeholder, hide it. Otherwise, show it.
        const hasPills = selectedArtistsBox ? selectedArtistsBox.querySelector('.artist-pill') : null;
        if (placeholderText) {
            placeholderText.style.display = hasPills ? 'none' : 'block';
        }
    };

    // Function to add an artist pill
    const addArtist = () => {
        const artistName = artistInput.value.trim(); // Get value and remove whitespace

        // Only add if the input is not empty
        if (artistName) {
            // Add as pill - the Play button will handle genre selection
            if (selectedArtistsBox) {
                // 1. Create the pill element
                const pill = document.createElement('div');
                pill.className = 'artist-pill';
                pill.textContent = artistName;

                // 2. Create the remove 'x'
                const removeBtn = document.createElement('span');
                removeBtn.className = 'remove-btn';
                removeBtn.innerHTML = '&times;'; // The 'x' character

                // 3. Add the remove button to the pill
                pill.appendChild(removeBtn);

                // 4. Add the complete pill to the selection box
                selectedArtistsBox.appendChild(pill);
                
                // 5. Clear the input field
                artistInput.value = '';

                // 6. Update placeholder visibility
                updatePlaceholderVisibility();

                // 7. Add an event listener to the pill to remove it on click
                pill.addEventListener('click', () => {
                    pill.remove();
                    updatePlaceholderVisibility(); // Check again after removing
                });
            }
        }
    };

    // Event listener for the add button
    if (addArtistBtn) {
        addArtistBtn.addEventListener('click', addArtist);
    }

    // Event listener to also add artist when 'Enter' is pressed in the input field
    if (artistInput) {
        artistInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault(); // Prevents form submission if it's in a form
                addArtist();
            }
        });
    }

    // Event listener for the main "Play" button
    if (playBtn) {
        playBtn.addEventListener("click", () => {
            // Get genre selection
            const genreSelect = document.getElementById('genre-select');
            const selectedGenre = genreSelect ? genreSelect.value : '';
            
            // Get all selected artists from pills
            const artistPills = selectedArtistsBox ? selectedArtistsBox.querySelectorAll('.artist-pill') : [];
            
            if (artistPills.length > 0) {
                // If there are artists selected, play them
                const artists = Array.from(artistPills).map(pill => {
                    return pill.textContent.replace('×', '').trim();
                });
                
                console.log('Selected artists:', artists);
                console.log('Selected genre:', selectedGenre);
                
                if (selectedGenre) {
                    // Use artgig integration with genre selection
                    startArtgigSession('artist', selectedGenre, artists);
                } else {
                    // Use original artist mix functionality
                    if (artists.length === 1) {
                        // Single artist - use original function
                        searchAndPlayArtist(artists[0]);
                    } else {
                        // Multiple artists - use new function
                        playMultipleArtists(artists, '', {
                            tracksPerArtist: 5,
                            shuffle: true
                        });
                    }
                }
            } else if (selectedGenre) {
                // No artists but genre selected - use artgig random mode
                startArtgigSession('random', selectedGenre, []);
            } else {
                // No artists selected and no genre - redirect to Frame page
                window.location.href = "Frame/index.html";
            }
        });
    }

    // Event listener for Reset button: remove all artist pills and reset genre/input UI state
    const resetBtn = document.getElementById('reset-visual');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            // Remove all artist pills but keep placeholder-wrapper intact
            if (selectedArtistsBox) {
                const pills = selectedArtistsBox.querySelectorAll('.artist-pill');
                pills.forEach(p => p.remove());
                updatePlaceholderVisibility();
            }

            // Reset genre select and hide genre info
            const genreSelect = document.getElementById('genre-select');
            if (genreSelect) {
                genreSelect.value = '';
                const genreInfo = document.getElementById('genre-info');
                if (genreInfo) genreInfo.style.display = 'none';
                const genreName = document.getElementById('selected-genre-name');
                if (genreName) genreName.textContent = '';
            }

            // Clear the input field
            if (artistInput) artistInput.value = '';

            // Hide results container if present
            const resultsContainer = document.getElementById('results-container');
            if (resultsContainer) resultsContainer.style.display = 'none';
        });
    }

    // Initial check in case you have pre-loaded artists
    if (updatePlaceholderVisibility) {
        updatePlaceholderVisibility();
    }
});


// ===================================================================
// --- BACKGROUND ANIMATION LOGIC (CONSTELLATION EFFECT) ---
// ===================================================================

const canvas = document.getElementById('background-animation');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let particlesArray;

    // Set canvas size to full screen
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    // Particle class
    class Particle {
        constructor(x, y, directionX, directionY, size, color) {
            this.x = x;
            this.y = y;
            this.directionX = directionX;
            this.directionY = directionY;
            this.size = size;
            this.color = color;
        }
        // Method to draw individual particle
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
            ctx.fillStyle = this.color;
            ctx.fill();
        }
        // Method to update particle position and handle wall collision
        update() {
            if (this.x > canvas.width || this.x < 0) {
                this.directionX = -this.directionX;
            }
            if (this.y > canvas.height || this.y < 0) {
                this.directionY = -this.directionY;
            }
            this.x += this.directionX;
            this.y += this.directionY;
            this.draw();
        }
    }

    // Create particle array
    function init() {
        particlesArray = [];
        let numberOfParticles = (canvas.height * canvas.width) / 9000;
        for (let i = 0; i < numberOfParticles; i++) {
            let size = (Math.random() * 2) + 1;
            let x = (Math.random() * ((innerWidth - size * 2) - (size * 2)) + size * 2);
            let y = (Math.random() * ((innerHeight - size * 2) - (size * 2)) + size * 2);
            let directionX = (Math.random() * .4) - .2;
            let directionY = (Math.random() * .4) - .2;
            // Use a greenish color palette to match the theme
            let hue = 140 + (Math.random() * 40); // Hues around green
            let saturation = 50 + (Math.random() * 30); // 50-80% saturation
            let lightness = 40 + (Math.random() * 20); // 40-60% lightness
            let color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

            particlesArray.push(new Particle(x, y, directionX, directionY, size, color));
        }
    }

    // Function to draw lines between nearby particles
    function connect() {
        let opacityValue = 1;
        for (let a = 0; a < particlesArray.length; a++) {
            for (let b = a; b < particlesArray.length; b++) {
                let distance = ((particlesArray[a].x - particlesArray[b].x) * (particlesArray[a].x - particlesArray[b].x)) 
                             + ((particlesArray[a].y - particlesArray[b].y) * (particlesArray[a].y - particlesArray[b].y));
                if (distance < (canvas.width/7) * (canvas.height/7)) {
                    opacityValue = 1 - (distance/20000);
                    let avgHue = (parseFloat(particlesArray[a].color.match(/hsl\((\d+)/)[1]) + parseFloat(particlesArray[b].color.match(/hsl\((\d+)/)[1])) / 2;
                    ctx.strokeStyle = `hsla(${avgHue}, 50%, 50%, ${opacityValue})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                    ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
                    ctx.stroke();
                }
            }
        }
    }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        ctx.clearRect(0, 0, innerWidth, innerHeight);

        for (let i = 0; i < particlesArray.length; i++) {
            particlesArray[i].update();
        }
        connect();
    }

    // Resize event listener
    window.addEventListener('resize', () => {
        canvas.width = innerWidth;
        canvas.height = innerHeight;
        init();
    });

    // Initial setup
    init();
    animate();
}
