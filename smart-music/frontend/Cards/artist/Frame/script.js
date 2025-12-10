// --- Spotify API Credentials ---
// IMPORTANT: You need to get your own Client ID and Client Secret from the Spotify Developer Dashboard.
// 1. Go to https://developer.spotify.com/dashboard/
// 2. Log in and create a new application.
// 3. Copy your Client ID and Client Secret and paste them below.
const clientId = 'YOUR_CLIENT_ID'; // Replace with your Spotify Client ID
const clientSecret = 'YOUR_CLIENT_SECRET'; // Replace with your Spotify Client Secret

// --- DOM Elements ---
const searchForm = document.getElementById('search-form');
const searchInput = document.getElementById('artist-search');
const playerContainer = document.getElementById('spotify-player-container');
const initialDisplay = document.getElementById('initial-artist-display');
const crowdContainer = document.getElementById('crowd-container');
const spotlightContainer = document.getElementById('spotlight-container');

/**
 * Gets an access token from the Spotify API.
 * @returns {Promise<string|null>} The access token or null if an error occurs.
 */
const getAccessToken = async () => {
    // Check if credentials are placeholders
    if (clientId === 'YOUR_CLIENT_ID' || clientSecret === 'YOUR_CLIENT_SECRET') {
        console.error("Spotify API credentials are not set. Please update script.js.");
        displayError("API credentials are not set up. Please contact the site administrator.");
        return null;
    }
    
    const result = await fetch('https://accounts.spotify.com/api/token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Basic ' + btoa(clientId + ':' + clientSecret)
        },
        body: 'grant_type=client_credentials'
    });

    if (!result.ok) {
        console.error("Failed to get access token from Spotify.");
        displayError("Could not connect to Spotify. Please try again later.");
        return null;
    }

    const data = await result.json();
    return data.access_token;
};

/**
 * Searches for an artist on Spotify.
 * @param {string} artistName - The name of the artist to search for.
 * @param {string} token - The Spotify API access token.
 * @returns {Promise<string|null>} The artist's Spotify ID or null if not found.
 */
const searchArtist = async (artistName, token) => {
    const result = await fetch(`https://api.spotify.com/v1/search?q=${encodeURIComponent(artistName)}&type=artist&limit=1`, {
        method: 'GET',
        headers: { 'Authorization': 'Bearer ' + token }
    });

    if (!result.ok) {
        console.error("Failed to search for artist.");
        displayError("There was an error searching for the artist.");
        return null;
    }

    const data = await result.json();
    if (data.artists.items.length === 0) {
        console.warn(`No artist found for "${artistName}"`);
        displayError(`No artist found matching "${artistName}". Please check the spelling and try again.`);
        return null;
    }
    return data.artists.items[0].id;
};

/**
 * Creates and embeds the Spotify player iframe.
 * @param {string} artistId - The Spotify ID of the artist.
 */
const embedSpotifyPlayer = (artistId) => {
    // Clear the container
    playerContainer.innerHTML = ''; 
    
    // Hide initial decorative elements
    if(crowdContainer) crowdContainer.style.display = 'none';
    if(spotlightContainer) spotlightContainer.style.display = 'none';

    // Create the iframe
    const iframe = document.createElement('iframe');
    iframe.style.borderRadius = '12px';
    iframe.src = `https://open.spotify.com/embed/artist/${artistId}?utm_source=generator`;
    iframe.width = "100%";
    iframe.height = "352";
    iframe.frameBorder = "0";
    iframe.allowFullscreen = "";
    iframe.allow = "autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture";
    iframe.loading = "lazy";

    // Append the player
    playerContainer.appendChild(iframe);
};

/**
 * Displays an error message in the player container.
 * @param {string} message - The error message to display.
 */
const displayError = (message) => {
    // Restore initial view if something goes wrong
    if(crowdContainer) crowdContainer.style.display = 'block';
    if(spotlightContainer) spotlightContainer.style.display = 'block';

    playerContainer.innerHTML = `
        <div class="bg-red-900 border border-red-600 text-white text-center p-4 rounded-lg">
            <p class="font-bold">An Error Occurred</p>
            <p>${message}</p>
        </div>
    `;
};


// --- Event Listener ---
searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const artistName = searchInput.value.trim();

    if (!artistName) {
        displayError("Please enter an artist's name.");
        return;
    }

    // Show loading state
    playerContainer.innerHTML = `<div class="text-center p-8"><p class="text-lg animate-pulse">Searching for ${artistName}...</p></div>`;

    const token = await getAccessToken();
    if (!token) {
        // Error is already displayed by getAccessToken
        return;
    }

    const artistId = await searchArtist(artistName, token);
    if (!artistId) {
        // Error is already displayed by searchArtist
        // We can restore the initial display if we want
        setTimeout(() => {
            if(initialDisplay) playerContainer.innerHTML = initialDisplay.outerHTML;
            if(crowdContainer) crowdContainer.style.display = 'block';
            if(spotlightContainer) spotlightContainer.style.display = 'block';
        }, 3000);
        return;
    }

    embedSpotifyPlayer(artistId);
});

console.log("Artist Mix page loaded successfully!");
