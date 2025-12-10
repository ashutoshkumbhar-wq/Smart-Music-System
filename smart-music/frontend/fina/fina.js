// Fina Recom Frontend JavaScript
class FinaRecomApp {
    constructor() {
        this.apiBase = 'http://localhost:3000/api';
        this.selectedMode = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.checkStatus();
    }

    bindEvents() {
        // Generate profile button
        document.getElementById('generate-profile-btn').addEventListener('click', () => {
            this.generateProfile();
        });

        // Mode selection
        document.querySelectorAll('[data-mode]').forEach(card => {
            card.addEventListener('click', () => {
                this.selectMode(card.dataset.mode);
            });
        });

        // Start recommendations button
        document.getElementById('start-recommendations-btn').addEventListener('click', () => {
            this.startRecommendations();
        });
    }

    async checkStatus() {
        try {
            const response = await fetch(`${this.apiBase}/fina-recom/status`);
            const data = await response.json();

            if (data.ok) {
                this.updateStatus(data.status);
                if (data.status.profile_exists) {
                    this.showProfileSection(data.status.profile_info);
                    this.showRecommendationModes();
                }
            } else {
                this.showError('Failed to check system status: ' + data.error);
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }

    updateStatus(status) {
        const statusContent = document.getElementById('status-content');
        
        if (status.profile_exists) {
            statusContent.innerHTML = `
                <div class="flex items-center space-x-2">
                    <div class="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span class="text-green-400">System Ready</span>
                </div>
                <div class="mt-2 text-sm">
                    <p>Profile exists with ${status.profile_info.top_artists_count} artists, ${status.profile_info.top_tracks_count} tracks</p>
                </div>
            `;
        } else {
            statusContent.innerHTML = `
                <div class="flex items-center space-x-2">
                    <div class="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <span class="text-yellow-400">Profile Not Found</span>
                </div>
                <div class="mt-2 text-sm">
                    <p>Generate your music profile to start getting recommendations</p>
                </div>
            `;
        }
    }

    showProfileSection(profileInfo) {
        const profileSection = document.getElementById('profile-section');
        const profileContent = document.getElementById('profile-content');
        
        profileContent.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-2xl font-bold text-blue-400">${profileInfo.top_artists_count}</div>
                    <div class="text-sm text-gray-400">Top Artists</div>
                </div>
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-2xl font-bold text-green-400">${profileInfo.top_tracks_count}</div>
                    <div class="text-sm text-gray-400">Favorite Tracks</div>
                </div>
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-2xl font-bold text-purple-400">${profileInfo.genres_count}</div>
                    <div class="text-sm text-gray-400">Genres</div>
                </div>
                <div class="bg-gray-700 rounded-lg p-4">
                    <div class="text-2xl font-bold text-orange-400">${profileInfo.playlists_summary ? 'Yes' : 'No'}</div>
                    <div class="text-sm text-gray-400">Playlist Data</div>
                </div>
            </div>
        `;
        
        profileSection.classList.remove('hidden');
    }

    showRecommendationModes() {
        document.getElementById('recommendation-modes').classList.remove('hidden');
        document.getElementById('generate-profile-section').classList.add('hidden');
    }

    selectMode(mode) {
        // Remove previous selection
        document.querySelectorAll('[data-mode]').forEach(card => {
            card.classList.remove('border-blue-500', 'border-2');
            card.classList.add('border-gray-600');
        });

        // Select new mode
        const selectedCard = document.querySelector(`[data-mode="${mode}"]`);
        selectedCard.classList.add('border-blue-500', 'border-2');
        selectedCard.classList.remove('border-gray-600');

        this.selectedMode = mode;
        document.getElementById('start-recommendations-btn').disabled = false;
    }

    async generateProfile() {
        this.showLoading('Generating your music profile...');
        
        try {
            const response = await fetch(`${this.apiBase}/fina-recom/generate-profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();
            this.hideLoading();

            if (data.ok) {
                this.showSuccess('Profile generated successfully!');
                this.showProfileSection(data.profile_summary);
                this.showRecommendationModes();
                this.updateStatus({ profile_exists: true, profile_info: data.profile_summary });
            } else {
                this.showError('Failed to generate profile: ' + data.error);
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Network error: ' + error.message);
        }
    }

    async startRecommendations() {
        if (!this.selectedMode) {
            this.showError('Please select a recommendation mode first');
            return;
        }

        this.showLoading('Starting smart recommendations...');
        
        try {
            const response = await fetch(`${this.apiBase}/fina-recom/start-recommendations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode: this.selectedMode })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.ok) {
                this.showSuccess(`Recommendations started in ${data.mode} mode!`);
                this.showCurrentStatus(data);
            } else {
                this.showError('Failed to start recommendations: ' + data.error);
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Network error: ' + error.message);
        }
    }

    showCurrentStatus(data) {
        const currentStatus = document.getElementById('current-status');
        const currentStatusContent = document.getElementById('current-status-content');
        
        currentStatusContent.innerHTML = `
            <div class="flex items-center space-x-2 mb-4">
                <div class="w-3 h-3 bg-green-500 rounded-full pulse-animation"></div>
                <span class="text-green-400 font-medium">Recommendations Active</span>
            </div>
            <div class="bg-gray-700 rounded-lg p-4">
                <h4 class="font-semibold mb-2">Mode: ${data.mode}</h4>
                <p class="text-sm text-gray-400 mb-2">${data.message}</p>
                <div class="text-xs text-gray-500">
                    The system is now building your personalized queue and will start playing music automatically.
                </div>
            </div>
        `;
        
        currentStatus.classList.remove('hidden');
    }

    showLoading(text) {
        document.getElementById('loading-text').textContent = text;
        document.getElementById('loading-overlay').classList.remove('hidden');
    }

    hideLoading() {
        document.getElementById('loading-overlay').classList.add('hidden');
    }

    showSuccess(message) {
        this.showMessage(message, 'success');
    }

    showError(message) {
        this.showMessage(message, 'error');
    }

    showMessage(message, type) {
        const container = document.getElementById('message-container');
        const messageEl = document.createElement('div');
        
        const bgColor = type === 'success' ? 'bg-green-600' : 'bg-red-600';
        const icon = type === 'success' ? 
            '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>' :
            '<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>';
        
        messageEl.innerHTML = `
            <div class="${bgColor} text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 max-w-md">
                ${icon}
                <span>${message}</span>
            </div>
        `;
        
        container.appendChild(messageEl);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (messageEl.parentNode) {
                messageEl.parentNode.removeChild(messageEl);
            }
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FinaRecomApp();
});
