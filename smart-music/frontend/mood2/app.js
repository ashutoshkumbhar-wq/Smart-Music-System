// ============================================================================
// SPOTIFY MOOD MIXER - FIXED VISUALIZATION
// ============================================================================

let currentLang = null;
const API_BASE = 'http://127.0.0.1:3000/api';

const langDisplay = {
    'hi': 'Hindi',
    'en': 'English',
    'both': 'Bilingual'
};

// Map frontend mood keys to backend mood keys
const moodKeyMapping = {
    'happy': 'happy',
    'sad': 'sad_breakup',
    'motivational': 'motivational',
    'chill': 'chill_lofi',
    'hiphop': 'hiphop',
    'rap': 'rap',
    'old': 'old_classic',
    'romantic': 'romantic',
    'party': 'party'
};

// Mood definitions
const moodConfig = {
    happy: { color: '#FFE066', name: 'Happy' },
    sad: { color: '#6B8AFF', name: 'Sad' },
    motivational: { color: '#FF8866', name: 'Motivational' },
    chill: { color: '#66D9E8', name: 'Chill' },
    hiphop: { color: '#9D7AFF', name: 'Hip Hop' },
    rap: { color: '#C79AFF', name: 'Rap' },
    old: { color: '#7AA8FF', name: 'Old' },
    romantic: { color: '#FF9AA2', name: 'Romantic' },
    party: { color: '#FFB347', name: 'Party' }
};

// ============================================================================
// LANGUAGE SELECTION
// ============================================================================
function selectLanguage(lang) {
    if (!lang) return;

    currentLang = lang;
    localStorage.setItem('moodMixerLanguage', lang);

    const songLangStat = document.getElementById('songLanguage');
    if (songLangStat) songLangStat.textContent = langDisplay[lang];

    if (!window.moodTracker) {
        initializeMoodTracker();
    }

    const select = document.getElementById('languageSelect');
    if (select) {
        select.style.borderColor = 'rgba(255,255,255,0.3)';
        setTimeout(() => {
            select.style.borderColor = '';
        }, 1000);
    }
}

// ============================================================================
// MOOD TRACKER CLASS
// ============================================================================
class MoodTracker {
    constructor() {
        this.moods = {};
        this.locks = {};
        this.isUpdating = false;
        this.maxTotal = 100;

        Object.keys(moodConfig).forEach(key => {
            this.moods[key] = {
                value: 0,
                element: null,
                color: moodConfig[key].color,
                name: moodConfig[key].name
            };
            this.locks[key] = false;
        });

        this.init();
    }

    init() {
        this.setupSliders();
        this.setupLockButtons();
        this.setupResetButton();
        this.setupCanvas();
        this.updateAllDisplays();
        this.updateStats();
        this.renderVisualization();
    }

    setupSliders() {
        Object.keys(this.moods).forEach(key => {
            const slider = document.getElementById(key);
            if (!slider) return;

            this.moods[key].element = slider;

            slider.addEventListener('input', (e) => {
                this.handleSliderChange(key, parseInt(e.target.value));
            });
        });
    }

    setupLockButtons() {
        document.querySelectorAll('.lock-toggle').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const key = e.currentTarget.getAttribute('data-lock');
                this.toggleLock(key, e.currentTarget);
            });
        });
    }

    setupResetButton() {
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetAll());
        }
    }

    setupCanvas() {
        this.canvas = document.getElementById('moodCanvas');

        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = this.canvas.getBoundingClientRect();
            this.canvas.width = rect.width * dpr;
            this.canvas.height = rect.height * dpr;
            this.ctx.scale(dpr, dpr);
        }
    }

    toggleLock(key, btn) {
        this.locks[key] = !this.locks[key];

        const slider = document.getElementById(key);

        if (this.locks[key]) {
            btn.classList.add('locked');
            if (slider) slider.disabled = true;
            this.addLockAnimation(btn);
        } else {
            btn.classList.remove('locked');
            if (slider) slider.disabled = false;
        }
    }

    addLockAnimation(btn) {
        btn.style.animation = 'none';
        setTimeout(() => {
            btn.style.animation = 'lockPulse 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
        }, 10);
    }

    handleSliderChange(changedKey, newValue) {
        if (this.isUpdating || this.locks[changedKey]) return;

        this.isUpdating = true;

        newValue = Math.round(newValue / 5) * 5;
        newValue = Math.max(0, Math.min(100, newValue));

        let currentTotal = 0;

        Object.keys(this.moods).forEach(key => {
            if (key !== changedKey) {
                currentTotal += this.moods[key].value;
            }
        });

        if (currentTotal + newValue > this.maxTotal) {
            newValue = this.maxTotal - currentTotal;
            newValue = Math.max(0, newValue);
        }

        this.moods[changedKey].value = newValue;

        if (this.moods[changedKey].element) {
            this.moods[changedKey].element.value = newValue;
        }

        this.updateAllDisplays();
        this.updateStats();
        this.renderVisualization();

        this.isUpdating = false;
    }

    updateAllDisplays() {
        Object.keys(this.moods).forEach(key => {
            const mood = this.moods[key];

            const card = document.querySelector(`.mood-card[data-mood="${key}"]`);

            if (card) {
                const valueNumber = card.querySelector('.value-number');
                if (valueNumber) valueNumber.textContent = mood.value;

                const sliderFill = card.querySelector('.slider-fill');
                if (sliderFill) {
                    sliderFill.style.width = `${mood.value}%`;
                    sliderFill.style.background = mood.value > 0 ? mood.color : 'rgba(255,255,255,0.8)';
                }

                card.style.borderColor = mood.value > 0 ? `${mood.color}40` : '';
            }
        });
    }

    updateStats() {
        const total = Object.values(this.moods).reduce((sum, mood) => sum + mood.value, 0);

        const activeMoods = Object.keys(this.moods).filter(
            key => this.moods[key].value > 0
        ).length;

        const totalElement = document.getElementById('totalPercentage');
        const activeMoodsElement = document.getElementById('activeMoods');

        if (totalElement) {
            totalElement.textContent = `${total}%`;

            if (total === 100) {
                totalElement.style.color = '#66D9E8';
            } else if (total > 100) {
                totalElement.style.color = '#FF8866';
            } else {
                totalElement.style.color = '#ffffff';
            }
        }

        if (activeMoodsElement) {
            activeMoodsElement.textContent = activeMoods;
        }
    }

    renderVisualization() {
        if (!this.canvas || !this.ctx) return;

        const ctx = this.ctx;

        const width = this.canvas.getBoundingClientRect().width;
        const height = this.canvas.getBoundingClientRect().height;

        ctx.clearRect(0, 0, width, height);

        const total = Object.values(this.moods).reduce(
            (sum, mood) => sum + mood.value,
            0
        );

        if (total === 0) return;

        let currentX = 0;
        const barHeight = height - 8;
        const barY = 4;

        Object.keys(this.moods).forEach(key => {
            const mood = this.moods[key];
            if (mood.value === 0) return;

            const segmentWidth = (mood.value / 100) * width;

            const gradient = ctx.createLinearGradient(
                currentX,
                barY,
                currentX,
                barY + barHeight
            );

            gradient.addColorStop(0, mood.color);
            gradient.addColorStop(1, this.adjustColor(mood.color, -20));

            ctx.fillStyle = gradient;
            ctx.fillRect(currentX, barY, segmentWidth, barHeight);

            currentX += segmentWidth;
        });

        this.updateLegend();
    }

    updateLegend() {
        const legendContainer = document.getElementById('moodLegend');
        if (!legendContainer) return;

        legendContainer.innerHTML = '';

        Object.keys(this.moods).forEach(key => {
            const mood = this.moods[key];
            if (mood.value === 0) return;

            const legendItem = document.createElement('div');
            legendItem.style.cssText = `
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 8px;
                background: rgba(255, 255, 255, 0.04);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            `;

            const colorBox = document.createElement('div');
            colorBox.style.cssText = `
                width: 12px;
                height: 12px;
                background: ${mood.color};
                border-radius: 3px;
                flex-shrink: 0;
            `;

            const label = document.createElement('div');
            label.style.cssText = `
                flex: 1;
                font-size: 11px;
                font-weight: 600;
                color: #ffffff;
            `;
            label.textContent = mood.name;

            const value = document.createElement('div');
            value.style.cssText = `
                font-size: 12px;
                font-weight: 700;
                color: ${mood.color};
            `;
            value.textContent = `${mood.value}%`;

            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);
            legendItem.appendChild(value);

            legendContainer.appendChild(legendItem);
        });
    }

    adjustColor(color, amount) {
        const num = parseInt(color.replace('#', ''), 16);

        const r = Math.max(0, Math.min(255, (num >> 16) + amount));
        const g = Math.max(0, Math.min(255, ((num >> 8) & 0x00ff) + amount));
        const b = Math.max(0, Math.min(255, (num & 0x0000ff) + amount));

        return `#${((r << 16) | (g << 8) | b)
            .toString(16)
            .padStart(6, '0')}`;
    }

    resetAll() {
        Object.keys(this.moods).forEach(key => {
            this.moods[key].value = 0;

            if (this.moods[key].element) {
                this.moods[key].element.value = 0;
                this.moods[key].element.disabled = false;
            }

            this.locks[key] = false;
        });

        document.querySelectorAll('.lock-toggle').forEach(btn => {
            btn.classList.remove('locked');
        });

        this.updateAllDisplays();
        this.updateStats();

        if (this.canvas && this.ctx) {
            const width = this.canvas.getBoundingClientRect().width;
            const height = this.canvas.getBoundingClientRect().height;
            this.ctx.clearRect(0, 0, width, height);
        }

        const legendContainer = document.getElementById('moodLegend');
        if (legendContainer) legendContainer.innerHTML = '';

        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.style.transform = 'scale(0.95)';
            setTimeout(() => (resetBtn.style.transform = ''), 150);
        }
    }
}

// ============================================================================
// INITIALIZE
// ============================================================================
function initializeMoodTracker() {
    if (!window.moodTracker) {
        window.moodTracker = new MoodTracker();
    }
}

window.addEventListener('resize', () => {
    if (window.moodTracker && window.moodTracker.canvas) {
        window.moodTracker.setupCanvas();
        window.moodTracker.renderVisualization();
    }
});

// Page Load
document.addEventListener('DOMContentLoaded', () => {
    const savedLang = localStorage.getItem('moodMixerLanguage');
    const select = document.getElementById('languageSelect');

    if (savedLang && select) {
        select.value = savedLang;
        selectLanguage(savedLang);
    }

    if (select) {
        select.addEventListener('change', (e) => selectLanguage(e.target.value));
    }

    const backBtn = document.getElementById('backBtn');
    if (backBtn) {
        backBtn.addEventListener('click', () => window.history.back());
    }

    const generateBtn = document.getElementById('generateBtn');

    if (generateBtn) {
        generateBtn.addEventListener('click', () => {
            if (!currentLang) {
                alert('Please select a language first!');
                return;
            }

            const total = Object.values(window.moodTracker?.moods || {}).reduce(
                (sum, mood) => sum + mood.value,
                0
            );

            if (total === 0) {
                alert('Please set at least one mood percentage!');
                return;
            }

            if (total !== 100) {
                alert(`Total mood percentage must equal 100%! Current total: ${total}%`);
                return;
            }

            generateBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <circle cx="12" cy="12" r="10"></circle>
                    <path d="M12 6v6l4 2"></path>
                </svg> Mixing...
            `;

            generateBtn.disabled = true;

            // Map language: 'both' -> 'mix' for backend
            const backendLang = currentLang === 'both' ? 'mix' : currentLang;

            // Build mood weights object with backend key names
            const moodWeights = {};
            Object.keys(window.moodTracker.moods).forEach(key => {
                const backendKey = moodKeyMapping[key] || key;
                moodWeights[backendKey] = window.moodTracker.moods[key].value;
            });

            // Call the API
            fetch(`${API_BASE}/mood-mixer/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    language: backendLang,
                    mood_weights: moodWeights
                })
            })
            .then(response => response.json())
            .then(data => {
                generateBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg> Mix It
                `;

                generateBtn.disabled = false;

                if (data.ok) {
                    alert(`🎵 Perfect mix! Started playing ${langDisplay[currentLang]} songs with your mood blend!`);
                } else {
                    alert(`❌ Error: ${data.error || 'Failed to start mood mixer'}`);
                }
            })
            .catch(error => {
                console.error('Mood mixer API error:', error);
                generateBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg> Mix It
                `;

                generateBtn.disabled = false;

                alert(`❌ Network error: ${error.message}. Make sure the backend server is running on ${API_BASE}`);
            });
        });
    }
});
