// Game client for Overcooked User Study
// Handles Socket.IO connection, keyboard input, and canvas rendering

(function() {
    'use strict';

    // Configuration
    const TICK_RATE = 100; // ms between action sends (match 10Hz server game loop)
    const NOOP_ACTION = 0;
    const EPISODE_TIME_LIMIT = 40; // 40 seconds per episode
    
    // Action mapping: keyboard keys to action codes
    // Based on ActionScheme4: 0=NOOP, 1=LEFT, 2=RIGHT, 3=DOWN, 4=UP, 5=INTERACT
    const KEY_MAP = {
        'ArrowUp': 4,
        'ArrowDown': 3,
        'ArrowLeft': 1,
        'ArrowRight': 2,
        'w': 4,
        's': 3,
        'a': 1,
        'd': 2,
        ' ': 5,
        'Space': 5
    };

    // DOM elements
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas ? canvas.getContext('2d') : null;
    const statusEl = document.getElementById('status');
    const startBtn = document.getElementById('startDemo');
    const continueBtn = document.getElementById('continueBtn');
    const scoreEl = document.getElementById('score');
    const timerEl = document.getElementById('timer');

    // State
    let socket = null;
    let connected = false;
    let gameActive = false;
    let currentAction = NOOP_ACTION;
    let lastSentAction = NOOP_ACTION;  // Track last sent action to avoid duplicates
    let pressedKeys = new Set();
    let actionInterval = null;
    let startTime = null;
    let timerInterval = null;
    let actionIdCounter = 0;  // Counter to generate unique action IDs

    // Initialize
    function init() {
        if (startBtn) {
            startBtn.addEventListener('click', startGame);
        } else {
            // Auto-start for non-demo pages
            connectSocket();
        }
        
        // Keyboard listeners
        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);
        
        // Prevent arrow key scrolling
        window.addEventListener('keydown', function(e) {
            if(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
                e.preventDefault();
            }
        });
    }

    function startGame() {
        if (startBtn) startBtn.disabled = true;
        connectSocket();
    }

    function connectSocket() {
        setStatus('Connecting...', 'connecting');
        
        socket = io();
        
        socket.on('connect', onConnect);
        socket.on('disconnect', onDisconnect);
        socket.on('joined', onJoined);
        socket.on('frame', onFrame);
        socket.on('episode_complete', onEpisodeComplete);
        socket.on('end_round', onEndRound);
        socket.on('error', onError);
    }

    function onConnect() {
        connected = true;
        setStatus('Connected', 'connected');
        
        // Join session with round_type for correct player positions
        const payload = {
            participant_id: PARTICIPANT_ID || null,
            level: LEVEL_ID || 'demo',
            model: MODEL_ID || null,
            demo: IS_DEMO || false,
            total_episodes: EPISODES_TOTAL || 1,
            round_type: (typeof ROUND_TYPE !== 'undefined') ? ROUND_TYPE : 'main',
            round_idx: (typeof ROUND_IDX !== 'undefined') ? ROUND_IDX : 0
        };
        
        socket.emit('join_session', payload);
    }

    function onDisconnect() {
        connected = false;
        gameActive = false;
        setStatus('Disconnected', 'error');
        stopActionLoop();
        stopTimer();
    }

    function onJoined(data) {
        setStatus('Game starting...', 'playing');
        gameActive = true;
        startActionLoop();
        startTimer();
    }

    function onFrame(data) {
        if (!data || !data.frame) return;
        
        const frameRecvTime = Date.now();
        console.log(`[FRAME_RECV] time=${frameRecvTime}`);
        
        // Draw frame on canvas
        const img = new Image();
        img.onload = function() {
            if (ctx) {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            }
        };
        img.src = 'data:image/png;base64,' + data.frame;
        
        // Update UI
        if (data.step !== undefined) {
            // Update any step-based UI here
        }
        if (data.total_reward !== undefined && scoreEl) {
            scoreEl.textContent = data.total_reward.toFixed(2);
        }
        
        if (gameActive) {
            setStatus('Playing...', 'playing');
        }
    }

    function onEpisodeComplete(data) {
        // Episode finished but more episodes to play in this round
        const summary = data.summary || {};
        const episodeNum = data.episode || 1;
        const totalEpisodes = data.total_episodes || 1;
        
        setStatus(`Episode ${episodeNum}/${totalEpisodes} complete! Starting next episode...`, 'connected');
        
        // Update episode counter using server-authoritative episode number
        // Server sends episode=N for just-completed episode, so next is N+1
        if (typeof currentEpisode !== 'undefined') {
            currentEpisode = episodeNum + 1;
            if (document.getElementById('episode-counter')) {
                document.getElementById('episode-counter').textContent = currentEpisode;
            }
        }
        
        // Reset timer for next episode
        stopTimer();
        setTimeout(() => {
            startTimer();
            setStatus('Playing...', 'playing');
        }, 1000);
    }

    function onEndRound(data) {
        gameActive = false;
        stopActionLoop();
        stopTimer();
        
        const summary = data.summary || {};
        setStatus(`Round complete!`, 'connected');
        
        if (continueBtn) {
            // Show continue button
            continueBtn.style.display = 'inline-block';
            continueBtn.textContent = 'Continue';
            
            // Always go to advance_round which handles routing server-side
            continueBtn.href = '/advance_round';
        } else {
            // Auto-advance after a short delay
            setTimeout(() => {
                window.location.href = '/advance_round';
            }, 2000);
        }
    }

    function onError(data) {
        setStatus('Please proceed to next round', 'connected');
        console.error('Server error:', data);
    }

    function startActionLoop() {
        // Send current action at 10Hz to keep in sync with server
        if (actionInterval) return;
        
        actionInterval = setInterval(() => {
            if (!socket || !connected || !gameActive) return;
            const actionToSend = currentAction !== null ? currentAction : NOOP_ACTION;
            sendAction(actionToSend);
        }, TICK_RATE);
    }
    
    let lastActionTime = 0;
    function sendAction(action) {
        if (!socket || !connected || !gameActive) return;
        
        // Only increment action ID if the action actually changed
        if (action !== lastSentAction) {
            actionIdCounter++;
        }
        lastSentAction = action;
        
        socket.emit('action', {
            participant_id: PARTICIPANT_ID || null,
            action: action,
            action_id: actionIdCounter,  // Unique ID for this action
            client_ts: Date.now()
        });
        lastActionTime = Date.now();
    }

    function stopActionLoop() {
        if (actionInterval) {
            clearInterval(actionInterval);
            actionInterval = null;
        }
    }

    function startTimer() {
        startTime = Date.now();
        if (timerInterval) clearInterval(timerInterval);
        
        timerInterval = setInterval(() => {
            if (!startTime || !timerEl) return;
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const remaining = Math.max(0, EPISODE_TIME_LIMIT - elapsed);
            
            const minutes = Math.floor(remaining / 60);
            const seconds = remaining % 60;
            timerEl.textContent = `${seconds}s`;
            
            // Warning when time is low
            if (remaining <= 10 && remaining > 0) {
                timerEl.style.color = '#ff6b6b';
            } else if (remaining === 0) {
                timerEl.style.color = '#ff0000';
                timerEl.textContent = 'TIME UP!';
                showTimeUpOverlay();
            }
        }, 100); // Update more frequently (10Hz) for smoother countdown
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        hideTimeUpOverlay();
    }

    function showTimeUpOverlay() {
        // Only show once per episode
        if (document.getElementById('timeup-banner')) return;
        
        // Create a simple banner at the top that blinks 3 times
        const banner = document.createElement('div');
        banner.id = 'timeup-banner';
        banner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #ff6b6b 0%, #ff0000 100%);
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            z-index: 9999;
            box-shadow: 0 4px 15px rgba(255, 0, 0, 0.4);
        `;
        banner.innerHTML = '⏰ TIME UP! - Episode ending...';
        
        document.body.appendChild(banner);
        
        // Blink 3 times then auto-hide
        let blinkCount = 0;
        const blinkInterval = setInterval(() => {
            banner.style.opacity = banner.style.opacity === '0' ? '1' : '0';
            blinkCount++;
            if (blinkCount >= 6) { // 3 full blinks (on-off-on-off-on-off)
                clearInterval(blinkInterval);
                banner.style.opacity = '1';
                // Auto-hide after 1 second
                setTimeout(() => {
                    if (banner.parentNode) banner.remove();
                }, 1000);
            }
        }, 200);
    }

    function hideTimeUpOverlay() {
        const banner = document.getElementById('timeup-banner');
        if (banner) {
            banner.remove();
        }
    }

    function handleKeyDown(e) {
        const key = e.key;
        if (pressedKeys.has(key)) return; // Already pressed
        
        pressedKeys.add(key);
        
        const action = KEY_MAP[key];
        if (action !== undefined) {
            currentAction = action;
            console.log(`[KEY_DOWN] key=${key}, action=${action}, time=${Date.now()}`);
            // Send immediately for instant feedback
            if (gameActive) {
                sendAction(action);
            }
            e.preventDefault();
        }
    }

    function handleKeyUp(e) {
        const key = e.key;
        pressedKeys.delete(key);
        
        // Find another pressed key or revert to NOOP
        let fallbackAction = NOOP_ACTION;
        for (let k of pressedKeys) {
            if (KEY_MAP[k] !== undefined) {
                fallbackAction = KEY_MAP[k];
                break;
            }
        }
        currentAction = fallbackAction;
        
        // Send immediately when key released
        if (KEY_MAP[e.key] !== undefined && gameActive) {
            sendAction(fallbackAction);
        }
        
        if (KEY_MAP[e.key] !== undefined) {
            e.preventDefault();
        }
    }

    function setStatus(message, cssClass) {
        if (!statusEl) return;
        statusEl.textContent = message;
        statusEl.className = 'status ' + (cssClass || '');
    }

    // Start when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
