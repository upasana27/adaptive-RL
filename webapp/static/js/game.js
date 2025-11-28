// Game client for Overcooked User Study
// Handles Socket.IO connection, keyboard input, and canvas rendering

(function() {
    'use strict';

    // Configuration
    const TICK_RATE = 50; // ms between action sends (20Hz for responsive controls)
    const NOOP_ACTION = 0;
    
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
    let pressedKeys = new Set();
    let actionInterval = null;
    let startTime = null;
    let timerInterval = null;

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
        socket.on('end_round', onEndRound);
        socket.on('error', onError);
    }

    function onConnect() {
        connected = true;
        setStatus('Connected', 'connected');
        
        // Join session
        const payload = {
            participant_id: PARTICIPANT_ID || null,
            level: LEVEL_ID || 'demo',
            model: MODEL_ID || null,
            demo: IS_DEMO || false
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

    function onEndRound(data) {
        gameActive = false;
        stopActionLoop();
        stopTimer();
        
        const summary = data.summary || {};
        setStatus(`Round complete! Steps: ${summary.steps || 0}`, 'connected');
        
        if (IS_DEMO && continueBtn) {
            continueBtn.style.display = 'inline-block';
        } else {
            // Auto-advance to next round
            setTimeout(() => {
                window.location.href = '/game';
            }, 3000);
        }
    }

    function onError(data) {
        setStatus('Error: ' + (data.msg || 'Unknown error'), 'error');
        console.error('Server error:', data);
    }

    function startActionLoop() {
        // Actions are now sent immediately on keypress/release
        // This interval is just a heartbeat in case no keys are pressed
        if (actionInterval) return;
        
        actionInterval = setInterval(() => {
            if (!socket || !connected || !gameActive) return;
            // Only send NOOP if we haven't sent anything recently
            if (currentAction === null && Date.now() - lastActionTime > 200) {
                sendAction(NOOP_ACTION);
            }
        }, 200); // Heartbeat at 5Hz
    }
    
    let lastActionTime = 0;
    function sendAction(action) {
        if (!socket || !connected || !gameActive) return;
        socket.emit('action', {
            participant_id: PARTICIPANT_ID || null,
            action: action,
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
        
        const TIME_LIMIT = 600; // 10 minutes in seconds
        
        timerInterval = setInterval(() => {
            if (!startTime || !timerEl) return;
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const remaining = Math.max(0, TIME_LIMIT - elapsed);
            
            const minutes = Math.floor(remaining / 60);
            const seconds = remaining % 60;
            timerEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            // Warning when time is low
            if (remaining <= 30 && remaining > 0) {
                timerEl.style.color = '#ff6b6b';
            } else if (remaining === 0) {
                timerEl.style.color = '#ff0000';
                timerEl.textContent = 'TIME UP!';
            }
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function handleKeyDown(e) {
        const key = e.key;
        if (pressedKeys.has(key)) return; // Already pressed
        
        pressedKeys.add(key);
        
        const action = KEY_MAP[key];
        if (action !== undefined) {
            currentAction = action;
            sendAction(action); // Send immediately for instant response
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
        if (KEY_MAP[e.key] !== undefined) {
            sendAction(fallbackAction);
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
