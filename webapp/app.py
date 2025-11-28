"""Main Flask+SocketIO app for Overcooked User Study.

Manages participant sessions, routes, SocketIO event handlers, and study scheduling.
Each participant plays 2 rounds per model (8 total rounds, randomized order).
"""

import os
import uuid
import hashlib
import time
import random
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort
from flask_socketio import SocketIO, emit, disconnect
import yaml

from webapp.models import REGISTRY
from webapp.env_wrapper import EnvManager
from webapp.logger import LOGGER

# Config
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'study_config.yaml')

app = Flask(__name__)
app.secret_key = os.urandom(32)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# Load config
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Globals
env_manager = EnvManager()
active_sessions = {}  # session_id -> {participant_id, env, model, level, start_ts, ...}


# === Helper Functions ===

def hash_pseudonym(pseudonym):
    """SHA256 hash of pseudonym for privacy."""
    return hashlib.sha256(pseudonym.encode()).hexdigest()


def generate_round_sequence():
    """Generate randomized sequence: 2 rounds per model."""
    models = CONFIG.get('models', [])
    rounds_per_model = CONFIG.get('rounds_per_model', 2)
    
    sequence = []
    for model_id in models:
        for _ in range(rounds_per_model):
            sequence.append({'model': model_id, 'level': 'default'})
    
    random.shuffle(sequence)
    return sequence


# === Routes ===

@app.route('/')
def index():
    """Landing page with consent form."""
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    """Process consent and start study."""
    alias = request.form.get('alias', '').strip()
    consent = request.form.get('consent')
    
    if not alias or not consent:
        return redirect(url_for('index'))
    
    # Create session
    participant_id = str(uuid.uuid4())
    session['participant_id'] = participant_id
    session['alias'] = alias
    session['alias_hash'] = hash_pseudonym(alias)
    session['round_sequence'] = generate_round_sequence()
    session['current_round_idx'] = 0
    session['start_time'] = time.time()
    session['completed_rounds'] = 0
    
    return redirect(url_for('demo'))


@app.route('/demo')
def demo():
    """Practice round."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('demo.html', 
                          participant_id=session['participant_id'],
                          is_demo=True)


@app.route('/game')
def game():
    """Main game round."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    round_idx = session.get('current_round_idx', 0)
    sequence = session.get('round_sequence', [])
    
    if round_idx >= len(sequence):
        return redirect(url_for('thanks'))
    
    round_info = sequence[round_idx]
    
    return render_template('game.html',
                          participant_id=session['participant_id'],
                          round_num=round_idx + 1,
                          total_rounds=len(sequence),
                          model_id=round_info['model'],
                          level_id=round_info['level'],
                          is_demo=False)


@app.route('/thanks')
def thanks():
    """Study completion page."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    elapsed = time.time() - session.get('start_time', time.time())
    completed = session.get('completed_rounds', 0)
    
    return render_template('thanks.html',
                          rounds_completed=completed,
                          total_time=int(elapsed))


# === SocketIO Handlers ===

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_sessions:
        print(f'Client disconnected: {sid}')
        del active_sessions[sid]


@socketio.on('join_session')
def handle_join(data):
    """Client joins a game session."""
    sid = request.sid
    participant_id = data.get('participant_id')
    level = data.get('level', 'default')
    model_id = data.get('model')
    is_demo = data.get('demo', False)
    
    if not participant_id:
        emit('error', {'msg': 'Missing participant_id'})
        return
    
    # Create environment
    try:
        env_id = env_manager.create_env_for_session(
            session_id=sid,
            level=level,
            model=model_id,
            demo=is_demo
        )
    except Exception as e:
        emit('error', {'msg': f'Failed to create env: {str(e)}'})
        return
    
    active_sessions[sid] = {
        'participant_id': participant_id,
        'env_id': env_id,
        'model_id': model_id,
        'level': level,
        'start_ts': time.time(),
        'is_demo': is_demo,
        'step_count': 0,
        'total_reward': 0
    }
    
    # Start logger session (get alias_hash from Flask session if available)
    alias_hash = 'unknown'  # Flask session not accessible in SocketIO context easily
    LOGGER.start_session(participant_id, alias_hash, sid, 
                        metadata={'model': model_id, 'level': level, 'demo': is_demo})
    
    emit('joined', {'session_id': sid})
    
    # Send initial frame
    frame_b64, frame_info = env_manager.get_frame(env_id)
    emit('frame', {'frame': frame_b64, 'step': 0})


@socketio.on('action')
def handle_action(data):
    """Process human action and step environment."""
    sid = request.sid
    if sid not in active_sessions:
        emit('error', {'msg': 'No active session'})
        return
    
    action = data.get('action', 0)
    client_ts = data.get('client_ts')
    sess = active_sessions[sid]
    env_id = sess['env_id']
    
    # Step environment
    result = env_manager.step(env_id, action)
    reward = result.get('reward', 0)
    done = result.get('done', False)
    info = result.get('info', {})
    
    sess['step_count'] += 1
    sess['total_reward'] += reward
    
    # Log to CSV
    LOGGER.log_step(
        session_id=sid,
        round_idx=0,  # TODO: Track round index properly
        model_id=sess['model_id'],
        level=sess['level'],
        step=sess['step_count'],
        action=action,
        reward=reward,
        done=done,
        info=info,
        client_ts=client_ts
    )
    
    # Send frame
    frame_b64, frame_info = env_manager.get_frame(env_id)
    emit('frame', {
        'frame': frame_b64,
        'step': sess['step_count'],
        'reward': float(reward) if reward is not None else 0.0,
        'total_reward': float(sess['total_reward']),
        'done': bool(done)
    })
    
    # Handle episode end
    if done:
        summary = {
            'steps': sess['step_count'],
            'total_reward': sess['total_reward'],
            'duration': time.time() - sess['start_ts']
        }
        emit('end_round', {'summary': summary})
        
        # End logger session
        LOGGER.end_session(sid, summary=summary)
        
        # Update session round counter (if not demo)
        if not sess['is_demo']:
            # This requires access to Flask session - simplified for now
            pass
        
        del active_sessions[sid]


# === Admin Routes ===

@app.route('/admin/models', methods=['GET'])
def list_models():
    """List discovered models."""
    info = REGISTRY.info()
    return jsonify(info)


@app.route('/admin/models/refresh', methods=['POST'])
def refresh_models():
    """Refresh model registry."""
    token = request.headers.get('X-ADMIN-TOKEN') or request.args.get('token')
    cfg_token = CONFIG.get('admin_token')
    
    if cfg_token and token != cfg_token:
        abort(401)
    
    REGISTRY.refresh()
    return jsonify({'status': 'ok', 'models': REGISTRY.list_models()})


# === Entry Point ===

if __name__ == '__main__':
    import os
    
    print('Starting study server...')
    print(f'Discovered models: {REGISTRY.list_models()}')
    
    # Support deployment platforms (Heroku, Cloud Run, etc.)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    host = os.environ.get('HOST', '0.0.0.0')
    
    socketio.run(app, 
                 host=host,
                 port=port,
                 debug=debug,
                 use_reloader=debug)  # Only use reloader in debug mode
