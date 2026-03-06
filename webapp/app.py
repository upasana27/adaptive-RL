"""Main Flask+SocketIO app for Overcooked User Study.

Manages participant sessions, routes, SocketIO event handlers, and study scheduling.
Each participant plays 2 rounds per model (8 total rounds, randomized order).
"""

import os
import uuid
import hashlib
import time
import random
import threading
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort
from flask_socketio import SocketIO, emit, disconnect
import yaml
import eventlet

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
participant_round_success = {}  # participant_id -> {round_idx: bool} - tracks 100% success per round


# === Helper Functions ===

def hash_pseudonym(pseudonym):
    """SHA256 hash of pseudonym for privacy."""
    return hashlib.sha256(pseudonym.encode()).hexdigest()


def generate_round_sequence():
    """Generate study sequence:
    - Practice: 1 episode (40s, random agent for controls)
    - 2 trial rounds with rule-based agents (5 episodes each, 40s or recipe completion) - NO questionnaire
    - Real game intro page
    - 4 main rounds with L2_left agents - questionnaire after EACH round
    
    NOTE: Human plays on the RIGHT side, AI partner plays on the LEFT side.
    Round = a block of episodes with one AI partner, followed by a questionnaire
    Episode = a single 40-second game session within a round
    """
    import random
    sequence = []
    
    # Practice round - 1 episode, 40 seconds, random agent (controls practice)
    sequence.append({
        'type': 'practice',
        'model': None,
        'episodes': 1,
        'round_name': 'Practice - Get Familiar with Controls',
        'is_trial': False,
        'needs_questionnaire': False
    })
    
    # 2 trial rounds with rule-based agents - 5 episodes each, NO questionnaire
    sequence.append({
        'type': 'trial',
        'model': 'rule_based_1',
        'episodes': 5,
        'round_name': 'Trial Block 1 (5 episodes)',
        'is_trial': True,
        'needs_questionnaire': False
    })
    sequence.append({
        'type': 'trial', 
        'model': 'rule_based_2',
        'episodes': 5,
        'round_name': 'Trial Block 2 (5 episodes)',
        'is_trial': True,
        'needs_questionnaire': False
    })
    
    # 4 main rounds: first 2 with pace_baseline, last 2 with L2_left_big (mixed)
    # Deterministic order for consistency
    main_agents = [
        ('pace_baseline', 'Round 1 (5 episodes)'),
        ('pace_baseline', 'Round 2 (5 episodes)'),
        ('L2_left_big', 'Round 3 (5 episodes)'),
        ('L2_left_big', 'Round 4 (5 episodes)'),
    ]
    
    for i, (model, name) in enumerate(main_agents, 1):
        sequence.append({
            'type': 'main',
            'model': model,
            'episodes': 5,
            'round_name': f'Round {i} (5 episodes)',
            'is_trial': False,
            'needs_questionnaire': True
        })
    
    return sequence


# === Routes ===

@app.route('/')
def index():
    """Landing page with consent form."""
    # Capture Prolific URL parameters
    prolific_pid = request.args.get('PROLIFIC_PID', '')
    study_id = request.args.get('STUDY_ID', '')
    session_id = request.args.get('SESSION_ID', '')
    return render_template('index.html',
                          prolific_pid=prolific_pid,
                          study_id=study_id,
                          session_id=session_id)


@app.route('/start', methods=['POST'])
def start():
    """Process consent and start study."""
    consent = request.form.get('consent')
    
    if not consent:
        return redirect(url_for('index'))
    
    # Create session - use Prolific PID if available, otherwise generate UUID
    prolific_pid = request.form.get('prolific_pid', '').strip()
    if prolific_pid:
        participant_id = prolific_pid
    else:
        participant_id = str(uuid.uuid4())
    session['participant_id'] = participant_id
    session['prolific_pid'] = prolific_pid
    session['prolific_study_id'] = request.form.get('study_id', '').strip()
    session['prolific_session_id'] = request.form.get('session_id', '').strip()
    session['alias'] = participant_id[:8]
    session['alias_hash'] = hash_pseudonym(participant_id[:8])
    session['round_sequence'] = generate_round_sequence()
    session['current_round_idx'] = 0
    session['current_episode_idx'] = 0  # Track episodes within a round
    session['start_time'] = time.time()
    session['completed_rounds'] = 0
    session['round_data'] = []  # Track round metadata for summary
    session['round_success'] = {}  # Track success per main round for remuneration
    
    # Initialize trajectory logging
    LOGGER.init_trajectory(session_id=participant_id, participant_id=participant_id)
    
    return redirect(url_for('questionnaire'))


@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    """Questionnaire page - collect age and education."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        # Store questionnaire responses
        age = request.form.get('age', '').strip()
        education = request.form.get('education', '').strip()
        overcooked_experience = request.form.get('overcooked_experience', '').strip()
        
        if not age or not education:
            return redirect(url_for('questionnaire'))
        
        # Store in session
        session['age'] = age
        session['education'] = education
        session['overcooked_experience'] = overcooked_experience
        
        # Also log to file for record keeping
        LOGGER.log_participant_info(
            participant_id=session['participant_id'],
            alias=session['alias'],
            age=age,
            education=education,
            overcooked_experience=overcooked_experience
        )
        
        return redirect(url_for('instructions'))
    
    # GET request - show questionnaire form
    return render_template('questionnaire.html')


@app.route('/instructions')
def instructions():
    """Show game instructions before practice round."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('instructions.html')


@app.route('/transition')
def transition():
    """Transition page between practice and trial rounds."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('transition.html')


@app.route('/real_game_intro')
def real_game_intro():
    """Introduction page before main study rounds."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('real_game_intro.html')


@app.route('/advance_round')
def advance_round():
    """Advance to the next round and redirect appropriately."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    sequence = session.get('round_sequence', [])
    current_round_idx = session.get('current_round_idx', 0)
    
    # Check if the JUST COMPLETED round needs a questionnaire
    if current_round_idx < len(sequence):
        completed_round = sequence[current_round_idx]
        if completed_round.get('needs_questionnaire'):
            # Show NASA TLX before advancing
            return redirect(url_for('nasa_tlx'))
    
    # Now increment round index
    session['current_round_idx'] = current_round_idx + 1
    session['current_episode_idx'] = 0
    
    next_round_idx = session['current_round_idx']
    
    # Check if study is complete
    if next_round_idx >= len(sequence):
        return redirect(url_for('thanks'))
    
    next_round = sequence[next_round_idx]
    next_type = next_round.get('type')
    
    # Special transitions:
    # After practice (idx=0) → show transition page before trial rounds
    if next_round_idx == 1 and next_type == 'trial':
        return redirect(url_for('transition'))
    
    # After trial rounds (idx=2) → show real_game_intro before main rounds
    if next_round_idx == 3 and next_type == 'main':
        return redirect(url_for('real_game_intro'))
    
    # Route based on round type
    if next_type == 'practice':
        return redirect(url_for('demo'))
    elif next_type == 'trial':
        return redirect(url_for('game'))
    elif next_type == 'main':
        return redirect(url_for('game'))
    else:
        return redirect(url_for('game'))


@app.route('/demo')
def demo():
    """Practice round."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    sequence = session.get('round_sequence', [])
    round_idx = session.get('current_round_idx', 0)
    
    if round_idx >= len(sequence):
        return redirect(url_for('thanks'))
    
    round_info = sequence[round_idx]
    
    return render_template('demo.html', 
                          participant_id=session['participant_id'],
                          round_name=round_info.get('round_name', 'Practice'),
                          episodes_total=round_info.get('episodes', 1),
                          model_id=round_info.get('model'),
                          is_demo=False)


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
    episode_idx = session.get('current_episode_idx', 0)
    round_type = round_info.get('type', 'main')  # 'practice', 'trial', or 'main'
    
    return render_template('game.html',
                          participant_id=session['participant_id'],
                          round_num=round_idx + 1,
                          total_rounds=len(sequence),
                          episode_num=episode_idx + 1,
                          total_episodes=round_info.get('episodes', 5),
                          round_name=round_info.get('round_name', f"Round {round_idx + 1}"),
                          model_id=round_info['model'],
                          level_id=round_info.get('level', 'default'),
                          is_demo=False,
                          is_trial=round_info.get('is_trial', False),
                          round_type=round_type,
                          round_idx=round_idx)


@app.route('/thanks')
def thanks():
    """Study completion page."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    participant_id = session['participant_id']
    elapsed = time.time() - session.get('start_time', time.time())
    completed = session.get('completed_rounds', 0)
    
    # Calculate remuneration
    base_pay = 2.50
    bonus_per_round = 2.50 / 4  # $0.625 per round
    
    # Get success data for main rounds only (round indices 3-6 in sequence)
    round_success_data = participant_round_success.get(participant_id, {})
    sequence = session.get('round_sequence', [])
    
    round_details = []
    bonus_rounds = 0
    for idx, r in enumerate(sequence):
        if r.get('type') == 'main':
            success_info = round_success_data.get(idx, {})
            all_success = success_info.get('all_success', False)
            episode_successes = success_info.get('episode_successes', [])
            if all_success:
                bonus_rounds += 1
            round_details.append({
                'round_name': r.get('round_name', f'Round {idx}'),
                'model_id': r.get('model', ''),
                'all_success': all_success,
                'episode_successes': episode_successes,
                'episodes_successful': sum(1 for s in episode_successes if s),
                'episodes_total': len(episode_successes) if episode_successes else r.get('episodes', 5)
            })
    
    total_bonus = bonus_rounds * bonus_per_round
    total_pay = base_pay + total_bonus
    
    remuneration = {
        'base_pay': base_pay,
        'bonus_per_round': bonus_per_round,
        'bonus_rounds': bonus_rounds,
        'total_bonus': total_bonus,
        'total_pay': round(total_pay, 2),
        'round_details': round_details
    }
    
    # Save remuneration to session for logging
    session['remuneration'] = remuneration
    
    # Save/update participant summary with remuneration data
    all_questionnaires = []
    for round_data in session.get('round_data', []):
        if 'questionnaire' in round_data:
            all_questionnaires.append(round_data['questionnaire'])
    
    LOGGER.save_participant_summary(
        participant_id=participant_id,
        alias_hash=session.get('alias_hash', ''),
        age=session.get('age', ''),
        education=session.get('education', ''),
        rounds=session.get('round_data', []),
        questionnaires=all_questionnaires,
        overcooked_experience=session.get('overcooked_experience', ''),
        remuneration=remuneration
    )
    
    return render_template('thanks.html',
                          rounds_completed=completed,
                          total_time=int(elapsed),
                          remuneration=remuneration,
                          participant_id=participant_id)


@app.route('/nasa_tlx')
def nasa_tlx():
    """NASA TLX questionnaire after each round."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    round_idx = session.get('current_round_idx', 0)
    sequence = session.get('round_sequence', [])
    
    if round_idx >= len(sequence):
        return redirect(url_for('thanks'))
    
    round_info = sequence[round_idx]
    
    return render_template('nasa_tlx.html',
                          participant_id=session['participant_id'],
                          round_num=round_idx + 1,
                          round_name=round_info.get('round_name', f"Round {round_idx + 1}"))


@app.route('/submit_nasa_tlx', methods=['POST'])
def submit_nasa_tlx():
    """Process NASA TLX responses and advance to next round."""
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    
    # Log the NASA TLX responses
    tlx_data = {
        'mental_demand': request.form.get('mental_demand'),
        'physical_demand': request.form.get('physical_demand'),
        'temporal_demand': request.form.get('temporal_demand'),
        'performance': request.form.get('performance'),
        'effort': request.form.get('effort'),
        'frustration': request.form.get('frustration'),
        'partner_proactive': request.form.get('partner_proactive'),
        'partner_adaptive': request.form.get('partner_adaptive'),
    }
    
    round_idx = session.get('current_round_idx', 0)
    sequence = session.get('round_sequence', [])
    round_info = sequence[round_idx] if round_idx < len(sequence) else {}
    
    LOGGER.log_nasa_tlx(session['participant_id'], 
                       round_idx,
                       tlx_data)
    
    # Save round trajectory if questionnaire is for a round that needs it
    if round_info.get('needs_questionnaire'):
        round_data = LOGGER.save_round_trajectory(
            session_id=session['participant_id'],
            participant_id=session['participant_id']
        )
        if round_data:
            # Add questionnaire with round and partner info
            round_data['questionnaire'] = {
                'round_num': round_idx + 1,
                'round_name': round_info.get('round_name'),
                'model_id': round_info.get('model'),
                'responses': tlx_data
            }
            session['round_data'].append(round_data)
    else:
        # For rounds without questionnaire, still track the round
        session['round_data'].append({
            'round_num': round_idx + 1,
            'round_name': round_info.get('round_name'),
            'model_id': round_info.get('model'),
            'pkl_file': f"round_{round_idx + 1}.pkl",
            'num_episodes': round_info.get('episodes', 1)
        })
    
    # Advance to next round
    session['current_round_idx'] = session.get('current_round_idx', 0) + 1
    session['current_episode_idx'] = 0
    session['completed_rounds'] = session.get('completed_rounds', 0) + 1
    
    # Check if study is complete
    if session['current_round_idx'] >= len(sequence):
        # Save final participant summary with all questionnaires
        all_questionnaires = []
        for round_data in session.get('round_data', []):
            if 'questionnaire' in round_data:
                all_questionnaires.append(round_data['questionnaire'])
        
        LOGGER.save_participant_summary(
            participant_id=session['participant_id'],
            alias_hash=session.get('alias_hash', ''),
            age=session.get('age', ''),
            education=session.get('education', ''),
            rounds=session.get('round_data', []),
            questionnaires=all_questionnaires,
            overcooked_experience=session.get('overcooked_experience', ''),
            remuneration=session.get('remuneration', {})
        )
        return redirect(url_for('thanks'))
    
    # If just completed practice round (index 0), show transition page
    if session['current_round_idx'] == 1:
        return redirect(url_for('transition'))
    
    # Go to next round
    return redirect(url_for('game'))



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
    round_type = data.get('round_type', 'main')  # 'practice', 'trial', or 'main'
    
    if not participant_id:
        emit('error', {'msg': 'Missing participant_id'})
        return
    
    # Create environment with round_type for correct player positions
    try:
        env_id = env_manager.create_env_for_session(
            session_id=sid,
            level=level,
            model=model_id,
            demo=is_demo,
            round_type=round_type
        )
    except Exception as e:
        emit('error', {'msg': f'Failed to create env: {str(e)}'})
        return
    
    active_sessions[sid] = {
        'participant_id': participant_id,
        'env_id': env_id,
        'model_id': model_id,
        'level': level,
        'round_type': round_type,  # 'practice', 'trial', or 'main'
        'round_idx': data.get('round_idx', 0),  # Index in round sequence
        'start_ts': time.time(),
        'is_demo': is_demo,
        'step_count': 0,
        'total_reward': 0,
        'last_action': 0,
        'pending_step': False,
        'episode_num': 1,  # Track current episode within round
        'total_episodes': data.get('total_episodes', 1),  # Total episodes for this round
        'trajectory_episode': {  # Track trajectory data for current episode
            'human_obs': [],
            'human_actions': [],
            'partner_obs': [],
            'partner_actions': [],
            'rewards': []
        }
    }
    
    # Start logger session (get alias_hash from Flask session if available)
    alias_hash = 'unknown'  # Flask session not accessible in SocketIO context easily
    LOGGER.start_session(participant_id, alias_hash, sid, 
                        metadata={'model': model_id, 'level': level, 'demo': is_demo})
    
    # Initialize round trajectory (get round info from global context would be needed here)
    # For now, we'll initialize this when the game starts
    LOGGER.start_round_trajectory(
        session_id=participant_id,
        round_num=active_sessions[sid]['episode_num'],
        round_name=f"Round {active_sessions[sid]['episode_num']}",
        model_id=model_id
    )
    LOGGER.start_episode_trajectory(session_id=participant_id)
    
    emit('joined', {'session_id': sid})
    
    # Send initial frame
    frame_b64, frame_info = env_manager.get_frame(env_id)
    emit('frame', {'frame': frame_b64, 'step': 0})
    
    # Start background game loop using eventlet greenlet
    def game_loop():
        """
        Step environment at 10Hz (human-perceivable rate), send frames at 20Hz.
        
        The game stepping (environment updates) happens at 10 Hz (100ms per step),
        which is a human-perceivable rate that matches typical game speeds.
        Rendered frames are sent to the client at 20Hz for smooth visuals between steps.
        
        Track last executed action to prevent action duplication - if client
        doesn't send a new action, use NOOP (0) instead of repeating old action.
        """
        last_step_time = time.time()
        step_interval = 1.0 / 10  # Step environment at 10Hz (100ms per step)
        last_frame_time = time.time()
        frame_interval = 1.0 / 20  # Send frames at 20Hz (50ms per frame)
        step_count_for_timing = 0
        episode_wall_start = time.time()
        last_action_id = None  # Track which action we last executed
        
        while sid in active_sessions:
            try:
                current_time = time.time()
                sess = active_sessions[sid]
                
                # Initialize step variables
                obs = None
                reward = 0
                done = False
                info = {}
                action = 0
                partner_action = 0
                
                # Step environment at 10Hz (only when enough time has passed)
                if current_time - last_step_time >= step_interval:
                    # Get pending action and its ID
                    pending_action = sess.get('last_action', 0)
                    pending_action_id = sess.get('last_action_id', None)
                    action_receive_time = sess.get('action_receive_time', None)
                    
                    # DEBUG: Log when we're about to step
                    step_start_time = time.time()
                    if action_receive_time:
                        print(f"[STEP_START] action={pending_action}, time_since_recv={(step_start_time - action_receive_time)*1000:.0f}ms")
                    
                    # Only execute action if it's new (different ID than what we last executed)
                    # Otherwise use NOOP (0) to prevent action repetition
                    if pending_action_id is not None and pending_action_id != last_action_id:
                        action = pending_action
                        last_action_id = pending_action_id
                        print(f"[STEP_EXEC] NEW action={action}, id={pending_action_id}")
                    else:
                        # No new action from client - use NOOP
                        action = 0
                    
                    # Step environment
                    result = env_manager.step(env_id, action)
                    step_end_time = time.time()
                    print(f"[STEP_DONE] action={action}, step_duration={(step_end_time - step_start_time)*1000:.0f}ms")
                    
                    obs = result.get('obs')
                    reward = result.get('reward', 0)
                    done = result.get('done', False)
                    info = result.get('info', {})
                    
                    # Get partner action and observation from info
                    partner_action = info.get('ai_action', 0)
                    
                    sess['step_count'] += 1
                    step_count_for_timing += 1
                    sess['total_reward'] += reward
                    
                    # Update step time for next iteration
                    last_step_time = current_time
                    
                    # Initialize trajectory data structure if needed
                    if not hasattr(sess, 'episode_data'):
                        sess['episode_data'] = {
                            'observations': [],
                            'actions': [],
                            'partner_actions': [],
                            'rewards': []
                        }
                    
                    # Only log actions if either:
                    # 1. User took a non-NOOP action (action != 0), OR
                    # 2. Agent took a non-NOOP action (partner_action != 0)
                    # This prevents logging of idle steps
                    if action != 0 or partner_action != 0:
                        sess['episode_data']['observations'].append(obs.tolist() if hasattr(obs, 'tolist') else obs)
                        sess['episode_data']['actions'].append(action)
                        sess['episode_data']['partner_actions'].append(partner_action)
                        sess['episode_data']['rewards'].append(reward)
                        
                        # Log step to CSV with actual environment info
                        LOGGER.log_step(
                            session_id=sid,
                            round_idx=sess.get('round_idx', 0),
                            model_id=sess['model_id'],
                            level=sess['level'],
                            step=sess['step_count'],
                            action=action,
                            reward=reward,
                            done=done,
                            info=info,  # Now includes human_side, ai_side, round_type, ai_action, etc.
                            client_ts=None
                        )
                        
                        # Log step to trajectory (for PKL file) - similar to evaluation_.py pattern
                        LOGGER.log_trajectory_step(
                            session_id=participant_id,
                            human_action=action,
                            partner_action=partner_action,
                            human_obs=obs,
                            partner_obs=info.get('opponent_obs'),
                            reward=reward,
                            done=done,
                            info=info
                        )
                    
                    # DEBUG: Log timing every 60 steps (~6 seconds at 10Hz)
                    if step_count_for_timing % 60 == 0:
                        wall_elapsed = current_time - episode_wall_start
                        env_elapsed = info.get('elapsed_time', 0)
                        wall_to_env_ratio = wall_elapsed / env_elapsed if env_elapsed > 0 else 0
                        print(f"[GAME_LOOP] steps={step_count_for_timing}, wall={wall_elapsed:.2f}s, env={env_elapsed:.2f}s, ratio={wall_to_env_ratio:.2f}x, step_rate=10Hz")
                
                if done:
                    # End current episode
                    if hasattr(sess, 'episode_data'):
                        LOGGER.log_trajectory_step(
                            session_id=participant_id,
                            human_action=None,
                            partner_action=None,
                            human_obs=None,
                            partner_obs=None,
                            reward=None,
                            done=True,
                            info=None
                        )
                    
                    # Episode ended
                    wall_duration = time.time() - sess['start_ts']
                    env_elapsed = info.get('elapsed_time', 0)
                    termination_reason = info.get('termination_reason', 'unknown')
                    recipes_delivered = info.get('recipes_delivered', 0)
                    env_done = info.get('env_done', False)
                    time_up = info.get('time_up', False)
                    # A recipe is delivered if env ended naturally (not timeout)
                    if env_done and not time_up:
                        recipes_delivered = max(recipes_delivered, 1)
                    episode_success = recipes_delivered > 0
                    
                    # Track episode success for this round
                    if 'episode_successes' not in sess:
                        sess['episode_successes'] = []
                    sess['episode_successes'].append(episode_success)
                    
                    summary = {
                        'steps': sess['step_count'],
                        'total_reward': sess['total_reward'],
                        'duration': wall_duration,
                        'episode': sess['episode_num'],
                        'recipes_delivered': recipes_delivered,
                        'episode_success': episode_success
                    }
                    
                    # DEBUG: Log episode end with timing details
                    print(f"[EPISODE_END] episode={sess['episode_num']}, steps={sess['step_count']}, wall={wall_duration:.2f}s, env={env_elapsed:.2f}s, reason={termination_reason}")
                    
                    # Check if more episodes needed
                    if sess['episode_num'] < sess['total_episodes']:
                        socketio.emit('episode_complete', {
                            'summary': summary,
                            'episode': sess['episode_num'],
                            'total_episodes': sess['total_episodes']
                        }, to=sid)
                        
                        LOGGER.start_episode_trajectory(session_id=participant_id)
                        
                        # Reset for next episode
                        sess['episode_num'] += 1
                        sess['step_count'] = 0
                        step_count_for_timing = 0
                        sess['total_reward'] = 0
                        sess['start_ts'] = time.time()
                        episode_wall_start = time.time()
                        
                        # Reset the environment
                        env_manager.reset(env_id)
                        
                        # Continue game loop
                    else:
                        # All episodes complete - save trajectory
                        LOGGER.save_round_trajectory(
                            session_id=participant_id,
                            participant_id=participant_id
                        )
                        
                        # Track round success for remuneration
                        # A round has 100% success if ALL episodes delivered at least 1 recipe
                        episode_successes = sess.get('episode_successes', [])
                        round_all_success = len(episode_successes) > 0 and all(episode_successes)
                        round_idx = sess.get('round_idx', 0)
                        
                        if participant_id not in participant_round_success:
                            participant_round_success[participant_id] = {}
                        participant_round_success[participant_id][round_idx] = {
                            'all_success': round_all_success,
                            'episode_successes': episode_successes,
                            'model_id': sess.get('model_id', ''),
                            'round_type': sess.get('round_type', '')
                        }
                        print(f"[ROUND_SUCCESS] participant={participant_id[:8]}, round_idx={round_idx}, all_success={round_all_success}, episodes={episode_successes}")
                        
                        socketio.emit('end_round', {'summary': summary}, to=sid)
                        LOGGER.end_session(sid, summary=summary)
                        del active_sessions[sid]
                        return
                
                # Send frame at 20Hz (separate from step)
                if current_time - last_frame_time >= frame_interval:
                    if sid in active_sessions:
                        frame_start = time.time()
                        frame_b64, frame_info = env_manager.get_frame(env_id)
                        frame_send_time = time.time()
                        print(f"[FRAME_SEND] render={(frame_send_time - frame_start)*1000:.0f}ms")
                        socketio.emit('frame', {
                            'frame': frame_b64,
                            'step': sess['step_count'],
                            'total_reward': float(sess['total_reward'])
                        }, to=sid)
                    last_frame_time = current_time
                
                # Very small sleep to allow other greenlets to run
                eventlet.sleep(0.001)
            except Exception as e:
                print(f'Game loop error for {sid}: {e}')
                import traceback
                traceback.print_exc()
                if sid in active_sessions:
                    del active_sessions[sid]
                return
    
    # Start game loop as eventlet greenlet
    eventlet.spawn(game_loop)


@socketio.on('action')
def handle_action(data):
    """Store human action for game loop to process."""
    sid = request.sid
    if sid not in active_sessions:
        emit('error', {'msg': 'No active session'})
        return
    
    action = data.get('action', 0)
    client_ts = data.get('client_ts')
    action_id = data.get('action_id')  # Unique ID for this action
    
    # DEBUG: Log action receipt with timing
    import time
    server_receive_time = time.time()
    print(f"[ACTION_RECV] action={action}, client_ts={client_ts}, server_time={server_receive_time*1000:.0f}, lag={(server_receive_time*1000 - client_ts):.0f}ms")
    
    # Store action and its ID for game loop to process
    # The action ID prevents the same action from being executed multiple times
    active_sessions[sid]['last_action'] = action
    active_sessions[sid]['last_action_id'] = action_id
    active_sessions[sid]['action_receive_time'] = server_receive_time
    
    # Note: Detailed step logging with env info happens in the game loop after env.step()
    # This is just acknowledging action receipt
    sess = active_sessions[sid]


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
    
    print(f'About to start SocketIO server on {host}:{port} (debug={debug})...')
    socketio.run(app, 
                 host=host,
                 port=port,
                 debug=debug,
                 use_reloader=debug)  # Only use reloader in debug mode
