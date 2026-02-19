"""Data logging for Overcooked User Study.

Writes per-step interaction data to CSV and session metadata to JSON.
Also saves trajectory data as pickle files for each round.
"""

import os
import csv
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent / 'logs'
DATA_DIR.mkdir(exist_ok=True)


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


CSV_HEADER = [
    'participant_id',
    'alias_hash',
    'session_id',
    'session_ts',
    'round_idx',
    'model_id',
    'level',
    'step',
    'action',
    'reward',
    'done',
    'info_json',
    'client_ts',
    'server_ts'
]


class InteractionLogger:
    """Logs participant interactions to CSV."""
    
    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else DATA_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session metadata cache
        self.sessions = {}
        
        # Trajectory data cache - keyed by session_id
        self.trajectories = {}
        
        
    def start_session(self, participant_id, alias_hash, session_id, metadata=None):
        """Register new session."""
        session_ts = time.time()
        
        self.sessions[session_id] = {
            'participant_id': participant_id,
            'alias_hash': alias_hash,
            'session_id': session_id,
            'session_ts': session_ts,
            'start_datetime': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Write session metadata to JSON
        self._write_session_metadata(session_id)
        
    def log_step(self, session_id, round_idx, model_id, level, step, 
                 action, reward, done, info=None, client_ts=None):
        """Log single environment step to CSV."""
        if session_id not in self.sessions:
            print(f"Warning: Session {session_id} not registered")
            return
        
        sess = self.sessions[session_id]
        server_ts = time.time()
        
        row = {
            'participant_id': sess['participant_id'],
            'alias_hash': sess['alias_hash'],
            'session_id': session_id,
            'session_ts': sess['session_ts'],
            'round_idx': round_idx,
            'model_id': model_id,
            'level': level,
            'step': step,
            'action': action,
            'reward': float(reward) if reward is not None else 0.0,
            'done': done,
            'info_json': json.dumps(convert_to_json_serializable(info)) if info else '{}',
            'client_ts': client_ts or '',
            'server_ts': server_ts
        }
        
        # Append to CSV
        csv_path = self.log_dir / f"{sess['participant_id']}_interactions.csv"
        is_new = not csv_path.exists()
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    
    def end_session(self, session_id, summary=None):
        """Mark session complete and update metadata."""
        if session_id not in self.sessions:
            return
        
        sess = self.sessions[session_id]
        sess['end_datetime'] = datetime.now().isoformat()
        sess['duration'] = time.time() - sess['session_ts']
        sess['summary'] = summary or {}
        
        self._write_session_metadata(session_id)
    
    def log_nasa_tlx(self, participant_id, round_idx, tlx_data):
        """Log NASA TLX questionnaire responses."""
        tlx_path = self.log_dir / f"{participant_id}_nasa_tlx.csv"
        is_new = not tlx_path.exists()
        
        tlx_row = {
            'participant_id': participant_id,
            'round_idx': round_idx,
            'timestamp': datetime.now().isoformat(),
            **tlx_data
        }
        
        with open(tlx_path, 'a', newline='') as f:
            fieldnames = ['participant_id', 'round_idx', 'timestamp', 
                         'mental_demand', 'physical_demand', 'temporal_demand',
                         'performance', 'effort', 'frustration']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(tlx_row)
    
    def log_participant_info(self, participant_id, alias, age, education):
        """Log participant demographic information."""
        info_path = self.log_dir / f"{participant_id}_participant_info.json"
        
        info_data = {
            'participant_id': participant_id,
            'alias': alias,
            'age': age,
            'education': education,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=2)
        
    def _write_session_metadata(self, session_id):
        """Write session metadata to JSON file."""
        if session_id not in self.sessions:
            return
        
        sess = self.sessions[session_id]
        json_path = self.log_dir / f"{sess['participant_id']}_{session_id}_session.json"
        
        with open(json_path, 'w') as f:
            json.dump(sess, f, indent=2)

    def init_trajectory(self, session_id, participant_id):
        """Initialize trajectory tracking for a session."""
        self.trajectories[session_id] = {
            'participant_id': participant_id,
            'rounds': []  # List of round data
        }
        
        # Create participant folder
        self.participant_dir = self.log_dir / participant_id
        self.participant_dir.mkdir(parents=True, exist_ok=True)
    
    def start_round_trajectory(self, session_id, round_num, round_name, model_id):
        """Start collecting trajectory for a new round."""
        if session_id not in self.trajectories:
            return
        
        self.trajectories[session_id]['current_round'] = {
            'round_num': round_num,
            'round_name': round_name,
            'model_id': model_id,
            'episodes': [],
            'start_time': time.time()
        }
    
    def start_episode_trajectory(self, session_id):
        """Start collecting trajectory for a new episode."""
        if session_id not in self.trajectories:
            return
        
        current_round = self.trajectories[session_id].get('current_round')
        if not current_round:
            return
        
        current_round['current_episode'] = {
            'observations': [],
            'actions': [],
            'partner_observations': [],
            'partner_actions': [],
            'rewards': [],
            'success': False
        }
    
    def log_trajectory_step(self, session_id, human_action, partner_action, 
                           human_obs, partner_obs, reward, done, info=None):
        """Log a single step in trajectory."""
        if session_id not in self.trajectories:
            return
        
        current_round = self.trajectories[session_id].get('current_round')
        if not current_round:
            return
        
        current_episode = current_round.get('current_episode')
        if not current_episode:
            return
        
        # Store as lists (convert numpy arrays to lists)
        if isinstance(human_obs, np.ndarray):
            human_obs = human_obs.tolist()
        if isinstance(partner_obs, np.ndarray):
            partner_obs = partner_obs.tolist()
        
        current_episode['observations'].append(human_obs)
        current_episode['actions'].append(int(human_action))
        current_episode['partner_observations'].append(partner_obs)
        current_episode['partner_actions'].append(int(partner_action))
        current_episode['rewards'].append(float(reward))
        
        if done:
            current_episode['success'] = True
    
    def end_episode_trajectory(self, session_id):
        """Mark episode complete and move to next."""
        if session_id not in self.trajectories:
            return
        
        current_round = self.trajectories[session_id].get('current_round')
        if not current_round:
            return
        
        current_episode = current_round.get('current_episode')
        if current_episode:
            current_round['episodes'].append(current_episode)
            current_round['current_episode'] = None
    
    def save_round_trajectory(self, session_id, participant_id):
        """Save round trajectory to pickle file."""
        if session_id not in self.trajectories:
            return None
        
        current_round = self.trajectories[session_id].get('current_round')
        if not current_round:
            return None
        
        # Get round number
        round_num = current_round['round_num']
        
        # Prepare trajectory dict
        trajectory_data = {
            'trajectories': current_round['episodes'],
            'num_trajectories': len(current_round['episodes']),
            'round_info': {
                'round_num': round_num,
                'round_name': current_round['round_name'],
                'model_id': current_round['model_id'],
                'num_episodes': len(current_round['episodes']),
                'duration_seconds': time.time() - current_round['start_time']
            }
        }
        
        # Save to pkl
        participant_dir = self.log_dir / participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)
        
        pkl_filename = f"round_{round_num}.pkl"
        pkl_path = participant_dir / pkl_filename
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        # Clear current round
        self.trajectories[session_id]['current_round'] = None
        
        return {
            'round_num': round_num,
            'round_name': current_round['round_name'],
            'model_id': current_round['model_id'],
            'pkl_file': pkl_filename,
            'num_episodes': len(current_round['episodes']),
            'duration_seconds': trajectory_data['round_info']['duration_seconds']
        }
    
    def save_participant_summary(self, participant_id, alias_hash, age, education, rounds, questionnaires=None):
        """Save final participant summary with inputs only (demographics + questionnaire responses)."""
        participant_dir = self.log_dir / participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'participant_id': participant_id,
            'alias_hash': alias_hash,
            'demographics': {
                'age': age,
                'education': education
            },
            'rounds': rounds,  # List of round info: round_num, round_name, model_id, pkl_file
            'questionnaires': questionnaires or [],  # List of NASA TLX responses per round
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as single JSON file
        summary_path = participant_dir / 'inputs.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_path



# Global logger instance
LOGGER = InteractionLogger()
