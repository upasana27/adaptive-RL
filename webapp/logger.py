"""Data logging for Overcooked User Study.

Writes per-step interaction data to CSV and session metadata to JSON.
"""

import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent / 'data'
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
        
    def _write_session_metadata(self, session_id):
        """Write session metadata to JSON file."""
        if session_id not in self.sessions:
            return
        
        sess = self.sessions[session_id]
        json_path = self.log_dir / f"{sess['participant_id']}_{session_id}_session.json"
        
        with open(json_path, 'w') as f:
            json.dump(sess, f, indent=2)


# Global logger instance
LOGGER = InteractionLogger()
