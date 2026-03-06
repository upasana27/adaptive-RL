"""Model discovery and lightweight registry for study models.

This module discovers trained model checkpoints under
`logs/Overcooked/*/ppo/*.pt` and registers them with stable ids.

It exposes a ModelRegistry that lists available models and returns
ModelPolicy stubs. Loading full PyTorch policies can be added later;
for now ModelPolicy provides metadata and a simple fallback action.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(os.getcwd())
LOGS_ROOT = ROOT / 'logs' / 'Overcooked'


def _find_latest_pt(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    # prefer latest.pt, otherwise highest numbered .pt
    latest = folder / 'latest.pt'
    if latest.exists():
        return latest
    pts = list(folder.glob('*.pt'))
    if not pts:
        return None
    # find numeric tokens in filename and pick highest number
    def score(p):
        m = re.findall(r"(\d+)", p.stem)
        if not m:
            return 0
        return max(int(x) for x in m)
    pts.sort(key=score, reverse=True)
    return pts[0]


def discover_models(logs_root: Optional[Path] = None) -> Dict[str, Path]:
    """Discover model checkpoints under `logs/Overcooked/*/ppo/`.

    Returns mapping model_id -> checkpoint_path
    model_id is the name of the parent folder under `logs/Overcooked`.
    """
    logs_root = Path(logs_root) if logs_root is not None else LOGS_ROOT
    models = {}
    if not logs_root.exists():
        return models
    for child in logs_root.iterdir():
        # each child may be a run folder
        ppo_dir = child / 'ppo'
        if not ppo_dir.exists():
            # sometimes ppo dir may be nested deeper; try to find any ppo folder
            for sub in child.rglob('ppo'):
                ppo_dir = sub
                break
        if not ppo_dir or not ppo_dir.exists():
            continue
        ckpt = _find_latest_pt(ppo_dir)
        if ckpt:
            model_id = child.name
            models[model_id] = ckpt
    
    # Explicitly add L2_left_big and pace_baseline models if they exist
    # L2_left_big now uses the ppo_L2_left_mixed_seed1 model
    l2_left_big_path = logs_root / 'ppo_L2_left_mixed_seed1' / 'ppo' / 'latest.pt'
    if l2_left_big_path.exists():
        models['L2_left_big'] = l2_left_big_path
    
    pace_baseline_path = logs_root / 'L2_left_agents' / 'baselines' / 'PACE' / 'ppo' / 'latest.pt'
    if pace_baseline_path.exists():
        models['pace_baseline'] = pace_baseline_path
    
    return models


class ModelPolicy:
    """Lightweight model policy wrapper.

    Currently a stub: returns a simple default action (0). The object
    stores checkpoint_path so later we can implement full loading.
    """
    def __init__(self, model_id: str, checkpoint_path: Optional[Path]):
        import torch
        self.model_id = model_id
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.loaded = False
        self._note = 'stub policy (heuristic fallback)'
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # PACE history storage - matching policy.py lines 36-73
        self.history_size = 5  # --history-size 5 from training
        self.history = None  # Will be initialized after model is loaded
        self.rnn_states = None
        self.masks = None
        self.last_obs = None
        
        # Try to load model immediately
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_model()

    def _load_model(self):
        """Load PACE model using PretrainedPolicy_test wrapper."""
        try:
            import sys
            import os
            # Add paths for imports
            sys.path.insert(0, '/home/asurite.ad.asu.edu/ubiswas2/adaptive-RL')
            sys.path.insert(0, '/home/asurite.ad.asu.edu/ubiswas2/adaptive-RL/environment/overcooked')
            sys.path.insert(0, '/home/asurite.ad.asu.edu/ubiswas2/adaptive-RL/environment/overcooked/gym_cooking/rebar')
            
            from environment.overcooked.policy import PretrainedPolicy_test
            
            # PACE model configuration (matching training parameters)
            self.model = PretrainedPolicy_test(
                latent_training=True,
                history_size=5,
                has_rew_done=False,
                has_meta_time_step=False,
                include_current_episode=True,
                merge_encoder_computation=True,
                last_episode_only=False,
                pop_oldest_episode=True,
                self_obs_mode=True,
                model_path=str(self.checkpoint_path),
                eval_history=True,
                agent_id=0,
                device='cuda' if self.device.type == 'cuda' else 'cpu'
            )
            
            self.loaded = True
            self._note = f'loaded PACE model from {self.checkpoint_path.name}'
            print(f"✓ Loaded model: {self.model_id} (PretrainedPolicy_test with LatentPolicy)")
            
            # Initialize history storage (must happen after model is loaded)
            self._init_history_storage()
            
        except Exception as e:
            print(f"⚠ Failed to load model {self.model_id}: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False
            self.model = None
            self._note = f'failed to load: {str(e)}'
    
    def _init_history_storage(self):
        """Initialize PeriodicHistoryStorage for PACE models matching policy.py setup."""
        try:
            import torch
            from learning.storage_ import PeriodicHistoryStorage
            
            # Setup args and variables matching policy.py lines 36-73
            self.num_test_policies = 1
            self.num_eval_policies = 1
            self.eval_history = True
            self.update_history = True
            self.last_obs = None
            
            # Create history storage exactly as in policy.py lines 71-91
            self.history = PeriodicHistoryStorage(
                num_processes=self.num_test_policies,
                num_policies=self.num_test_policies,
                history_storage_size=self.history_size,
                clear_period=self.history_size,
                refresh_interval=1,
                sample_size=None,
                has_rew_done=False,
                max_samples_per_period=None,
                step_mode=False,
                use_episodes=True,
                has_meta_time_step=False,
                include_current_episode=True,
                obs_shape=(73,),  # 71 base + 2 agent ID
                act_shape=tuple(),
                max_episode_length=500,  # Webapp episodes are ~400 steps (40s at ~10Hz), not 40 steps like training
                merge_encoder_computation=True,
                last_episode_only=False,
                pop_oldest_episode=True,
            )
            self.history.to(self.device)
            
            # Initialize RNN states and masks exactly as in policy.py lines 64-69
            # self.model is PretrainedPolicy_test; self.model.policy is the LatentPolicy
            latent_policy = self.model.policy
            if hasattr(latent_policy, 'is_recurrent') and latent_policy.is_recurrent:
                rnn_dim = latent_policy.rnn_hidden_dim
                num_states = 1 if latent_policy.share_actor_critic else 2
                self.rnn_states = torch.zeros(self.num_eval_policies, rnn_dim * num_states).to(self.device)
            else:
                self.rnn_states = None
            
            self.masks = torch.zeros(self.num_eval_policies, 1).to(self.device)
            
            print(f"✓ Initialized history storage for {self.model_id} (history_size={self.history_size})")

        except Exception as e:
            print(f"⚠ Failed to initialize history storage for {self.model_id}: {e}")
            self.history = None
    
    def load(self):
        """Public load method (for compatibility)."""
        if not self.loaded and self.checkpoint_path:
            self._load_model()
    
    def reset_history(self):
        """Reset history buffer (call between episodes)."""
        if hasattr(self, 'history') and self.history is not None:
            self.history.clear()
            # Reset RNN states and masks
            if self.rnn_states is not None:
                self.rnn_states.zero_()
            if hasattr(self, 'masks'):
                self.masks.zero_()
            self.last_obs = None
            print(f"🔄 Reset history for {self.model_id}")
        else:
            print(f"🔄 Reset (no history) for {self.model_id}")

    def act(self, obs):
        """Return an action for given observation.
        
        Args:
            obs: Observation from environment (numpy array)
            
        Returns:
            int: Action to take (0-5)
        """
        if not self.loaded or self.model is None:
            # Fallback: random action
            import numpy as np
            return np.random.randint(0, 6)
        
        try:
            import torch
            import numpy as np
            
            # Handle dict observations
            if isinstance(obs, dict):
                if 'observation' in obs:
                    obs = obs['observation']
                elif 'obs' in obs:
                    obs = obs['obs']
                else:
                    for v in obs.values():
                        if isinstance(v, np.ndarray):
                            obs = v
                            break
            
            # Convert to numpy if needed
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            
            # IMPORTANT: Model was trained with --self-obs-mode which adds agent ID (2-d one-hot)
            # Current obs is 71-d, need to pad to 73-d
            if obs.shape[0] == 71:
                # Add 2-d one-hot for agent ID based on actual player position
                aid = getattr(self, 'agent_id', 0)  # Default to player_0 (LEFT) for main rounds
                agent_id_onehot = np.zeros(2, dtype=np.float32)
                agent_id_onehot[aid] = 1.0
                obs = np.concatenate([obs, agent_id_onehot])
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Use policy.act() with history exactly like policy.py __call__ method (lines 110-118)
            with torch.no_grad():
                if self.eval_history and hasattr(self, 'history') and self.history is not None:
                    all_agent_indices = torch.arange(1).to(self.device)
                    indices = self.history.get_all_current_indices()
                    history = (self.history, (all_agent_indices,) + indices)
                    
                    # Call LatentPolicy.act with history (matching policy.py line 113)
                    # self.model is PretrainedPolicy_test; self.model.policy is the LatentPolicy
                    _, action, action_log_prob, self.rnn_states = self.model.policy.act(
                        obs_tensor,
                        self.rnn_states,
                        self.masks,
                        all_agent_indices,
                        history=history,
                        deterministic=False
                    )
                else:
                    # No history - shouldn't happen for PACE models
                    raise ValueError(f"PACE model {self.model_id} requires history storage")
            
            # Store observation for next step (matching policy.py line 151)
            self.last_obs = obs_tensor
            
            # Extract action int
            if str(action.device) == 'cuda':
                action = action.unsqueeze(-1)
            action_int = action.squeeze(-1).item()
            
            return action_int
            
        except Exception as e:
            print(f"⚠ Error in model inference for {self.model_id}: {e}")
            import traceback
            traceback.print_exc()
            import numpy as np
            return np.random.randint(0, 6)

    def set_id(self, agent_id):
        """Set the agent ID (0 or 1) for correct observation padding."""
        self.agent_id = agent_id
        print(f"🔧 Set agent_id={agent_id} for {self.model_id}")

    def update_opp_history(self, step_data, info):
        """Update history buffer after each step - matching PretrainedPolicy_test.update_opp_history."""
        if self.history is None or self.last_obs is None:
            return
        
        try:
            import torch
            reward = step_data.reward
            done = step_data.done
            
            # Use the stored observation (already a tensor)
            obs_to_store = self.last_obs.squeeze(0) if self.last_obs.dim() > 1 else self.last_obs
            
            # has_rew_done=False, so reward_tensor is None
            reward_tensor = None
            
            # Add to history - pass partner (human) action
            partner_action = info.get('self_act', 0)
            self.history.add(0, obs_to_store, partner_action, reward_tensor)
            
            if done:
                self.history.finish_episode(0)
        except Exception as e:
            print(f"⚠ Error updating history for {self.model_id}: {e}")

    def info(self):
        return {'model_id': self.model_id, 'checkpoint': str(self.checkpoint_path) if self.checkpoint_path else None, 'loaded': self.loaded, 'note': self._note}


class ModelRegistry:
    def __init__(self, logs_root: Optional[Path] = None):
        self.logs_root = Path(logs_root) if logs_root is not None else LOGS_ROOT
        self._models: Dict[str, ModelPolicy] = {}
        self.refresh()

    def refresh(self):
        found = discover_models(self.logs_root)
        self._models = {mid: ModelPolicy(mid, path) for mid, path in found.items()}

    def list_models(self):
        return list(self._models.keys())

    def get_policy(self, model_id: str) -> Optional[ModelPolicy]:
        return self._models.get(model_id)

    def info(self):
        return {mid: p.info() for mid, p in self._models.items()}


# Expose a default registry instance
REGISTRY = ModelRegistry()


if __name__ == '__main__':
    print('Discovered models:')
    for mid, info in REGISTRY.info().items():
        print(mid, info)
