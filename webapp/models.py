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
    return models


class ModelPolicy:
    """Lightweight model policy wrapper.

    Currently a stub: returns a simple default action (0). The object
    stores checkpoint_path so later we can implement full loading.
    """
    def __init__(self, model_id: str, checkpoint_path: Optional[Path]):
        self.model_id = model_id
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.loaded = False
        self._note = 'stub policy (heuristic fallback)'
        self.model = None
        self.device = None
        
        # History buffer for encoder (collect last N obs-action pairs)
        self.history_size = 5  # From --history-size 5 in training script
        self.obs_history = []  # List of observations
        self.action_history = []  # List of actions
        self._cached_latent = None
        self._cached_params = None
        
        # Try to load model immediately
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_model()

    def _load_model(self):
        """Load PyTorch model from checkpoint."""
        try:
            import torch
            
            self.device = torch.device('cpu')
            
            # Load checkpoint (weights_only=False for compatibility with older checkpoints)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Check if checkpoint is the model itself or a state dict
            if hasattr(checkpoint, 'eval'):
                # Checkpoint is the model directly
                self.model = checkpoint
                self.model.eval()
                self.model.to(self.device)
                self.loaded = True
                self._note = f'loaded model from {self.checkpoint_path.name}'
                print(f"✓ Loaded model: {self.model_id} ({self.model.__class__.__name__})")
            else:
                # Checkpoint is a state dict - need to create model first
                from learning.model import Policy
                
                obs_shape = (154,)  # Overcooked typical size
                action_space_n = 6
                
                self.model = Policy(
                    obs_shape,
                    action_space_n,
                    base_kwargs={'recurrent': False}
                )
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.eval()
                self.model.to(self.device)
                self.loaded = True
                self._note = f'loaded state dict from {self.checkpoint_path.name}'
                print(f"✓ Loaded model: {self.model_id}")
            
        except Exception as e:
            print(f"⚠ Failed to load model {self.model_id}: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False
            self.model = None
            self._note = f'failed to load: {str(e)}'
    
    def load(self):
        """Public load method (for compatibility)."""
        if not self.loaded and self.checkpoint_path:
            self._load_model()
    
    def reset_history(self):
        """Reset history buffer (call between rounds/episodes)."""
        self.obs_history = []
        self.action_history = []
        self._cached_latent = None
        self._cached_params = None
        print(f"🔄 Reset history for {self.model_id}")

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
                # Add 2-d one-hot for agent ID
                # AI partner is agent 1 (second agent), so one-hot is [0, 1]
                agent_id = np.array([0.0, 1.0], dtype=np.float32)  # Agent 1 (AI partner)
                obs = np.concatenate([obs, agent_id])
                print(f"🔍 Padded obs from {71} to {obs.shape[0]} dimensions for agent 1")
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Add observation to history
            self.obs_history.append(obs)
            if len(self.obs_history) > self.history_size:
                self.obs_history.pop(0)
            
            # LatentPolicy/PACE: use actor with latent
            if hasattr(self.model, 'actor') and self.model.actor is not None:
                actor = self.model.actor
                
                # RNN state (zeros if recurrent)
                rnn_hxs = None  # Model is not recurrent
                masks = torch.ones(1, 1).to(self.device)
                
                # TODO: Implement proper PeriodicHistoryStorage for full PACE encoder adaptation
                # For now, sample random latent at each step to get varied behavior
                # This is better than fixed latent which causes AI to get stuck
                # Full PACE adaptation requires:
                #   1. PeriodicHistoryStorage with episode/period/step tracking
                #   2. Proper indices format: (proc_idx, period_idx, episode_idx, length_idx)
                #   3. Agent indices mapping for multi-agent coordination
                # See train_.py lines 640-650 and evaluation_.py lines 70-80 for reference
                
                latent_dim = self.model.encoder.latent_dim
                latents = torch.randn(1, latent_dim).to(self.device) * 0.3  # Sample from N(0, 0.3^2)
                
                # Call actor.act (use deterministic=False for stochastic policy)
                with torch.no_grad():
                    action, _, _, _ = actor.act(obs_tensor, rnn_hxs, masks, latents, deterministic=False)
                    action_int = int(action.item())
                
                # Add action to history for future encoder implementation
                self.action_history.append(action_int)
                if len(self.action_history) > self.history_size:
                    self.action_history.pop(0)
                
                return action_int
            else:
                raise NotImplementedError("Model structure not recognized")
            
        except Exception as e:
            print(f"⚠ Error in model inference for {self.model_id}: {e}")
            import traceback
            traceback.print_exc()
            import numpy as np
            return np.random.randint(0, 6)

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
