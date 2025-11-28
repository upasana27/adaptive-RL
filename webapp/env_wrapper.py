"""Environment manager for the study.

Provides EnvManager that creates per-session Overcooked environments with a 
random or trained AI partner. Wraps OvercookedMaker to handle the multi-agent
interface. Falls back to MockEnv for testing.
"""

import threading
import time
import io
import base64
import os
import random
import sys
from pathlib import Path

# Add local rebar to path
_overcooked_dir = Path(__file__).parent.parent / 'environment' / 'overcooked'
_rebar_dir = _overcooked_dir / 'gym_cooking' / 'rebar'
if _rebar_dir.exists() and str(_rebar_dir) not in sys.path:
    sys.path.insert(0, str(_rebar_dir))
# Add gym_cooking to path
if str(_overcooked_dir) not in sys.path:
    sys.path.insert(0, str(_overcooked_dir))

try:
    import numpy as np
    from PIL import Image
except Exception:
    np = None
    Image = None

from typing import Optional


class MockEnv:
    """Mock environment for testing - shows colored rectangles based on actions."""
    def __init__(self):
        self.step_count = 0
        self.done = False
        self.last_action = 0
        
    def reset(self):
        self.step_count = 0
        self.done = False
        self.last_action = 0
        return {}

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        reward = 0.1 if action != 0 else 0.0  # Small reward for any action
        self.done = self.step_count >= 50  # Shorter episodes for testing
        obs = {}
        info = {'mock': True, 'action': action}
        return obs, reward, self.done, info

    def render(self, mode='rgb_array'):
        # Show different colors based on last action
        # 0=NOOP (gray), 1=UP (red), 2=DOWN (blue), 3=LEFT (green), 4=RIGHT (yellow), 5=SPACE (purple)
        if np is None:
            return None
        h, w = 480, 640
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Base color depends on last action
        if self.last_action == 0:  # NOOP - gray
            arr[:, :] = [100, 100, 100]
        elif self.last_action == 1:  # UP - red
            arr[:, :] = [255, 50, 50]
        elif self.last_action == 2:  # DOWN - blue  
            arr[:, :] = [50, 50, 255]
        elif self.last_action == 3:  # LEFT - green
            arr[:, :] = [50, 255, 50]
        elif self.last_action == 4:  # RIGHT - yellow
            arr[:, :] = [255, 255, 50]
        elif self.last_action == 5:  # SPACE - purple
            arr[:, :] = [200, 50, 200]
        
        # Add step counter as visual feedback (flashing border)
        if self.step_count % 5 < 2:
            arr[0:10, :] = 255  # Top border
            arr[-10:, :] = 255  # Bottom border
            arr[:, 0:10] = 255  # Left border
            arr[:, -10:] = 255  # Right border
        
        return arr

    def close(self):
        pass


class RandomPolicy:
    """Random agent that picks actions randomly."""
    def __init__(self, action_space_size=6):
        self.action_space_size = action_space_size
    
    def act(self, obs):
        return random.randint(0, self.action_space_size - 1)
    
    def info(self):
        return {'type': 'random', 'name': 'RandomAgent'}


class OvercookedWrapper:
    """Wrapper around OvercookedMaker to provide a simple single-agent interface.
    
    The human player is player_0, the AI partner is player_1.
    """
    def __init__(self, overcooked_env, ai_policy=None):
        self.env = overcooked_env
        self.ai_policy = ai_policy or RandomPolicy()
        self.players = overcooked_env.players  # Should be ['player_0', 'player_1']
        self.human_player = self.players[0]
        self.ai_player = self.players[1]
        self.last_obs = None
        self.cumulative_reward = 0
        
    def reset(self):
        """Reset environment and return initial observation for human."""
        from gym_cooking import Arrdict
        data = self.env.reset()
        self.last_obs = data
        self.cumulative_reward = 0
        return data[self.human_player].obs
    
    def step(self, human_action):
        """Step with human action, AI acts automatically.
        
        Args:
            human_action: int, action for human player (0-5)
            
        Returns:
            obs: observation for human player
            reward: float, reward for human player
            done: bool, episode done
            info: dict, info
        """
        from gym_cooking import Arrdict
        
        # Get AI action
        if self.last_obs is not None:
            ai_obs = self.last_obs[self.ai_player].obs
        else:
            ai_obs = None
        ai_action = self.ai_policy.act(ai_obs)
        
        # Create decision with both actions
        decision = Arrdict(action={
            self.human_player: human_action,
            self.ai_player: ai_action
        })
        
        # Step environment
        data, info = self.env.step(decision)
        self.last_obs = data
        
        # Extract human player's data
        human_data = data[self.human_player]
        obs = human_data.obs
        reward = float(human_data.reward)
        done = bool(human_data.done)
        
        self.cumulative_reward += reward
        
        # Add AI action to info for logging
        info_dict = dict(info) if hasattr(info, '__dict__') else {}
        info_dict['ai_action'] = ai_action
        info_dict['human_reward'] = reward
        info_dict['cumulative_reward'] = self.cumulative_reward
        
        return obs, reward, done, info_dict
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


class EnvState:
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.lock = threading.Lock()
        self.step = 0


class EnvManager:
    def __init__(self):
        self._envs = {}
        self._by_session = {}
        self._counter = 0

    def _new_env_id(self):
        self._counter += 1
        return f'env-{int(time.time())}-{self._counter}'

    def create_env_for_session(self, session_id: str, level: Optional[str] = None, model: Optional[str] = None, demo: bool = True):
        """Create an environment and attach model policy (if available).

        - For demo=True or if import fails: uses MockEnv
        - For demo=False: creates real Overcooked environment with AI partner
        - AI partner is random by default, or trained model if `model` provided
        """
        env = None
        ai_policy = None
        
        # Try to create real Overcooked environment (not for demo)
        if not demo:
            try:
                from environment.overcooked.overcooked_maker import OvercookedMaker
                
                # Load config - use the same config as training (forced_coord_1_to_1_full.yaml)
                config_path = Path(__file__).parent.parent / 'environment' / 'overcooked' / 'config' / 'forced_coord_1_to_1_full.yaml'
                
                if config_path.exists():
                    import yaml
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    
                    # Use available level from config
                    mode = config.get('mode', '1to1_env_3_divider')
                    
                    # Import cooking_zoo directly to bypass problematic wrappers
                    from gym_cooking.environment.cooking_zoo import CookingEnvironment
                    from gym_cooking.environment.game.graphic_pipeline import GraphicPipeline
                    from gym_cooking import Dotdict
                    from pettingzoo.utils import wrappers
                    
                    # Create environment with training config parameters
                    env_init = CookingEnvironment(
                        level=mode,
                        num_agents=config.get('num_agents', 2),
                        record=False,
                        max_steps=200,  # 10 seconds at 20Hz action rate
                        recipes=config.get('recipes', []),
                        desire=[1] * 6,  # Default desire for all ingredients
                        obs_spaces=[config.get('obs_spaces', 'dense')],  # Must be a list
                        obs_range=config.get('obs_range', None),  # Use obs_range from config
                        interact_reward=config.get('interact_reward', 0.0),
                        progress_reward=config.get('progress_reward', 0.0),
                        complete_reward=config.get('complete_reward', 10.0),
                        step_cost=config.get('step_cost', 0.05)
                    )
                    
                    # Patch reset BEFORE wrapping so wrappers use patched version
                    original_reset = env_init.reset
                    def patched_reset(seed=None, options=None):
                        return original_reset()
                    env_init.reset = patched_reset
                    
                    # Create graphic pipeline for rendering
                    graphic_pipeline = GraphicPipeline(env_init, display=False)
                    
                    # Add render_mode attribute to avoid wrapper issues
                    env_init.render_mode = None
                    
                    # Set metadata to allow parallel conversion
                    if not hasattr(env_init, 'metadata'):
                        env_init.metadata = {}
                    env_init.metadata['is_parallelizable'] = True
                    
                    # Wrap with essential wrappers only (skip CaptureStdoutWrapper)
                    env_init = wrappers.AssertOutOfBoundsWrapper(env_init)
                    env_init.metadata['is_parallelizable'] = True
                    env_init = wrappers.OrderEnforcingWrapper(env_init)
                    env_init.metadata['is_parallelizable'] = True
                    
                    # Use AEC environment directly (don't convert to parallel)
                    # Wrap in our custom interface that handles multi-agent coordination
                    class AECOvercookedWrapper:
                        def __init__(self, aec_env, graphics, ai_policy):
                            self._env = aec_env
                            self.graphic_pipeline = graphics
                            self.ai_policy = ai_policy
                            self.players = list(aec_env.possible_agents)
                            self.human_player = self.players[0]  # player_0
                            self.ai_player = self.players[1]  # player_1
                            self.episode_start_time = None
                            self.time_limit_seconds = 600  # 10 minutes
                            self.steps_taken = 0
                            # Initialize graphics
                            self.graphic_pipeline.on_init()
                            
                        def reset(self):
                            import time as time_module
                            self._env.reset()
                            self.episode_start_time = time_module.time()
                            self.steps_taken = 0
                            # Reset AI policy history
                            if self.ai_policy and hasattr(self.ai_policy, 'reset_history'):
                                self.ai_policy.reset_history()
                            # Get initial observation for human
                            return self._env.observe(self.human_player)
                        
                        def step(self, human_action):
                            """Step with human action, AI acts automatically."""
                            import time as time_module
                            
                            self.steps_taken += 1
                            
                            # Check time limit
                            time_up = False
                            if self.episode_start_time:
                                elapsed = time_module.time() - self.episode_start_time
                                if elapsed >= self.time_limit_seconds:
                                    time_up = True
                            
                            # Alternate between agents in AEC environment
                            # Get current agent
                            agent = self._env.agent_selection
                            
                            if agent == self.human_player:
                                # Human's turn
                                self._env.step(human_action)
                            else:
                                # AI's turn - get action from policy
                                ai_obs = self._env.observe(self.ai_player)
                                ai_action = self.ai_policy.act(ai_obs)
                                print(f"🤖 AI taking action: {ai_action}")
                                self._env.step(ai_action)
                            
                            # Advance to next agent if needed
                            if not self._env.dones.get(self.human_player, False):
                                # Check if we need to do AI move
                                if self._env.agent_selection == self.ai_player:
                                    ai_obs = self._env.observe(self.ai_player)
                                    ai_action = self.ai_policy.act(ai_obs)
                                    print(f"🤖 AI taking additional action: {ai_action}")
                                    self._env.step(ai_action)
                            
                            # Get state for human player
                            obs = self._env.observe(self.human_player)
                            reward = self._env.rewards.get(self.human_player, 0)
                            done = self._env.dones.get(self.human_player, False) or time_up
                            
                            info = {
                                'elapsed_time': time_module.time() - self.episode_start_time if self.episode_start_time else 0,
                                'steps_taken': self.steps_taken,
                                'cumulative_reward': reward
                            }
                            if time_up:
                                info['termination_reason'] = 'time_limit'
                            
                            return obs, reward, done, info
                        
                        def render(self, mode='rgb_array'):
                            """Render using the graphic pipeline."""
                            try:
                                img = self.graphic_pipeline.on_render(mode)
                                if img is not None and hasattr(img, 'shape') and len(img.shape) == 3:
                                    return img
                                # Fallback
                                h, w = 700, 1300
                                arr = np.zeros((h, w, 3), dtype=np.uint8)
                                arr[:, :] = [40, 120, 40]
                                return arr
                            except Exception as e:
                                print(f"⚠ Render error: {e}")
                                h, w = 700, 1300
                                arr = np.zeros((h, w, 3), dtype=np.uint8)
                                arr[:, :] = [40, 120, 40]
                                return arr
                        
                        def close(self):
                            if hasattr(self._env, 'close'):
                                self._env.close()
                    
                    # Get AI policy (random or trained model) BEFORE creating wrapper
                    if model:
                        try:
                            from webapp.models import REGISTRY
                            ai_policy = REGISTRY.get_policy(model)
                            print(f"✓ Loaded trained model: {model}")
                        except Exception as e:
                            print(f"⚠ Failed to load model {model}: {e}")
                            ai_policy = RandomPolicy()
                            print(f"✓ Using RandomPolicy instead")
                    else:
                        ai_policy = RandomPolicy()
                        print(f"✓ Using RandomPolicy for AI partner")
                    
                    # Now create wrapper with initialized AI policy
                    # AECOvercookedWrapper already provides single-agent interface
                    env = AECOvercookedWrapper(env_init, graphic_pipeline, ai_policy)
                    print(f"✓ Created Overcooked environment: {config.get('mode')}")
                else:
                    print(f"⚠ Config not found: {config_path}")
                    env = None
                    
            except Exception as e:
                print(f"⚠ Failed to create Overcooked environment: {e}")
                import traceback
                traceback.print_exc()
                env = None

        # Fallback to MockEnv for demo or if creation failed
        if env is None:
            env = MockEnv()
            if demo:
                print(f"✓ Using MockEnv for demo session {session_id}")
            else:
                print(f"✓ Using MockEnv fallback for session {session_id}")

        # Reset environment
        try:
            env.reset()
        except Exception as e:
            print(f"⚠ Reset failed: {e}")

        env_id = self._new_env_id()
        # Store ai_policy if env has it (AECOvercookedWrapper or OvercookedWrapper)
        self._envs[env_id] = EnvState(env, model=ai_policy if hasattr(env, 'ai_policy') else None)
        self._by_session[session_id] = env_id
        return env_id

    def get_env_id(self, session_id: str):
        return self._by_session.get(session_id)

    def step(self, env_id: str, action):
        state = self._envs.get(env_id)
        if state is None:
            return {'step': None, 'reward': 0, 'done': True, 'summary': {'error': 'no env'}}
        with state.lock:
            try:
                obs, reward, done, info = state.env.step(action)
            except Exception as e:
                print(f"⚠ Step error: {e}")
                import traceback
                traceback.print_exc()
                traceback.print_exc()
                return {'step': state.step, 'reward': 0, 'done': True, 'summary': {'error': str(e)}}
            state.step += 1
            result = {'obs': obs, 'reward': reward, 'done': done, 'info': info, 'step': state.step}
            if done:
                result['summary'] = {'steps': state.step, 'total_reward': reward}
            return result

    def get_frame(self, env_id: str):
        state = self._envs.get(env_id)
        if state is None:
            return ('', {})
        with state.lock:
            try:
                arr = state.env.render(mode='rgb_array')
            except Exception:
                arr = None
            if arr is None:
                # return a blank 1x1 pixel to avoid client errors
                return ('', {'step': state.step, 'model': state.model.info() if state.model else None})
            if Image is None:
                return ('', {'step': state.step, 'model': state.model.info() if state.model else None})
            im = Image.fromarray(arr)
            buf = io.BytesIO()
            im.save(buf, format='PNG')
            b = buf.getvalue()
            b64 = base64.b64encode(b).decode('ascii')
            info = {'step': state.step, 'model': state.model.info() if state.model else None}
            return (b64, info)

    def reset(self, env_id: str):
        state = self._envs.get(env_id)
        if state:
            with state.lock:
                try:
                    state.env.reset()
                except Exception:
                    pass
                state.step = 0

    def cleanup_env(self, env_id: str):
        state = self._envs.pop(env_id, None)
        if state:
            try:
                state.env.close()
            except Exception:
                pass
            to_delete = [s for s, e in self._by_session.items() if e == env_id]
            for s in to_delete:
                del self._by_session[s]

    def get_model_info_for_session(self, session_id: str):
        env_id = self._by_session.get(session_id)
        if not env_id:
            return None
        state = self._envs.get(env_id)
        if not state or not state.model:
            return None
        return state.model.info()
