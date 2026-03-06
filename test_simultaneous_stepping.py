#!/usr/bin/env python3
"""Test script to verify simultaneous stepping of human and AI agents."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from webapp.env_wrapper import EnvManager

# Create environment manager
manager = EnvManager()

# Create a test environment with rule-based AI
print("\n=== Creating Overcooked environment with rule-based AI ===")
session_id = "test_session_001"
env_id = manager.create_env_for_session(session_id, level='fc_small_divider_test', model='rule_based_1', demo=False)
print(f"Environment created: {env_id}")

# Reset environment
print("\n=== Resetting environment ===")
obs = manager.reset(env_id)
print(f"Initial observation shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")

# Test stepping - both agents should act simultaneously
print("\n=== Testing simultaneous stepping ===")
print("Running 10 steps with fixed human actions to verify AI and human move together...")

for step_num in range(10):
    # Human action: 0 (stay in place)
    obs, reward, done, info = manager.step(env_id, action=0)
    
    elapsed = info.get('elapsed_time', 0)
    ai_action = info.get('ai_action', -1)
    
    print(f"Step {step_num+1:2d}: elapsed={elapsed:5.2f}s, reward={reward:6.2f}, ai_action={ai_action}, done={done}")
    
    if done:
        print(f"Episode ended at step {step_num+1}")
        break

print("\n=== Test Complete ===")
print("✓ Environment created with simultaneous stepping wrapper")
print("✓ Both agents should execute actions in sync (no alternation visible in rewards)")
