import gymnasium as gym
from gym import spaces
import numpy as np
import torch
import random

class KuhnPoker_SingleEnv(gym.Env):
    def __init__(self):
        # self id is 0(player 1), opponent id is 1(player 2)
        # obs: [stage : 7-dim one-hot, self_card: 3-dim one hot, oppo_card: 3-dim one hot] 
        # stage 0: player-1 first action / player-2 faces Check
        # stage 1: player-1 Check and faces Bet / player-2 faces Bet
        # stage 2: Check Check terminal state
        # stage 3: Check Bet Fold terminal state
        # stage 4: Check Bet Call terminal state
        # stage 5: Bet Fold terminal state
        # stage 6: Bet Call terminal state
        self.id = 0
        self.oppo_id = 1
        self.observation_space = spaces.Box(low=0,high=1,shape=(13,)) # 7 + 3 + 3
        self.action_space = spaces.Discrete(2)
        self.opponent = self.random_policy
        self.dealt_card = None
        self.win_result = None
        self.prev_action = None
        self.stage = 0 # self stage, id=0

    def seed(self, seed=None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def episode_length():
        return 2

    @staticmethod
    def get_reward_scale():
        return 2.0

    def step(self,action):
        if self.stage == 0:
            self.prev_action[self.id]=action
            oppo_obs = self.get_obs(self.oppo_id)
            oppo_action = self.opponent(oppo_obs)
            self.prev_action[self.oppo_id] = oppo_action
            info = {"opponent_obs":oppo_obs, "opponent_act":oppo_action}
            self.stage_transition()
            reward = self.compute_reward()
            obs = self.get_obs(self.id)
            done = (self.stage > 1)
        elif self.stage == 1:
            self.prev_action[self.id] = action
            self.stage_transition()
            obs = self.get_obs(self.id)
            reward = self.compute_reward()
            done = True
            info = {}
        else:
            raise ValueError("Invalid stage")
        if done:
            info['showdown'] = int(self.stage in [2, 4, 6])
        return obs,reward,done,info
    
    def reset(self):
        self.stage = 0
        self.prev_action= np.array([-1,-1])
        self.deal_card()
        return self.get_obs(self.id)
    
    def stage_transition(self):
        if self.stage == 0:
            if self.prev_action[self.id] == 0 and self.prev_action[self.oppo_id] == 0:
                # check check
                self.stage = 2
            elif self.prev_action[self.id] == 0 and self.prev_action[self.oppo_id] == 1:
                # check bet
                self.stage = 1
            elif self.prev_action[self.id] == 1 and self.prev_action[self.oppo_id] == 0:
                # bet fold
                self.stage = 5
            elif self.prev_action[self.id] == 1 and self.prev_action[self.oppo_id] == 1:
                # bet call
                self.stage = 6
            else:
                raise ValueError("Wrong Transition")
        elif self.stage == 1:
            if self.prev_action[self.id] == 0 and self.prev_action[self.oppo_id] == 1:
                # check Bet Fold
                self.stage = 3
            elif self.prev_action[self.id] == 1 and self.prev_action[self.oppo_id] == 1:
                # check bet call
                self.stage = 4
            else:
                raise ValueError("Wrong Transition")
            
    def get_obs(self,id):
        obs = np.zeros(13)
        # stage
        if id == 0:
            #self observation
            obs[int(self.stage)] = 1.0
        else:
            #opponent observation
            if self.stage == 0:
                # opponent faces check or bet
                obs[int(self.prev_action[1-id])] = 1.0
            else:
                raise ValueError("No need to return opponent observation if not at stage 0")
        # self card
        obs[7+int(self.dealt_card[id])] = 1.0

        if self.stage in [2,4,6]:
            # showdown, opponent's card is revealed
            obs[10+int(self.dealt_card[self.oppo_id])] = 1.0

        return obs

    def set_opponent(self, policy):
        self.opponent = policy

    def compute_reward(self):
        if self.stage == 2:
            # check check showdown
            return self.win_result[self.id]
        elif self.stage == 4 or self.stage == 6:
            # check bet call/bet call showdown
            return 2*self.win_result[self.id]
        elif self.stage == 3:
            # check bet fold
            return -1
        elif self.stage == 5:
            # bet fold
            return 1
        else:
            # not terminal state
            return 0

    def deal_card(self):
        all_card = [0,1,2]
        np.random.shuffle(all_card)
        self.dealt_card = all_card[:2]
        win = 2*(self.dealt_card[0] > self.dealt_card[1])-1  # 1 or -1
        self.win_result = np.array([win,-win])
        

    def random_policy(self, obs):
        return np.random.randint(2)