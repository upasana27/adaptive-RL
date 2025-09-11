import gymnasium as gym
from gym import spaces
import numpy as np

class KuhnPoker_SingleEnv(gym.Env):
    def __init__(self):
        # self id is 0(player 1), opponent id is 1(player 2)
        # obs include the card and the previous action of the opponent
        self.id = 0
        self.oppo_id = 1
        self.observation_space = spaces.Box(low=0,high=1,shape=(6,)) 
        self.action_space = spaces.Discrete(2)
        self.opponent = self.random_policy
        self.dealt_card = None
        self.win_result = None
        self.prev_action = None
        self.stage = 0

    def step(self,action):
        if self.stage == 0:
            self.stage+=1
            self.prev_action[self.id]=action
            oppo_obs = self.get_obs(self.oppo_id)
            oppo_action = self.opponent(oppo_obs)
            self.prev_action[self.oppo_id] = oppo_action
            info = {"opponent_obs":oppo_obs, "opponent_act":oppo_action}
            if action==0 and oppo_action==0:
                #check check
                reward = self.win_result[self.id] * 1
                obs = np.full(6, np.nan) #self.get_obs(self.id)
                done = True
            elif action==0 and oppo_action==1:
                #check bet
                reward = 0
                obs = self.get_obs(self.id)
                done = False
            elif action==1 and oppo_action==0:
                #bet fold
                reward = 1
                obs = np.full(6, np.nan) #self.get_obs(self.id)
                done = True
            else:
                #bet call
                reward = self.win_result[self.id] * 2
                obs = np.full(6, np.nan)
                done = True
            return obs,reward,done,info
        else:
            obs = np.full(6, np.nan)  # self.get_obs(self.id)
            done = True
            info = {}
            if action==0:
                # check bet fold
                reward = -1
            else:
                # check bet call
                reward = self.win_result[self.id] * 2
            return obs,reward,done,info
    
    def reset(self):
        self.stage = 0
        self.prev_action= np.zeros(2)
        self.deal_card()
        return self.get_obs(self.id)
    
    def get_obs(self,id):
        idx = self.prev_action[1-id]*3+self.dealt_card[id]
        obs = np.zeros(6)
        obs[int(idx)] = 1.0
        return obs

    def set_opponent(self, policy):
        self.opponent = policy

    def deal_card(self):
        all_card = [0,1,2]
        np.random.shuffle(all_card)
        self.dealt_card = all_card[:2]
        win = 2*(self.dealt_card[0] > self.dealt_card[1])-1  # 1 or -1
        self.win_result = np.array([win,-win])
        

    def random_policy(self, obs):
        return np.random.randint(2)