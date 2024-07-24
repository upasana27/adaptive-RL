import numpy as np
from kuhn_poker.kuhn_poker_oppo_hand import KuhnPoker_SingleEnv
from kuhn_poker.policy_new import KuhnPokerPolicy

def nash_1(obs):
    action_2 = np.argmax(obs[:7])
    card = np.argmax(obs[7:10])
    if card==0 and action_2==0:
        a=np.random.choice([0,1],1,p=[2/3,1/3])
    elif card==0 and action_2==1:
        a=0
    elif card==1 and action_2==0:
        a=0
    elif card==1 and action_2==1:
        a=np.random.choice([0,1],1,p=[1/3,2/3])
    elif card==2 and action_2==0:
        a=np.random.choice([0,1],1,p=[2/3,1/3])
    elif card==2 and action_2==1:
        a=1
    return a

def nash_2(obs):
    obs = np.argmax(obs)
    card = obs%3
    action_1 = int(obs/3)
    if card==2:
        a = 1
    elif card==1 and action_1==0:
        a = 0
    elif card==1 and action_1==1:
        a=np.random.choice([0,1],1,p=[2/3,1/3])
    elif card==0 and action_1==0:
        a=np.random.choice([0,1],1,p=[2/3,1/3])
    elif card==0 and action_1==1:
        a=0
    return a 

env = KuhnPoker_SingleEnv()
env.set_opponent(KuhnPokerPolicy([1/3,1/3]))

reward_sum=0
cnt=int(1e6)
for eps in range(cnt):
    if eps%1000==0:
        print(eps)
    obs = env.reset()
    # print("A new game")
    # print("dealt card:", env.dealt_card)
    # print("obs 1:",obs)
    #a = np.random.randint(2)
    a=nash_1(obs)
    # print("action 1:",a)
    obs,reward,done,info=env.step(a)
    reward_sum+=reward
    # print(obs,reward,done,info)
    if not done:
        #a = np.random.randint(2)
        a=nash_1(obs)
        # print("action 1+:",a)
        obs,reward,done,info=env.step(a)
        # print(obs,reward,done,info)
        reward_sum+=reward
    # print("------------------")
print(reward_sum/cnt)