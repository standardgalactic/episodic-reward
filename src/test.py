import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
import yaml
import sys
import datetime
import pickle

from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
random.seed(5714149178)

rewards = []
def test(model, e, out):
    env = gym.make(e)
    if 'FC' in model:
        q = Q_FC(env, device).to(device)
    elif 'CNN' in model:
        q = Q_CNN(env, device).to(device)
    elif 'Small' in model:
        q = Q_Small(env, device).to(device)
    else:
        q = Q_Single(env, device).to(device)
    q.load_state_dict(torch.load(model))
    q.eval()
    max_ep = 250
    ep = 0
    while ep < max_ep:
        with torch.no_grad():
            s = env.reset()
            ep_r = 0
            while True:
                if 'CNN' in model:
                    gp_s = torch.tensor(np.array(s, copy=False)).view(1,1,s.shape[0]).to(device)
                    a = int(np.argmax(q(gp_s).cpu()))
                else:
                    gp_s = torch.tensor(np.array(s, copy=False)).to(device)
                    a = int(np.argmax(q(gp_s).cpu()))
                #Get the next state, reward, and info based on the chosen action
                s2, r, done, _ = env.step(int(a))
                ep_r += r

                #If it reaches a terminal state then break the loop and begin again, otherwise continue
                if done:
                    rewards.append(ep_r)
                    print(f'{ep}: {ep_r}')
                    ep += 1
                    break
                else:
                    s = s2
    print(sum(rewards)/len(rewards))
    write_reward_data(out)

def write_reward_data(fname):
    with open('../data/test/' + fname, 'w') as f:
        f.write('Episodic Reward,\n')
        [f.write(str(d) + ',\n') for d in rewards]

test(sys.argv[1], sys.argv[2], sys.argv[3])