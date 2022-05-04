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
from replay_buffer import *
from visualize import *

device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def test(model, e, out):
    with torch.no_grad():
        rewards = []
        env = gym.make(e)
        if 'FC' in model:
            q = Q_FC(env, device)
        elif 'CNN' in model:
            q = Q_CNN(env, device)
        else:
            q = Q_Single(env, device)
        q.load_state_dict(torch.load(model))
        q.eval()
        max_ep = 100
        ep = 0
        while ep < max_ep:
            s = env.reset()
            while True:
                a = int(np.argmax(q(s)))
                #Get the next state, reward, and info based on the chosen action
                s2, r, done, _ = env.step(int(a))
                ep_r += r

                #If it reaches a terminal state then break the loop and begin again, otherwise continue
                if done:
                    rewards.append(ep_r)
                    ep += 1
                    break
                else:
                    s = s2
        write_reward_data(out)

def write_reward_data(fname):
    with open('../data/test/' + fname, 'w') as f:
        f.write('Episodic Reward,\n')
        [f.write(str(d) + ',\n') for d in data]

test(sys.argv[1], sys.argv[2], sys.argv[3])