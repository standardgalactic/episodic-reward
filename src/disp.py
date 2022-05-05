
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
import time
import pickle

from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
random.seed(5714149178)

rewards = []
def test():
    env = gym.make('Asteroids-ram-v0')
    q = Q_FC(env, device).to(device)

    q.load_state_dict(torch.load('../model/final/Asteroids-ram-v0_DQN-FC-AsteroidFinal.pt'))
    q.eval()
    max_ep = 10
    ep = 0
    while ep < max_ep:
        with torch.no_grad():
            s = env.reset()
            ep_r = 0
            while True:
                env.render()
                gp_s = torch.tensor(np.array(s, copy=False)).to(device)
                a = int(np.argmax(q(gp_s).cpu()))
                #Get the next state, reward, and info based on the chosen action
                s2, r, done, _ = env.step(int(a))
                ep_r += r

                #If it reaches a terminal state then break the loop and begin again, otherwise continue
                if done:
                    print(f'{ep}: {ep_r}')
                    ep += 1
                    break
                else:
                    s = s2
                time.sleep(.05)
    env.close()

test()