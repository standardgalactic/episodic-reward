import gym
from gym.spaces import Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy

from model import *
from replay_buffer import *
from visualize import *
from util import *

algo_name = 'DQN'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env = gym.make('SpaceInvaders-v0')
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.ResizeObservation(env, (84,84))
#env = gym.wrappers.AtariPreprocessing(env,frame_skip=1)
env = PyTorchObservation(env)
env = gym.wrappers.FrameStack(env, 4)

"""
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.ResizeObservation(env, (84,84))
env = ImageToPyTorch(env)
print(env.observation_space,shape)
quit()
"""

epsilon = .01
gamma = .99
#Proportion of network you want to keep
tau = .995
random.seed(5714149178)

q = Q_CNN(env)
q_target = deepcopy(q)
q_target = q_target

optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
max_ep = 1000

batch_size = 128
rb = ReplayBuffer(1e6)

#Training the network
def train():
    explore(1000)
    ep = 0
    while ep < max_ep:
        s = env.reset()
        ep_r = 0
        while True:
            with torch.no_grad():
                #Epsilon greed exploration
                if random.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    q, a = int(torch.max(q(torch.tensor(np.array(s)))[0],dim=-1)[1])
            #Get the next state, reward, and info based on the chosen action
            s2, r, done, _ = env.step(int(a))
            rb.store(s, a, r, s2, done)
            ep_r += r

            #If it reaches a terminal state then break the loop and begin again, otherwise continue
            if done:
                update_viz(ep, ep_r, algo_name)
                ep += 1
                break
            else:
                s = s2

            update()


#Updates the Q by taking the max action and then calculating the loss based on a target
def update():
    s, a, r, s2, m = rb.sample(batch_size)

    with torch.no_grad():
        max_next_q, _ = q_target(torch.unsqueeze(s2,0)).max(dim=1, keepdim=True)
        y = r + m*gamma*max_next_q
    loss = F.mse_loss(torch.gather(q(s), 1, a.long()), y)

    #Update q
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Update q_target
    for param, target_param in zip(q.parameters(), q_target.parameters()):
        target_param.data = target_param.data*tau + param.data*(1-tau)

#Explores the environment for the specified number of timesteps to improve the performance of the DQN
def explore(timestep):
    ts = 0
    while ts < timestep:
        s = env.reset()
        while True:
            a = env.action_space.sample()
            s2, r, done, _ = env.step(int(a))
            rb.store(s, a, r, s2, done)
            ts += 1
            if done:
                break
            else:
                s = s2


train()
