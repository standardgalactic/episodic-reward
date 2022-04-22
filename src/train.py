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
import yaml
import sys
import datetime

with open(sys.argv[1], "r") as f:
    config = yaml.load(f)

algo_name = config['algo_name']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env = gym.make(config['env'])

epsilon = config['model_params']['epsilon']
gamma = config['model_params']['gamma']
#Proportion of network you want to keep
tau = .995
random.seed(5714149178)


if algo_name == 'DQN-CNN':
    q = Q_CNN(env,device).to(device)
    q_target = Q_CNN(env, device).to(device)
elif algo_name == 'DQN-FC':
    q = Q_FC(env,device).to(device)
    q_target = Q_FC(env, device).to(device)
elif algo_name == 'DQN-Single':
    q = Q_Single(env,device).to(device)
    q_target = Q_Single(env, device).to(device)



optimizer = torch.optim.Adam(q.parameters(), lr=1e-5)
max_ep = config['max_ep']

batch_size = config['batch_size']
rb = ReplayBuffer(config['replay_buffer_size'])

#Training the network
def train():
    explore(config['exploration'])
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
                    if algo_name == 'DQN-CNN':
                        gp_s = torch.tensor(np.array(s, copy=False)).view(1,1,s.shape[0]).to(device)
                    else:
                        gp_s = torch.tensor(np.array(s, copy=False)).to(device)
                    a = int(np.argmax(q(gp_s).cpu()))

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
        if eq % config['save_interval'] == 0 and eq != 0:
            torch.save(q.state_dict(), config['model_save_path'] + algo_name + datetime.datetime.now().strftime("%Y-%m-%d::%H:%M:%S"))


#Updates the Q by taking the max action and then calculating the loss based on a target
def update():
    s, a, r, s2, m = rb.sample(batch_size)

    states = torch.tensor(s).to(device)
    actions = torch.tensor(a).to(device)
    rewards = torch.tensor(r).to(device)
    states2 = torch.tensor(s2).to(device)
    masks = torch.tensor(m).to(device)

    with torch.no_grad():
        max_next_q, _ = torch.max(q_target(states2.view(states2.shape[0],1,states2.shape[1])),dim=1, keepdim=True)
        y = rewards + masks*gamma*max_next_q
    loss = F.mse_loss(torch.gather(q(states.view(states.shape[0],1,states.shape[1])), 1, actions.long()), y)


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
