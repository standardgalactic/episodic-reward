import torch
import torch.nn as nn
import numpy as np


class Q_CNN(nn.Module):
    def __init__(self,env,device):
        super(Q_CNN, self).__init__()

        self.main = nn.Sequential(           
            nn.Conv1d(1,4,8),
            nn.ReLU(),
            nn.Conv1d(4,8,8),
            nn.ReLU(),    
            nn.Conv1d(8,12,8),
            nn.ReLU()   
        )
        self.fc = nn.Sequential(
            nn.Linear(12 * (env.observation_space.shape[0] - ((8-1) * 3)), 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        self.device = device


    def forward(self, s):
        x = self.main(s.float())
        return self.fc(x.view(x.size(0), -1))


class Q_FC(nn.Module):
    def __init__(self,env, device):
        super(Q_FC, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.n)
        )
        self.device = device

    def forward(self, s): 
        return self.main(s.float())

class Q_Single(nn.Module):
    def __init__(self,env, device):
        super(Q_Single, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 1040),
            nn.ReLU(),
            nn.Linear(1040, env.action_space.n)
        )
        self.device = device

    def forward(self, s): 
        return self.main(s.float())
