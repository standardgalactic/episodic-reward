import torch
import torch.nn as nn
import numpy as np


class Q_CNN(nn.Module):
    def __init__(self,env,device):
        super(Q_CNN, self).__init__()

        self.main = nn.Sequential(           
            nn.Conv1d(1,16,8),
            nn.ReLU(),
            nn.Conv1d(16,32,8),
            nn.ReLU()    
        )
        self.fc = nn.Sequential(
            nn.Linear(env.observation_space.shape[0] - 14, 128),
            nn.Linear(128, env.action_space.n)
        )
        self.device = device


    def forward(self, s):
        x = self.main(s.float())
        return torch.mean(self.fc(x), dim=1)


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
