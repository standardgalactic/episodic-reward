import torch
import torch.nn as nn
import numpy as np


class Q_CNN(nn.Module):
    def __init__(self,env):
        super(Q_CNN, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU()            
        )
        self.fc = nn.Sequential(
            nn.Linear(2592, 256),
            nn.Linear(256, env.action_space.n)
        )

    def forward(self, s):
        x = self.main(torch.FloatTensor(np.array(s)))
        return self.fc(x.view(4,-1))


class Q_FC(nn.Module):
    def __init__(self,env):
        super(Q_FC, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, env.action_space.n)
        )

    def forward(self, s):
        return self.main(torch.FloatTensor(s))
