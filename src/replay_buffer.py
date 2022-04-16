import numpy as np
import torch
import random
from collections import deque


class ReplayBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=int(size))
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_arr = torch.FloatTensor(np.array([arr[0] for arr in batch]))
        a_arr = torch.FloatTensor(np.array([arr[1] for arr in batch]))
        r_arr = torch.FloatTensor(np.array([arr[2] for arr in batch]))
        s2_arr = torch.FloatTensor(np.array([arr[3] for arr in batch]))
        m_arr = torch.FloatTensor(np.array([arr[4] for arr in batch]))

        return s_arr, a_arr.unsqueeze(1), r_arr.unsqueeze(1), s2_arr, m_arr.unsqueeze(1)

    def len(self):
        return self.len

    def store(self, s, a, r, s2, d):
        def fix(x):
            if not isinstance(x, np.ndarray): return np.array(x)
            else: return x

        data = [s, np.array(a,dtype=np.float64), r, s2, 1 - d]
        transition = tuple(fix(x) for x in data)
        self.len = min(self.len + 1, self.maxSize)
        self.buffer.append(transition)
