import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        
    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, device="cpu"):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        res = []
        for i in range(4):
            k = np.stack(tuple(item[i] for item in batch), axis=0)
            res.append(torch.tensor(k, device=device))
        return res[0], res[1], res[2], res[3]
    
    def save(self):
        
        self.buffer = [np.array(item) for item in self.buffer]
        np.save('replay_buffer3.npy', np.array(self.buffer))

    def loads(self):
        self.buffer = np.load('replay_buffer3.npy', allow_pickle=True).tolist() 

        return self
