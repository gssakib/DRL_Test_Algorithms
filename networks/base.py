import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from .core import NetworkBase

class CriticBase(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, hidden_dims=[256,256]):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.to(self.device)

    def forward(self,sa):
        state, action = sa
        x = T.cat([state,action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class LinearBase(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, hidden_dims=[256]):
       super().__init__(name=name, chkpt_dir=chkpt_dir)
       self.fc1 = nn.Linear(*input_dims, hidden_dims[0])
       self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
       self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
       self.to(self.device)

    def forward(self,state):
        f1 = F.relu(self.fc(state))
        f2 = F.relu(self.fc2(f1))

class AtariBase(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, channels = (32,64,64), kernels =(8,4,3), strides = (4,2,1)):
        super().__init__(name=name, chkpt_dir=chkpt_dir)

        assert len(channels) == 3
        assert len(kernels) == 3
        assert len(strides) == 3

        self.conv1 = nn.Conv2d(input_dims[0], channels[0], kernels[0], strides[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernels[1], strides[1])
        self.conv3 = nn.Conv3d(channels[1],channels[2],kernels[2],strides[2])
        self.flat = nn.Flatten()
        self.device = T.device('cuda: 0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv2(conv2))
        conv_state = F.flat(conv3)

        return conv_state  

