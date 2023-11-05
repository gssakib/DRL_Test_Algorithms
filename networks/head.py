import torch as T
import torch.nn as nn
import torch.nn.functional as F
from .core import NetworkBase
 
class QHead(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, n_actions, hidden_layers=[512]):
        super().__init__(name=name, chkpt_dir = chkpt_dir)
        assert len(hidden_layers) == 1

        self.fc1 = nn.Linear(*input_dims, hidden_layers[0])
        self.fc2= nn.Linear(hidden_layers[0], n_actions)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        f1 = F.relu(self.fc1(state))
        q_values = self.fc2(f1)

        return q_values
    
class DeterministicHead(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, n_actions):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.fc1 - nn.Linear(*input_dims, n_actions)

        self.to(self.device)   

    def forward(self, x):
        mu = T.tanh(self.fc1(x))

        return mu

class ValueHead(NetworkBase, nn.Module):
    def __init__(self, name,chkpt_dir, input_dims):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.v = nn.Linear(*input_dims, 1)
        self.to(self.device)

    def forward(self, x):
        value = self.v(x)

        return value
    

class MuAndSigmaHead(NetworkBase, nn.Module):
    def __init__(self, name, chkpt_dir, input_dims, n_actions, std_min=1e-6):
        super().__init__(name=name, chkpt_dir=chkpt_dir)
        self.std_min = std_min
        self.mu = nn.Linear(*input_dims, n_actions)
        self.sigma = nn.Linear(*input_dims, n_actions)
        self.to(self.device)

    def forward(self, x):
        mu = self.mu(x)
        sigma = T.clamp(sigma, min = self.std_min, max=1)

        return mu,sigma
     

