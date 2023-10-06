import numpy as np 
import torch as T 
from utils import convert_arrays_to_tensors

class Agent:
    def __init__(self, memory, policy, gamma=0.99, tau=0.001):
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.tau = tau
        self.networks = []

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state,action, reward, state_,done)

    def sample_memory(self):
        memory_batch = self.memory.sample_buffer()
        memory_tensors = convert_arrays_to_tensors(memory_batch, self.device)
        states, actions, rewards, states_, dones = memory_tensors
        return states,actions,rewards,states_,dones
    
    def save_models(self):
        for network in self.networks:
            for x in network: 
                x.save_checkpoint()
    
    def load_models(self):
        for network in self.networks:
            for x in network:
                x.load_checkpoint()

    def update_network_parameters(self, src, dest, tau=None):
        if tau is None:
            tau = self.tau
        
        for param, target in zip(src.parameters(), dest.parameters()):
            target.data.copy_(tau*param.data + (1-tau)*target.data)

        def update(self):
            raise(NotImplementedError)
        
        def choose_action(self, observation):
            raise(NotImplementedError)
        
class DQN(Agent):
    def __init__(self, network, memory, policy, gamma =0.99, lr=1e-4, replace=1000):
        super().__init__(memory,policy, gamma)
        self.replace_target_cnt = replace
        self.learn_step_counter = 0 

        self.q_eval = network
        self.q_next = network

        self.networks = [net for net in [self.q_eval, self.q_next]]
        
        self.optimizer = T.optim.Adam(self.q_eval.parameters(), lr=lr)
        self.loss = T.nn.MSELoss()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        q_values = self.q_eval(state)
        action = self.policy(q_values)
        return action
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.update_network_parameters(self.q_eval, self.q_next, tau=1.0)

    def update(self):
        if not self.memory.ready():
            return
        
        self.optimizer.zero_grad()

        self.replace_target_network

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(len(states))

        q_pred = self.q_eval.forward(states_).max(dim=1)[0]
        
        q_next = self.q_next(states_).max(dim=1)[0]
        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter +=1

        




    