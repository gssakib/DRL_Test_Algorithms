import numpy as np 

class ReplayBuffer:
    def __init__(self, max_size, input_shape, batch_size, n_actions=None, action_space ='discrete'):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)

        if action_space == 'discrete':
            self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        elif action_space == 'continuous':
            assert n_actions is not None
            self.action_memory = np.zeros((self.mem_size, n_actions),dtype=np.float32)

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_) 

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.new_state_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return [states, actions, states_, terminal]
    
    def ready(self):
        return self.mem_cntr>= self.batch_size
    


