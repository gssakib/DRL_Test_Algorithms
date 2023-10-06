import numpy as np 
import gym 
from torch.nn import Sequential
from agents import DQN as Agent
from loops import EpisodeLoop
from memory import ReplayBuffer
from networks import LinearBase, QHead
from policy import EpsilonGreedyPolicy

def main():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    n_games = 250
    bs = 64

    base = LinearBase(name='q_base', input_dims = env.observation_space.shape, chkpt_dir='models')
    head = QHead(n_actions=env.action_space.n, input_dims=[256],name='q_head', chkpt_dir='models')
    actor = Sequential(base, head)

    memory = ReplayBuffer(max_size=100_00, input_shape=env.observation_shape, batch_size=bs, action_space='discrete')
    policy = EpsilonGreedyPolicy(n_actions=env.action_space.n,eps_dec=1e-4)
    agent= Agent(actor, memory, policy)
    ep_loop = EpisodeLoop(agent, env)
    scores, steps_array = ep_loop.run(n_games)

if __name__ == '__main__':
    main()



