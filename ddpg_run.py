import numpy as np 
import gym 
from torch.nn import Sequential
from agents.ddpg import DDPG as Agent
from loops.episodes import EpisodeLoop
from memory.memory import ReplayBuffer
from networks.base import LinearBase, CriticBase
from networks.head import DeterministicHead, ValueHead
from policy.noisy_deteministic import NoisyDeterministicPolicy

def main():
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)
    n_games = 1000
    bs = 64

    actor_base = LinearBase(name='ddpg_actor_base', hidden_dims=[400, 300], input_dims=env.observation_space.shape, chkpt_dir='models')

    actor_head = DeterministicHead(n_actions=env.action_space.shape[0], input_dims=[300],name='ddpg_actor_head', chkpt_dir='models')
    
    actor = Sequential(actor_base, actor_head)

    target_actor_base = LinearBase(name='ddpg_target_actor_base', hidden_dims=[400, 300], input_dims=env.observation_space.shape, chkpt_dir='models')

    target_actor_head = DeterministicHead(n_actions=env.action_space.shape[0], input_dims=[300],name='ddpg_target_actor_head', chkpt_dir='models')

    target_actor = Sequential(target_actor_base, target_actor_base)

    critic_base = CriticBase(name='ddpg_critic_base', input_dims= input_dims,hidden_dims=[400, 300], chkpt_dir='models')

    critic_head = ValueHead(name='ddpg_critic_head', input_dims=[300],chkpt_dir='models')

    critic = Sequential(critic_base,critic_head)

    target_critic_base = CriticBase(name='ddpg_target_critic_base', input_dims=input_dims,hidden_dims=[400, 300], chkpt_dir='models')

    target_critic_head = ValueHead(name='ddpg_target_critic_head', input_dims=[300],chkpt_dir='models')

    target_critic = Sequential(target_critic_base,target_critic_head)

    memory = ReplayBuffer(max_size=100_000, 
                          input_shape=env.observation_space.shape, 
                          batch_size=bs, 
                          n_actions=env.action_space=.shape[0], 
                          action_space='continuous')
    
    policy = NoisyDeterministicPolicy(n_actions=env.action_space.shape[0],
                                      min_action=env.action_space.low[0],
                                      max_action=env.action_space.high[0])
    
    agent= Agent(actor, critic, target_actor, target_critic, memory, policy)
    ep_loop = EpisodeLoop(agent, env)
    scores, steps_array = ep_loop.run(n_games)

if __name__ == '__main__':
    main()



