import pybullet_envs
import gym 
from torch.nn import Sequential
from agents.sac import SACAgent as Agent
from loops.episodes import EpisodeLoop
from memory.memory import ReplayBuffer
from networks.base import LinearBase, CriticBase
from networks.head import MuAndSigmaHead, ValueHead
from policy.gaussian import GaussianPolicy

def main():
    env_name = 'InvertedPendulumBulletEnv-v1'
    env = gym.make(env_name)
    n_games = 1500
    bs = 256

    actor_base = LinearBase(name='sac_actor_base', input_dims = env.observation_space.shape, chkpt_dir='models')
    actor_head = MuAndSigmaHead(n_actions=env.action_space.n, input_dims=[256],name='sac_actor_head', chkpt_dir='models')
    actor = Sequential(actor_base, actor_head)

    input_dims = [env.observation_space.shape[0]+env.action_space.shape[0]]
    critic_base_1 = CriticBase(name='sac_critic_1_base', input_dims=input_dims,chkpt_dir='models')
    critic_head_1 = ValueHead(name='sac_critic_1_head', input_dims=[256], chkpt_dir='models')
    critic_1 = Sequential(critic_base_1, critic_head_1)
    critic_base_2 = CriticBase(name='sac_critic_2_base', input_dims=input_dims, chkpt_dir='models')
    critic_head_2 = ValueHead(name='sac_critic_2_head', input_dims=[256], chkpt_dir='models')
    critic_2 = Sequential(critic_base_2, critic_head_2)

    value_base = LinearBase(name='sac_value_base', hidden_dims=[256,256], input_dims=env.observation_space.shape, chkpt_dir='models')
    value_head = ValueError(name='sac_value_head', input_dims=[256], chkpt_dir='models')
    value = Sequential(value_base,value_head)

    target_value_base = LinearBase(name='sac_target_value_base', hidden_dims=[256,256], input_dims=env.observation_space.shape, chkpt_dir='models')
    target_value_head = ValueError(name='sac_target_value_head', input_dims=[256], chkpt_dir='models')
    target_value = Sequential(target_value_base,target_value_head)
    
    memory = ReplayBuffer(max_size=100_000, input_shape=env.observation_space.shape, batch_size=bs, n_actions=env.action_space.shape[0], action_space='continuous')
    policy = GaussianPolicy(n_actions=env.action_space.shape[0], min_action = env.action_space.low[0], max_action= env.action_space.high[0])
    agent= Agent(actor, critic_1, critic_2, value, target_value, memory, policy)
    ep_loop = EpisodeLoop(agent, env)
    scores, steps_array = ep_loop.run(n_games)

if __name__ == '__main__':
    main()



