import sys
sys.path.insert(0,'..')
from opcua import Client
import time
from collections import deque
import numpy as np 
import torch as T
from torch.nn import Sequential
from agents.ddpg import DDPG as Agent
from loops.episodes import EpisodeLoop
from memory.memory import ReplayBuffer
from networks.base import LinearBase, CriticBase
from networks.head import DeterministicHead, ValueHead
from policy.noisy_deteministic import NoisyDeterministicPolicy

def main():
    client = Client("opc.tcp://DESKTOP-A9QNR1L:4990/FactoryTalkLinxGateway1")
    
    try:
        client.connect()

        #importing model
        # model = T.import_ir_module('')

        root = client.get_root_node()

        setpoint = client.get_node("ns=2;s=[opc_server]setpoint_diameter")
        diameter = client.get_node("ns=2;s=[opc_server]Diameter")
        setpoint_speed = client.get_node("ns=2;s=[opc_server]DC_output")

        setpoint_data = deque(maxlen=32)
        diameter_data = deque(maxlen=32)

        setpoint_data.append(np.zeros(32))
        diameter_data.append(np.zeros(32))

        while(True):
            try:
                d = diameter.get_value()
                s = setpoint.get_value()

                setpoint_data.append(s)
                diameter_data.append(d)

                #Checking the shape of the input data
                """
                setpoint_array = np.array(setpoint_data).reshape(-1, 1)
                print('The shape of the set point array is ', setpoint_array.shape())
                diameter_array = np.array(diameter_data).reshape(-1, 1)
                print('The shape of the set point array is ', diameter_array.shape())
                """
                #normalize
                d_means = 0.42954475
                s_means = 0.40073333
                d_std = 0.13086448
                s_std = 0.16299119 

                d_normalized = (diameter_data-d_means)/d_std
                s_normalized = (setpoint_data-s_means)/s_std 

                #Inputting memory array into DDPG model
                #---------------Needs Review------------------------
                
                actor_base = LinearBase(name='ddpg_actor_base', hidden_dims=[400, 300], input_dims=[d_normalized.shape, s_normalized.shape], chkpt_dir='models')

                actor_head = DeterministicHead(n_actions=[0], input_dims=[300],name='ddpg_actor_head', chkpt_dir='models')
                
                actor = Sequential(actor_base, actor_head)

                target_actor_base = LinearBase(name='ddpg_target_actor_base', hidden_dims=[400, 300], input_dims=[d_normalized.shape, s_normalized.shape], chkpt_dir='models')

                target_actor_head = DeterministicHead(n_actions=[0], input_dims=[300],name='ddpg_target_actor_head', chkpt_dir='models')

                target_actor = Sequential(target_actor_base, target_actor_base)

                critic_base = CriticBase(name='ddpg_critic_base', input_dims= [d_normalized.shape, s_normalized.shape],hidden_dims=[400, 300], chkpt_dir='models')

                critic_head = ValueHead(name='ddpg_critic_head', input_dims=[300],chkpt_dir='models')

                critic = Sequential(critic_base,critic_head)

                target_critic_base = CriticBase(name='ddpg_target_critic_base', input_dims=[d_normalized.shape, s_normalized.shape],hidden_dims=[400, 300], chkpt_dir='models')

                target_critic_head = ValueHead(name='ddpg_target_critic_head', input_dims=[300],chkpt_dir='models')

                target_critic = Sequential(target_critic_base,target_critic_head)

                memory = ReplayBuffer(max_size=100_000, 
                                    input_shape= [d_normalized.shape, s_normalized.shape], 
                                    batch_size=32, 
                                    n_actions=[0], 
                                    action_space='continuous')
                
                policy = NoisyDeterministicPolicy(n_actions=[0],
                                                min_action=[0],
                                                max_action=[0])
                
                agent= Agent(actor, critic, target_actor, target_critic, memory, policy)

                # --------------------------------------------------------------------
                
                #Need to output the rpm prediction
                #Need to clip prediction so that it fits in the range 15<rpm<200

                time.sleep(0.05)

            except KeyboardInterrupt:
                print('Keyboard interrupt detected. Exiting...')
                break

    finally:
        client.disconnect()
            
if __name__ == '__main__':
    main()



