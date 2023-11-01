import sys
import ddpg_tf
from ddpg_tf import Agent
sys.path.insert(0, "..")
from opcua import Client
import time
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    #Connect to the client
    client = Client("opc.tcp://DESKTOP-A9QNR1L:4990/FactoryTalkLinxGateway1")
    agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[2], tau=0.001,
              batch_size=64, layer1_size=800, layer2_size=600,
              n_actions=1)
    
    try:
        client.connect()
        root = client.get_root_node()
        setpoint = client.get_node("ns=2;s=[opc_server]setpoint_diameter")
        diameter = client.get_node("ns=2;s=[opc_server]Diameter")
        setpoint_speed = client.get_node("ns=2;s=[opc_server]DC_output")
        model = agent.load_models() #input the path address of the model

        def get_state_from_system():
            s = setpoint.get_value()
            d = diameter.get_value()

            d_means = 0.42954475
            s_means = 0.40073333
            d_std = 0.13086448
            s_std = 0.16299119 

            d_normalized = (d-d_means)/d_std
            s_normalized = (s-s_means)/s_std

            return [s_normalized, d_normalized]
        
        def send_action_to_system(action):
            rpm = setpoint_speed.set_value(action)

        while True:
            try:
                current_state = get_state_from_system()
                action = Agent.choose_action(current_state)

                #Limiting rpm range to 15-200
                if action <= 15:
                    send_action_to_system(15)
                elif action >= 200:
                    send_action_to_system(200)
                else:
                    send_action_to_system(float(action))

                time.sleep(0.05)

            except KeyboardInterrupt:
                print("Keyboard interrupt detected Exiting...")
                break
    finally:
        client.disconnect()

        
        