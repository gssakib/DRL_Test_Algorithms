import sys
sys.path.insert(0, "..")
from opcua import Client
import time
import numpy as np

if __name__ == "__main__":
    client = Client("opc.tcp://DESKTOP-A9QNR1L:4990/FactoryTalkLinxGateway1")
    #connect using a user
    # client = Client("opc.tcp://admin@localhost:4840/freeopcua/server/")
    try:
        client.connect()

        # set connection node with 
        setpoint = client.get_node("ns=2;s=[opc_server]setpoint_diameter")
        diameter = client.get_node("ns=2;s=[opc_server]Diameter")
        setpoint_speed = client.get_node("ns=2;s=[opc_server]DC_output")
        # print(var.get_value())
        # print("motor_state:", end='')
        # state = input()
        # var.set_value(int(state))
        
        #var.write_value(True)
        #print(state)

        # var =  client.nodes.root.get_child("ns=2;s=[opc_server]Motor_Run")
        # var.write(True)

        #var.get_data_value() # get value of node as a DataValue object
        #var.get_value() # get value of node as a python builtin
        #var.set_value(3.9) # set node value using implicit data type

        # Now getting a variable node using its browse path
        # myvar = root.get_child(["0:Objects", "2:MyObject", "2:MyVariable"])
        # obj = root.get_child(["0:Objects", "2:MyObject"])
        # print("myvar is: ", myvar)
        # print("myobj is: ", obj)
        # print(myvar.get_value())


        # time.sleep(1)

        # #try to write to the server
        # var.set_value(2)
        # time.sleep(5)
        # var.set_value(0)
        while(True):
            try: 
                d = diameter.get_value()
                s = setpoint.get_value()

                #normalize
                d_means = 0.42954475
                s_means = 0.40073333
                d_std = 0.13086448
                s_std = 0.16299119 

                d_normalized = (d-d_means)/d_std
                s_normalized = (s-s_means)/s_std     
                
                # print(d,d_normalized)
                # print(type(d_normalized))
                # print(type(0.5))
                input = np.array([[d_normalized,s_normalized]])
                predictions = model.predict(input)
                predictions = predictions.squeeze()
                print(predictions)
                print(type(predictions))
                if predictions <= 15:
                    setpoint_speed.set_value(15)
                elif predictions >= 200:
                    setpoint_speed.set_value(200)
                else:
                    setpoint_speed.set_value(float(predictions))

                time.sleep(0.05)
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting...")
                break



    finally:
        client.disconnect()