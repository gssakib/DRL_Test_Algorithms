import pandas as py 
import ddpg_tf
from ddpg_tf import Agent
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

#Intialize DDPG Agent
agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[2], tau=0.001,
              batch_size=64, layer1_size=800, layer2_size=600,
              n_actions=1)

#Importing the DataSet

csv_file_path_1 = "C:/Users/keegh/Documents/Orbtronics Agri Sensor/DRL_Test_Algorithms/diameter&RPM_SP0.2.CSV" # Replace with the actual path to your Excel file
df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[3,4], nrows=5000) # Load the Excel sheet, including the specified column
df_1['Setpoint'] = 0.2

csv_file_path_2 = "C:/Users/keegh/Documents/Orbtronics Agri Sensor/DRL_Test_Algorithms/diameter&RPM_SP0.4.CSV"  # Replace with the actual path to your Excel file
df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[3,4], nrows=5000) # Load the Excel sheet, including the specified column
df_2['Setpoint'] = 0.4

csv_file_path_3 = "C:/Users/keegh/Documents/Orbtronics Agri Sensor/DRL_Test_Algorithms/diameter&RPM_SP0.6.CSV"  # Replace with the actual path to your Excel file
df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[3,4], nrows=5000) # Load the Excel sheet, including the specified column
df_3['Setpoint'] = 0.6

#Combine DataSet 
features = pd.concat([df_1, df_2, df_3], axis=0)

# Assume X_train, y_train, X_test, y_test are your data
X_train_o, X_test_o, y_train, y_test = train_test_split(features[['Diameter','Setpoint']],features[['Measured_rpm']], test_size=0.2, random_state=42)


#Normalize the Data
scaler = preprocessing.StandardScaler().fit(X_train_o)
X_train = scaler.transform(X_train_o)
scaler_2 = preprocessing.StandardScaler().fit(X_test_o)
X_test = scaler_2.transform(X_test_o)


#Training Loop
num_epochs = 10
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
score_history = []

for epoch in range(num_epochs):
    total_reward = 0
    counter = 0
    for state, action in zip(X_train.values, y_train.values):
        #Add state and action to an array
        state = np.array(state)
        state = np.array(state, dtype=np.float32)
        #state = (state, {})
        #print("The state is", state)
        action = np.array([action])
        #print("The action is", action)
                
        #Predict the next state 
        next_state = agent.choose_action(state)
        #print("The prediction of the next state is", next_state)

        #Calculate the reward [Note this reward function may need to be corrected]
        reward = -abs(state[0]-state[1]) #diameter - setpoint
        #print("The reward is ", state[0], "-", state[1], "=", reward)

        #Store the experience
        agent.remember(state, action, reward, next_state)

        #Learn from the experience
        agent.learn()

        total_reward += reward
        counter += 1
        print(counter)

score_history.append(total_reward)
action = np.full(y_train.shape, action)
mse = mean_squared_error(y_train, action)
print ("Mean Squared Error: ", mse)
print(f'Epoch {epoch+1}/{num_epochs}, Total Reward: {total_reward}')   

#print(y_train.shape)
#print(action.shape)

#Testing Loop
num_epochs_testing = 1
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
score_history_testing = []

for epoch in range(num_epochs):
    total_reward = 0
    counter= 0
    for state, action in zip(X_test.values, y_test.values):
        #Add state and action to an array
        state = np.array(state)
        state = np.array(state, dtype=np.float32)
        #state = (state, {})
        action = np.array([action])

        #Predict the next state 
        next_state = agent.choose_action(state)

        #Calculate the reward [Note this reward function may need to be corrected]
        reward = -abs(state[0]-state[1]) #diameter - set point

        #Store the experience
        agent.remember(state, action, reward, next_state)

        #Learn from the experience
        agent.learn()

        total_reward += reward
        counter += 1
        print(counter)

score_history.append(total_reward)
action = np.full(y_test.shape, action)
print(f'Epoch {epoch+1}/{num_epochs}, Total Reward: {total_reward}')
mse = mean_squared_error(y_test, action)
print ("Mean Squared Error: ", mse)

#Save the model
agent.save_models()