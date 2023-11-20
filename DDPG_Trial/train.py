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
from data import Import_Data

#Intialize DDPG Agent
agent = Agent(alpha=0.00001, beta=0.0001, input_dims=[2], tau=0.0005,
              batch_size=1, layer1_size=400, layer2_size=400,
              n_actions=1)

#Importing the DataSet
#data = Import_Data()

csv_file_path_1 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_2 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_3 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_4 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
df_4 = pd.read_csv(csv_file_path_4, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_5 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
df_5 = pd.read_csv(csv_file_path_5, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_6 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
df_6 = pd.read_csv(csv_file_path_6, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_7 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
df_7 = pd.read_csv(csv_file_path_7, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_8 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
df_8 = pd.read_csv(csv_file_path_8, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

csv_file_path_9 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
df_9 = pd.read_csv(csv_file_path_9, skiprows=13, header=0, usecols=[5,7,8], nrows=500) # Load the Excel sheet, excluding the specified column

#Combine DataSet 
features = pd.concat([df_1,df_2,df_3, df_4,df_5,df_6,df_7,df_8,df_9], axis=0)

# Assume X_train, y_train, X_test, y_test are your data
X_train_o, X_test_o, y_train, y_test = train_test_split(features[['Setpoint','Diameter']],features[['Measured_rpm_Filtered']], test_size=0.2, random_state=42)
"""
#Normalize the Data
scaler = preprocessing.StandardScaler().fit(X_train_o)
X_train = scaler.transform(X_train_o)
scaler_2 = preprocessing.StandardScaler().fit(X_test_o)
X_test = scaler_2.transform(X_test_o)
"""
#agent.load_models()

#Training Loop
num_epochs = 10
X_train = pd.DataFrame(X_train_o)
y_train = pd.DataFrame(y_train)
X_train = X_train.dropna()
y_train = y_train.dropna()

score_history = []
mse_history = []
counter_history = []
rpm = []
predicted_rpm = []
data_to_export = []
for epoch in range(num_epochs):
    total_reward = 0
    counter = 0
    next_states = []
    actions = []
    setpoint = []
    predicted_diameter = []
    rpm =[]
    for state, action in zip(X_train.values, y_train.values):
        #Add state and action to an array. state["Setpoint", "Diameter"]
        state = np.array(state)
        state = np.array(state, dtype=np.float32)
        #state = (state, {})
        #action["rpm"]
        action = np.array([action])
                
        #Predict the next state 
        next_state = agent.choose_action(state)
        next_state = 67.5*next_state + 82.5
        """
        if next_state < 15:
            next_state = next_state + 15
        elif next_state > 200:
            next_state = 200
        else:
            next_state = next_state
        next_states.append(next_state)
        actions.append(action)
        """

        #Calculate the reward [Note this reward function may need to be corrected]
        error =action-next_state #rpm - predicted rpm
        #print("rpm", action, "predicted_rpm", next_states[0])
        reward = -abs(error)

        #Every 200 counts, calculate MSE and reset the predictions list
        #setpoint, diameter+reward
        if (counter + 1)%200 ==0:
            #setpoint.append(state[0])
            rpm.append(action[0])
            predicted_rpm.append(next_state)
            batch_mse = mean_squared_error(rpm, predicted_rpm)
            mse_history.append(batch_mse)
            counter_history.append(counter+1)
            print(counter, "RPM", rpm[0], "predicted rpm ", predicted_rpm[0], "error", error, "reward ", reward, "epoch ", epoch, "/", num_epochs)
            #rpm_value = float(rpm[0])
            #predicted_rpm_value = float(predicted_rpm[0])
            #data_to_export.append([counter, rpm_value, predicted_rpm_value, error, reward, epoch, num_epochs])
            
            #Reset the predictions list for the next batch
            rpm = []
            predicted_rpm=[]

        #Store the experience
        agent.remember(state, action, reward, next_state)

        #Learn from the experience
        #agent.learn()

        total_reward += reward
        counter += 1
        
#df = pd.DataFrame(data_to_export, columns=['Counter', 'RPM', 'Predicted RPM', 'Error', 'Reward', 'Epoch', 'Total Epochs'])
#df.to_csv('training_data.csv', index=False)

score_history.append(total_reward)
print(f'Epoch {epoch+1}/{num_epochs}, Total Reward: {total_reward}')
print ("Mean Squared Error: ", batch_mse)

#Plotting the graph of MSE versus Counter
plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.plot(counter_history, mse_history, marker='o', linestyle='-', color='b')
plt.xlabel('Counter')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Counter (Every 200 Counts)')
plt.grid(True)
plt.show()

"""
#Plotting the graph of RPM and Predicted RPM vs Counter
plt.figure(figsize=(10, 5))  # Set the figure size as desired
plt.plot(counter_history, rpm, label='Setpoint', marker='o')  # Plot actual RPM
plt.plot(counter_history, predicted_rpm, label='Predicted RPM', marker='x')  # Plot predicted RPM

plt.title('RPM and Predicted RPM over Time')  # Set the title of the graph
plt.xlabel('Counter')  # Set the x-axis label
plt.ylabel('RPM')  # Set the y-axis label
plt.legend()  # Show the legend
plt.grid(True)  # Show grid
plt.tight_layout()  # Fit the plot neatly into the figure area
plt.show()  # Display the plot
"""

#Testing Loop
num_epochs_testing = 10
X_test = pd.DataFrame(X_test_o)
y_test = pd.DataFrame(y_test)
X_test = X_test.dropna()
y_test = y_test.dropna()
score_history_testing = []
mse_history = []
counter_history = []
rpm = []
predicted_rpm = []

for epoch in range(num_epochs):
    total_reward = 0
    counter = 0
    next_states = []
    actions = []
    setpoint = []
    predicted_diameter = []
    for state, action in zip(X_train.values, y_train.values):
        #Add state and action to an array. state["Setpoint", "Diameter"]
        state = np.array(state)
        state = np.array(state, dtype=np.float32)
        #state = (state, {})
        #action["rpm"]
        action = np.array([action])
                
        #Predict the next state 
        next_state = agent.choose_action(state)
        next_state = 67.5*next_state + 82.5
        
        """
        if next_state < 15:
            next_state = 15
        elif next_state > 200:
            next_state = 200
        else:
            next_state = next_state
        next_states.append(next_state)
        actions.append(action)
        """

        #Calculate the reward [Note this reward function may need to be corrected]
        error =state[0]-state[1] #diameter - setpoint
        reward = -abs(error)

        #Every 200 counts, calculate MSE and reset the predictions list
        #setpoint, diameter+reward
        if (counter + 1)%200 ==0:
            #setpoint.append(state[0])
            rpm.append(action[0])
            predicted_rpm.append(next_state)
            batch_mse = mean_squared_error(rpm, predicted_rpm)
            mse_history.append(batch_mse)
            counter_history.append(counter+1)
            print(counter, "RPM", rpm[0], "predicted rpm ", predicted_rpm[0], "error", error, "reward ", reward, "epoch ", epoch, "/", num_epochs)
            #rpm_value = float(rpm[0])
            #predicted_rpm_value = float(predicted_rpm[0])
            #data_to_export.append([counter, rpm_value, predicted_rpm_value, error, reward, epoch, num_epochs])
            
            #Reset the predictions list for the next batch
            rpm = []
            predicted_rpm=[]

        #Store the experience
        agent.remember(state, action, reward, next_state)

        #Learn from the experience
        agent.learn()

        total_reward += reward
        counter += 1
        
#df = pd.DataFrame(data_to_export, columns=['Counter', 'RPM', 'Predicted RPM', 'Error', 'Reward', 'Epoch', 'Total Epochs'])
#df.to_csv('training_data.csv', index=False)

score_history.append(total_reward)
print(f'Epoch {epoch+1}/{num_epochs}, Total Reward: {total_reward}')
print ("Mean Squared Error: ", batch_mse)

#Plotting the graph of MSE versus Counter
plt.figure(figsize=(10, 5))  # Set the figure size (optional)
plt.plot(counter_history, mse_history, marker='o', linestyle='-', color='b')
plt.xlabel('Counter')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Counter (Every 200 Counts)')
plt.grid(True)
plt.show()

"""
#Plotting the graph of RPM and Predicted RPM vs Counter
plt.figure(figsize=(10, 5))  # Set the figure size as desired
plt.plot(counter_history, rpm, label='Setpoint', marker='o')  # Plot actual RPM
plt.plot(counter_history, predicted_rpm, label='Predicted RPM', marker='x')  # Plot predicted RPM

plt.title('RPM and Predicted RPM over Time')  # Set the title of the graph
plt.xlabel('Counter')  # Set the x-axis label
plt.ylabel('RPM')  # Set the y-axis label
plt.legend()  # Show the legend
plt.grid(True)  # Show grid
plt.tight_layout()  # Fit the plot neatly into the figure area
plt.show()  # Display the plot
"""

#Save the model
agent.save_models()