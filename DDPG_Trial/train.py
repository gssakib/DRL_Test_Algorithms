import time
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
from utils import plotLearning, plotRPMandPrediction

#Intialize DDPG Agent
agent = Agent(alpha=0.000001, beta=0.00001, input_dims=[2], tau=0.005,
              batch_size=120, layer1_size=800, layer2_size=600,
              n_actions=1)

#Importing the DataSet
csv_file_path_1 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_2 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_3 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.1Hz_2min/Freq_0.1Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_4 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
df_4 = pd.read_csv(csv_file_path_4, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_5 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
df_5 = pd.read_csv(csv_file_path_5, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_6 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.2Hz_2min/Freq_0.2Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
df_6 = pd.read_csv(csv_file_path_6, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_7 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter01.CSV" # Replace with the actual path to your Excel file
df_7 = pd.read_csv(csv_file_path_7, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_8 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter02.CSV"  # Replace with the actual path to your Excel file
df_8 = pd.read_csv(csv_file_path_8, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

csv_file_path_9 = "C:/Users/keegh/Dropbox (MIT)/_MIT_mengm_2023_plcdeploy_/ML Model Training Data/11_8 data/Surebonder_straight_gluestick_data/Freq_0.3Hz_2min/Freq_0.3Hz_2min_Iter03.CSV"  # Replace with the actual path to your Excel file
df_9 = pd.read_csv(csv_file_path_9, skiprows=13, header=0, usecols=[5,7,8], nrows=100) # Load the Excel sheet, excluding the specified column

#Combine DataSet 
features = pd.concat([df_1,df_2,df_3, df_4,df_5,df_6,df_7,df_8,df_9], axis=0)

# Assume X_train, y_train, X_test, y_test are your data
X_train_o, X_test_o, y_train, y_test = train_test_split(features[['Setpoint','Diameter']],features[['Measured_rpm_Filtered']], test_size=0.1, random_state=42, shuffle=False)

#Normalize the Data
scaler = preprocessing.StandardScaler().fit(X_train_o)
X_train = scaler.transform(X_train_o)
scaler_2 = preprocessing.StandardScaler().fit(X_test_o)
X_test = scaler_2.transform(X_test_o)

#Load Previously saved models
#agent.load_models()

start_time = time.time()

#Training Loop
num_epochs = 20
X_train = pd.DataFrame(X_train_o)
y_train = pd.DataFrame(y_train)
X_train = X_train.dropna()
y_train = y_train.dropna()

score_history = []
counter_history = []
rpms = []
predictions = []
correctPredictions = 0 
totalPredictions=0

for epoch in range(num_epochs):
    score = 0
    counter = 0
    for state, act in zip(X_train.values-1, y_train.values-1):
        #Change State list into a Numpy Array
        state = np.array(state)
        state = np.array(state, dtype=np.float32)
               
        #Select action
        action = agent.choose_action(state)
        #action = np.mean(y_train.values)*action*2 + np.std(y_train.values)*2
        d = np.array(y_train.max())-np.array(y_train.min())
        action = ((action+0.4)/0.75)*d + np.array(y_train.min())

        if action < 15:
            action = np.array([action[0]+50])
        elif action > 200:
            action = np.array([action[0]-50])
        
        #Observe new state
        if counter+1< X_train.shape[0]:
            new_state = X_train.iloc[counter+1].values
        else:
            new_state = np.array([0,0])

        #Calculate Error 
        error = int(y_train.iloc[counter+1]) - int(action) #rpm - predicted rpm

        #Calculate Accuracy 
        if abs(error)>10:
            totalPredictions +=1
        else:
            correctPredictions += 1
            totalPredictions+= 1

        #Calculate Reward
        reward = -abs(error)

        #Store the experience
        agent.remember(state, action, reward, new_state)

        #Learn from the experience
        agent.learn()

        score = -reward
        counter += 1
        rpms.append(y_train.iloc[counter+1])
        predictions.append(action)
        print("Count|", counter, "|epoch|", epoch, "/", num_epochs, "|action|", action, "action_raw", agent.choose_action(state), "|RPM at t+1|",int(y_train.iloc[counter+1]), "|error", error)
        
    score_history.append(score)
    print('epoch ', epoch, 'Score %.2f' % score,
              'trailing 5 data points avg %.3f' % np.mean(score_history[-100:]))
    accuracy = (correctPredictions/totalPredictions)*100
    print('Accuracy Score: %.2f' % accuracy, '%')

filename = 'DDPG Training_Avg Error vs Time .png'
plotLearning(score_history, filename, window=100)
plotRPMandPrediction(rpms, predictions, filename="RPM and Predicted RPMs", window=100)

end_time = time.time()
duration = (end_time - start_time)/60
print("Time taken to train the model", duration, "mins")

#Save the model
agent.save_models()

