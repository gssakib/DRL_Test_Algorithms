import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Generate sample data (replace with your own dataset)
import pandas as pd

# Load the Excel file and skip a specific column
csv_file_path_1 = 'E:/grad/thesis/8-18 Experiments/diameter&RPM_SP0.2.CSV'  # Replace with the actual path to your Excel file

# Load the Excel sheet, excluding the specified column

df_1 = pd.read_csv(csv_file_path_1, skiprows=13, header=0, usecols=[3,4], nrows=5000)
df_1['Setpoint'] = 0.2

csv_file_path_2 = 'E:/grad/thesis/8-18 Experiments/diameter&RPM_SP0.4.CSV'  # Replace with the actual path to your Excel file

# Load the Excel sheet, excluding the specified column

df_2 = pd.read_csv(csv_file_path_2, skiprows=13, header=0, usecols=[3,4], nrows=5000)
df_2['Setpoint'] = 0.4

csv_file_path_3 = 'E:/grad/thesis/8-18 Experiments/diameter&RPM_SP0.6.CSV'  # Replace with the actual path to your Excel file

# Load the Excel sheet, excluding the specified column

df_3 = pd.read_csv(csv_file_path_3, skiprows=13, header=0, usecols=[3,4], nrows=5000)
df_3['Setpoint'] = 0.6

# Display the DataFrame
print(df_1)
print(df_2)
print(df_3)

features = pd.concat([df_1, df_2, df_3], axis=0)
# Assume X_train, y_train, X_test, y_test are your data
X_train_o, X_test_o, y_train, y_test = train_test_split(features[['Diameter','Setpoint']],features[['Measured_rpm']], test_size=0.2, random_state=42)

# Normalize the data
scaler = preprocessing.StandardScaler().fit(X_train_o)
X_train = scaler.transform(X_train_o)
scaler_2 = preprocessing.StandardScaler().fit(X_test_o)
X_test = scaler_2.transform(X_test_o)
# y_train = preprocessing.normalize(y_train)
# y_test = preprocessing.normalize(y_test)

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],) ),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear' )  # Change units for a multi-class problem
])


custom_learning_rate = 0.0001  # You can adjust this value

# Create an optimizer with the custom learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)

# Compile the model using the custom optimizer
model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set using mean squared error
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
train_predictions = model.predict(X_train)

# Check accuracy within a threshold
threshold = 10
correct_predictions = np.abs(predictions.squeeze() - y_test.values.squeeze()) <= threshold
accuracy = np.mean(correct_predictions)
print("Accuracy within threshold:", accuracy)
means_values = np.mean(X_train_o, axis = 0)
std_values = np.std(X_train_o,axis = 0)
print("Means:", scaler.mean_)
print("Std:", scaler.scale_)


y_test.to_excel('y_test.xlsx',index = False)
df_predictions = pd.DataFrame(predictions) 
df_predictions.to_excel('predictions.xlsx',index = False)

# save the model
model.save('E:/grad/thesis/8-18 Experiments/model1')




# Retrieve training and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create an array for the number of epochs
epochs = range(1, len(train_loss) + 1)

# Plot training and validation loss
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()














# Define the neural network architecture
layers = [2, 10, 10, 1]  # Input, hidden layers, output


# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))


# Draw the nodes (circles)
for i, n in enumerate(layers):
    if n != 0:
        layer_height = (layers[0] - 1) * 2 / max((n - 1), 1) if i != 0 else 0
        for j in range(n):
            ax.add_artist(plt.Circle((i, j * 2 + layer_height), 0.3, color='blue'))

# Draw the connections (lines)
for i in range(len(layers) - 1):
    for j in range(layers[i]):
        for k in range(layers[i + 1]):
            ax.plot([i, i + 1], [j * 2, k * 2], color='blue')

# Set axis limits and hide the axes
ax.set_xlim(-0.5, len(layers) - 0.5)
ax.set_ylim(-1, max(layers) * 2)
ax.axis('off')

plt.title('Neural Network Architecture')
plt.show()