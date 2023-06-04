import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


# Neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(predictor), 32)
        self.fc2 = nn.Linear(32, len(targets))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = torch.load('input_model_v3.pth')
model.eval()
model.to(device)

# Load the future date time stamps
future_data = pd.read_csv('future_data_one_column.csv')

# Convert time to datetime and add 'elapsed' column
future_data['time'] = pd.to_datetime(future_data['time'])
future_data['elapsed'] = (future_data['time'] - future_data['time'].min()).dt.total_seconds()

# Predictor variable
predictor = 'elapsed'

# Target variable
targets = ['packets_input', 'packets_output', 'packets_dropped']

# Scale the data
scaler_predictor = MinMaxScaler().fit(future_data[predictor].values.reshape(-1, 1))
future_data_scaled = scaler_predictor.transform(future_data[predictor].values.reshape(-1, 1))
future_data_scaled = pd.DataFrame(future_data_scaled, columns=[predictor])

# Convert to PyTorch tensor
future_data_scaled = torch.tensor(future_data_scaled.values, dtype=torch.float32).to(device)

# Make predictions using the model
with torch.no_grad():
 predictions = model(future_data_scaled)

# Inverse transform the data
scaler_targets = MinMaxScaler().fit(future_data[targets])
predictions = predictions.cpu().numpy()
inverse_predictions = scaler_targets.inverse_transform(predictions)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
for i in range(3):
 axes[i].plot(future_data['time'], inverse_predictions[:, i], label='Predicted')
 axes[i].set_title('Prediction for {}'.format(targets[i]))
 axes[i].legend()
plt.show()

# Inverse transform the predictions
inverse_predictions = scaler_targets.inverse_transform(predictions)

# Save the predictions to a csv file
prediction_df = pd.DataFrame(inverse_predictions, columns=targets)
prediction_df.to_csv('model_prediction.csv', index=False)
