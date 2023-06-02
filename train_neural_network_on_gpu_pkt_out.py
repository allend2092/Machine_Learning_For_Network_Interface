import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the data
data = pd.read_csv('asa_data.csv')

# Convert time to datetime and add 'elapsed' column
data['time'] = pd.to_datetime(data['time'])
data['elapsed'] = (data['time'] - data['time'].min()).dt.total_seconds()

# Predictor and target variables
predictors = ['elapsed']
targets = ['packets_output']

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[predictors + targets])
data_scaled = pd.DataFrame(data_scaled, columns=predictors + targets)

# Split into training and testing sets
train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_data = torch.tensor(train_data.values, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)

# Neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(predictors), 32)
        self.fc2 = nn.Linear(32, len(targets))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model and move it to the GPU
model = Net().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    outputs = model(train_data[:, :len(predictors)])
    loss = criterion(outputs, train_data[:, len(predictors):])

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.item()))

# Testing
model.eval()
with torch.no_grad():
    predictions = model(test_data[:, :len(predictors)])

print('Test Loss: {:.4f}'.format(criterion(predictions, test_data[:, len(predictors):]).item()))

# Save the model
torch.save(model.state_dict(), 'output_model.pth')

# Fit a new scaler on the target column alone
scaler_target = StandardScaler()
scaler_target.fit(data[['packets_output']])

dummy_pred = np.zeros((predictions.shape[0], len(predictors)))
pred_with_dummy = np.concatenate((dummy_pred, predictions.cpu().numpy()), axis=1)

# Plotting the results
plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(test_data.cpu().numpy())[:, -1], label='True')
plt.plot(scaler.inverse_transform(pred_with_dummy)[:, -1], label='Predicted')
plt.title('Packet Output Prediction')
plt.xlabel('Time')
plt.ylabel('Packet Output')
plt.legend()
plt.show()

