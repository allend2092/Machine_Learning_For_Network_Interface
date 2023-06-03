import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the data
data = pd.read_csv('asa_data.csv')

# Convert time to datetime and add 'elapsed' column
data['time'] = pd.to_datetime(data['time'])
data['elapsed'] = (data['time'] - data['time'].min()).dt.total_seconds()

# Predictor and target variables. Predictor is can be thought of as an independent variable
# targets can be thought of as independent variable(s).
predictors = ['elapsed']
targets = ['packets_input', 'packets_output', 'packets_dropped']

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[predictors + targets])
data_scaled = pd.DataFrame(data_scaled, columns=predictors + targets)

# Split into training and testing sets
train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
train_data = torch.tensor(train_data.values, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)

# Create a custom dataset and a data loader
class ASADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :len(predictors)]
        y = self.data[idx, len(predictors):]
        return x, y

train_dataset = ASADataset(train_data)
test_dataset = ASADataset(test_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(predictors), 32)
        self.fc2 = nn.Linear(32, len(targets))

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create the model and move it to the GPU
model = Net().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Loop over batches
    for inputs, labels in train_loader:
        # Move inputs and labels to the GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 1000, loss.item()))

# Testing
model.eval()
with torch.no_grad():
 # Initialize empty lists to store predictions and true values
 predictions = []
 true_values = []

 # Loop over batches
 for inputs, labels in test_loader:
    # Move inputs and labels to the GPU
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(inputs)

 # Append predictions and true values to the lists
 predictions.append(outputs.cpu().numpy())
 true_values.append(labels.cpu().numpy())

# Concatenate the lists
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

print('Test Loss: {:.4f}'.format(criterion(torch.tensor(predictions), torch.tensor(true_values)).item()))

# Save the model
torch.save(model, 'input_model_improved_v3.pth')

# Inverse transform the data
scaler_predictors = MinMaxScaler().fit(data[predictors])
scaler_targets = MinMaxScaler().fit(data[targets])

inverse_predictions = scaler_targets.inverse_transform(predictions)
inverse_true_values = scaler_targets.inverse_transform(true_values)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
for i in range(3):
 axes[i].plot(inverse_true_values[:, i], label='True')
 axes[i].plot(inverse_predictions[:, i], label='Predicted')
 axes[i].set_title('Test Data vs Prediction for {}'.format(targets[i]))
 axes[i].legend()
plt.show()
