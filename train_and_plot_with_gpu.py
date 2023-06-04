import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
torch.save(model, 'input_model_v3.pth')

# Inverse transform the data
scaler_predictors = MinMaxScaler().fit(data[predictors])
scaler_targets = MinMaxScaler().fit(data[targets])

predictions = predictions.cpu().numpy()
test_data = test_data.cpu().numpy()

inverse_predictions = scaler_targets.inverse_transform(predictions)
inverse_true_values = scaler_targets.inverse_transform(test_data[:, len(predictors):])

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
for i in range(3):
 axes[i].plot(inverse_true_values[:, i], label='True')
 axes[i].plot(inverse_predictions[:, i], label='Predicted')
 axes[i].set_title('Test Data vs Prediction for {}'.format(targets[i]))
 axes[i].legend()
plt.show()

# Save the predictions to a csv file
prediction_df = pd.DataFrame(inverse_predictions, columns=targets)
prediction_df.to_csv('test_data_model_prediction.csv', index=False)

# Save the predictions to a csv file
prediction_df = pd.DataFrame(inverse_true_values, columns=targets)
prediction_df.to_csv('test_data_model_true.csv', index=False)
