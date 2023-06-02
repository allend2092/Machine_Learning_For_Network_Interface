import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
data = pd.read_csv('asa_data.csv')

# Convert time to datetime
data['time'] = pd.to_datetime(data['time'])

# Create an 'elapsed' column that holds the time elapsed since the start of data
data['elapsed'] = (data['time'] - data['time'].min()).dt.total_seconds()

# Choose the predictor variables, here 'elapsed', and select the target variable, 'packets_input'
X = data['elapsed'].values.reshape(-1,1)
y = data['packets_input']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Use the model to make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)

print('Mean Squared Error:', mse)

# Plot the original data and the predicted data
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()
