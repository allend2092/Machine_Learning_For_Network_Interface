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

# Define a function to create a model, train it, and visualize the results
def model_and_plot(data, target_variable):
    # Choose the predictor variables, here 'elapsed', and select the target variable
    X = data['elapsed'].values.reshape(-1,1)
    y = data[target_variable]

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

    print(f'Mean Squared Error for {target_variable}:', mse)

    # Plot the original data and the predicted data
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=2)
    plt.title(f'Prediction of {target_variable} over time')
    plt.xlabel('Time elapsed (seconds)')
    plt.ylabel(target_variable)
    plt.show()

# Apply the function to 'packets_input', 'packets_output', and 'packets_dropped'
model_and_plot(data, 'packets_input')
model_and_plot(data, 'packets_output')
model_and_plot(data, 'packets_dropped')
