import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import optuna

# Step 1: Load data
inflow_data = pd.read_csv('Inflow_Data.csv', parse_dates=['TimeStample'], index_col='TimeStample')
rainfall_data = pd.read_csv('Rainfall_Data.csv', parse_dates=['TimeStample'], index_col='TimeStample')
env_data = pd.read_csv('Environment_Data.csv', parse_dates=['TimeStample'], index_col='TimeStample')

# Step 2: Data Preprocessing (Normalization)
data = pd.concat([inflow_data, rainfall_data, env_data], axis=1).dropna()  # Merge the datasets based on TimeStample
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)  # Normalize all the columns

# Step 3: Add 'jour_pred' variable to adjust prediction horizon
jour_pred = 8  # Predict next 8 time steps (24 hours, 8 * 3-hour intervals)

# Step 4: Define function to create sequences
def create_sequences(data_scaled, sequence_length, jour_pred):
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length - jour_pred + 1):
        X.append(data_scaled[i:i+sequence_length])  # Use the past 'sequence_length' records
        y.append(data_scaled[i+sequence_length:i+sequence_length + jour_pred, 0])  # Predict next 'jour_pred' steps after the sequence
    return np.array(X), np.array(y)

# Step 5: Define evaluation metrics
def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    nrmse = rmse / (np.max(y_true_flat) - np.min(y_true_flat))
    
    # NSE (Nash-Sutcliffe Efficiency)
    numerator_nse = np.sum((y_true_flat - y_pred_flat) ** 2)
    denominator_nse = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    nse = 1 - (numerator_nse / denominator_nse)
    
    return rmse, mae, nrmse, nse

# Step 6: Define the objective function for Optuna optimization
def objective(trial):
    # Suggest a sequence_length to try
    sequence_length = trial.suggest_int('sequence_length', 1, 30)  # Example: 1 to 30 time steps

    # Create sequences based on the suggested sequence length
    X, y = create_sequences(data_scaled, sequence_length, jour_pred)

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and compile the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(jour_pred, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=120, batch_size=32, verbose=0)

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Inverse transform the predictions and true values to original scale
    def inverse_transform(y_scaled):
        dummy = np.zeros((y_scaled.shape[0], data.shape[1]))
        for i in range(jour_pred):
            dummy[:, 0] = y_scaled[:, i]
        y_original = scaler.inverse_transform(dummy)[:, 0]
        return y_original
    
    y_test_original = inverse_transform(y_test)
    y_pred_test_original = inverse_transform(y_pred_test)

    # Calculate RMSE for Optuna optimization
    rmse, mae, nrmse, nse = calculate_metrics(y_test_original, y_pred_test_original)
    return rmse  # Optuna will minimize this value

# Step 7: Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Run 50 trials

# Best sequence length found by Optuna
best_sequence_length = study.best_trial.params['sequence_length']
print(f"Best sequence length: {best_sequence_length}")

# Step 8: Use the best sequence length for final training and evaluation
X, y = create_sequences(data_scaled, best_sequence_length, jour_pred)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 9: Train final LSTM model with the best sequence length
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(jour_pred, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=120, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred_test = model.predict(X_test)

# Inverse transform predictions and true values
def inverse_transform(y_scaled):
    dummy = np.zeros((y_scaled.shape[0], data.shape[1]))
    for i in range(jour_pred):
        dummy[:, 0] = y_scaled[:, i]
    y_original = scaler.inverse_transform(dummy)[:, 0]
    return y_original

y_test_original = inverse_transform(y_test)
y_pred_test_original = inverse_transform(y_pred_test)

# Step 10: Calculate final metrics
rmse, mae, nrmse, nse = calculate_metrics(y_test_original, y_pred_test_original)

# Display metrics
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, NRMSE: {nrmse:.4f}, NSE: {nse:.4f}")

# Step 11: Plot actual vs predicted inflows for the test set
dates = data.index[best_sequence_length:]
test_dates = dates[train_size:]

plt.figure(figsize=(10, 6))
plt.plot(test_dates[:len(y_test_original)], y_test_original, label='Actual Inflows (Test Set)', color='blue', linewidth=1)
plt.plot(test_dates[:len(y_pred_test_original)], y_pred_test_original, label='Predicted Inflows (Test Set)', color='red', linewidth=1)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Inflows', fontsize=12)
plt.title('Actual vs Predicted Inflows (Test Set)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
