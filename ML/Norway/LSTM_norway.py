import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Parameter for predicting future days
jour_pred = 30

# Load the inflow data
inflow_data = pd.read_csv('R1.csv', parse_dates=['Date'], index_col='Date')

# Add features to the dataset
inflow_data['month'] = inflow_data.index.month
inflow_data['season'] = inflow_data.index.month % 12 // 3 + 1
inflow_data['inflow_ma_3'] = inflow_data['Inflows'].rolling(window=3).mean()
inflow_data['lag_1'] = inflow_data['Inflows'].shift(1)
inflow_data['lag_2'] = inflow_data['Inflows'].shift(2)

# Normalize 'Inflows' column
scaler_inflows = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler_inflows.fit_transform(inflow_data[['Inflows']])

# Normalize other features
scaler_other = MinMaxScaler(feature_range=(0, 1))
other_features_scaled = scaler_other.fit_transform(inflow_data[['month', 'season', 'inflow_ma_3', 'lag_1', 'lag_2']].fillna(0))

# Combine all features
data_scaled_combined = np.hstack((data_scaled, other_features_scaled))

# Prepare sequences
seq_length = 30 # Can be optimized or adjusted

X = []
y = []
for i in range(len(data_scaled_combined) - seq_length - jour_pred + 1):
    X.append(data_scaled_combined[i:i+seq_length])
    y.append(data_scaled[i+seq_length:i+seq_length+jour_pred, 0])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape y_train and y_test to match the prediction horizon
y_train = y_train.reshape(-1, jour_pred)
y_test = y_test.reshape(-1, jour_pred)

# Create LSTM model
model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(jour_pred, activation='linear'))  # Output layer to predict for the next 30 days

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=250, batch_size=64, verbose=1, validation_split=0.2)

# Predictions for the test set
y_pred_test = model.predict(X_test)

# Predictions for the training set (for performance comparison)
y_pred_train = model.predict(X_train)

# Combine training and test predictions for comparison
y_combined = np.concatenate([y_train, y_test], axis=0)
y_pred_combined = np.concatenate([y_pred_train, y_pred_test], axis=0)

# Inverse transform the scaled predictions and actual values for comparison
y_pred_combined_original = scaler_inflows.inverse_transform(y_pred_combined.flatten().reshape(-1, 1)).flatten()

# Function to calculate performance metrics (NSE, R², RMSE, MAE, NRMSE)
def calculate_metrics(y_true, y_pred):
    # Flatten arrays
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # Calculate averages
    y_mean = np.mean(y_true_f)
    y_pred_mean = np.mean(y_pred_f)

    # NSE Calculation
    numerator_nse = np.sum((y_true_f - y_pred_f) ** 2)
    denominator_nse = np.sum((y_true_f - y_mean) ** 2)
    nse = 1 - (numerator_nse / denominator_nse)

    # R² Calculation
    numerator_r2 = np.sum((y_true_f - y_mean) * (y_pred_f - y_pred_mean))
    denominator_r2 = np.sqrt(np.sum((y_true_f - y_mean) ** 2) * np.sum((y_pred_f - y_pred_mean) ** 2))
    r2 = numerator_r2 / denominator_r2

    return nse, r2

# Calculate metrics for the combined data
nse_combined, r2_combined = calculate_metrics(scaler_inflows.inverse_transform(y_combined.flatten().reshape(-1, 1)).flatten(),
                                              y_pred_combined_original)

# Additional performance metrics
rmse_combined = np.sqrt(mean_squared_error(scaler_inflows.inverse_transform(y_combined.flatten().reshape(-1, 1)).flatten(),
                                           y_pred_combined_original))
mae_combined = mean_absolute_error(scaler_inflows.inverse_transform(y_combined.flatten().reshape(-1, 1)).flatten(),
                                   y_pred_combined_original)
nrmse_combined = rmse_combined / (np.max(scaler_inflows.inverse_transform(y_combined.flatten().reshape(-1, 1)).flatten()) - 
                                  np.min(scaler_inflows.inverse_transform(y_combined.flatten().reshape(-1, 1)).flatten()))

# Display the performance metrics
print(f"Metrics:")
print(f"  Nash-Sutcliffe Efficiency (NSE): {nse_combined}")
print(f"  Coefficient de Détermination (R²): {r2_combined}")
print(f"  Mean Absolute Error (MAE): {mae_combined}")
print(f"  Root Mean Squared Error (RMSE): {rmse_combined}")
print(f"  Normalized RMSE (NRMSE): {nrmse_combined}")

# Future predictions
last_sequence = data_scaled_combined[-seq_length:]
last_sequence = last_sequence.reshape(1, seq_length, -1)

# Make predictions for the future
y_pred_next = model.predict(last_sequence)
y_pred_next_original = scaler_inflows.inverse_transform(y_pred_next)  # Inverse normalization
y_pred_next_original = np.maximum(0, y_pred_next_original)  # Set negative values to zero

# Generate timestamps for future predictions
last_timestamp = inflow_data.index[-1]
predicted_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=jour_pred, freq='D')

# Create a DataFrame for future predictions
predictions_df = pd.DataFrame({
    'Date': predicted_timestamps,
    'Predictions': y_pred_next_original.flatten()
})

# Display future predictions
print(predictions_df)

# Comparison with actual values
actual_data = pd.read_csv('R1_org.csv', parse_dates=['Date'], index_col='Date')
actual_values_df = actual_data.loc[predictions_df['Date']]

# Create a comparison DataFrame
comparison_df = pd.merge(predictions_df, actual_values_df, left_on='Date', right_index=True, how='inner')
comparison_df.rename(columns={'Inflows': 'Actual'}, inplace=True)

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Date'], comparison_df['Predictions'], label='Predictions', color='blue', linestyle='--')
plt.plot(comparison_df['Date'], comparison_df['Actual'], label='Actual Values', color='red')
plt.xlabel('Date')
plt.ylabel('Inflows')
#plt.ylim(0, 50) 
plt.title('Predictions vs Actual Inflows')
plt.legend()
plt.show()
plt.savefig('inflow_LSTM_aout_R1.png')

