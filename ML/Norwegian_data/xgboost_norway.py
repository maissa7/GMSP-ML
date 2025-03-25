import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Paramètre pour déterminer le nombre d'heures à prédire
jour_pred =4  # 8 intervalles de 3 heures = 24 heures

# Fixer les graines pour la reproductibilité
np.random.seed(42)

# Charger les données CSV pour Inflows, Rainfall, et Environnement (enregistrement toutes les 3 heures)
inflow_data = pd.read_csv('Inflow_Data.csv', parse_dates=['TimeStample'], index_col='TimeStample')
rainfall_data = pd.read_csv('Rainfall_Data.csv', parse_dates=['TimeStample'], index_col='TimeStample')
env_data = pd.read_csv('Environment_Data.csv', parse_dates=['TimeStample'], index_col='TimeStample')

# Fusionner les données sur la colonne 'TimeStample'
data = inflow_data.join([rainfall_data, env_data], how='inner')

# Appliquer le scaler à toutes les colonnes
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

def objective(trial):
    seq_length = trial.suggest_int('seq_length',23, 23)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'objective': 'reg:squarederror',
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),  # L1 régularisation
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),  # L2 régularisation
        'random_state': 42 
    }

    # Create train and test sets
    X = []
    y = []
    for i in range(len(data_scaled) - seq_length - jour_pred + 1):
        X.append(data_scaled[i:i+seq_length])
        #X.append(data_scaled[i:i+seq_length, 1:]) 
        y.append(data_scaled[i+seq_length:i+seq_length+jour_pred, 0])
    
    X = np.array(X)
    y = np.array(y)

    # X_train, X_test = X[:int(len(X) * 0.80)], X[int(len(X) * 0.80):]
    # y_train, y_test = y[:int(len(y) * 0.80)], y[int(len(y) * 0.80):]

    # y_train = y_train.reshape(-1, jour_pred)
    # y_test = y_test.reshape(-1, jour_pred)

    # model = XGBRegressor(**param)
    # #model = XGBRegressor(**best_params, random_state=42)
    # model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    # y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

    # y_test_f= y_test.flatten()
    # y_pred_f= y_pred.flatten()

    # mse = mean_squared_error(y_test_f, y_pred_f)
    # return mse
    tscv = TimeSeriesSplit(n_splits=2)
    mse_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train = y_train.reshape(-1, jour_pred)
        y_test = y_test.reshape(-1, jour_pred)

        model = XGBRegressor(**param)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

        y_test_f = y_test.flatten()
        y_pred_f = y_pred.flatten()

        mse = mean_squared_error(y_test_f, y_pred_f)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# Utiliser Optuna pour trouver les meilleurs hyperparamètres
#study = optuna.create_study(direction='minimize')
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=350)

# Meilleurs hyperparamètres
best_params = study.best_params
best_seq_length = study.best_trial.params['seq_length']

# Entraîner le modèle XGBoost avec les meilleurs hyperparamètres
X = []
y = []
for i in range(len(data_scaled) - best_seq_length - jour_pred + 1):
    X.append(data_scaled[i:i+best_seq_length])
    #X.append(data_scaled[i:i+best_seq_length, 1:]) 
    y.append(data_scaled[i+best_seq_length:i+best_seq_length+jour_pred, 0])  # 'Inflows' remains the target

X = np.array(X)
y = np.array(y)

X_train, X_test = X[:int(len(X) * 0.80)], X[int(len(X) * 0.80):]
y_train, y_test = y[:int(len(y) * 0.80)], y[int(len(y) * 0.80):]

# Reshape y_train and y_test to be 1D arrays
y_train = y_train.reshape(-1, jour_pred)  # Reshape to match the prediction horizon
y_test = y_test.reshape(-1, jour_pred)

best_model = XGBRegressor(**best_params)
best_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)  # Flatten X_train for XGBoost

# Predictions for test data
y_pred_test = best_model.predict(X_test.reshape(X_test.shape[0], -1))  # Flatten X_test for XGBoost

# Predictions for training data (for performance comparison)
y_pred_train = best_model.predict(X_train.reshape(X_train.shape[0], -1))  # Flatten X_train for XGBoost

# Concatenate predictions and true values
y_combined = np.concatenate([y_train, y_test], axis=0)
y_pred_combined = np.concatenate([y_pred_train, y_pred_test], axis=0)



# Flatten combined predictions and true values for performance metrics
y_combined_f= y_combined.flatten()
y_pred_combined_f= y_pred_combined.flatten()

# Function to calculate NSE and R² as per the figure
def calculate_metrics(y_true, y_pred):
    # Flatten arrays
    y_true_f= y_true.flatten()
    y_pred_f= y_pred.flatten()
    

    # Calculate averages
    y_mean = np.mean(y_true_f)
    y_pred_mean = np.mean(y_pred_f)

    # NSE Calculation
    numerator_nse = np.sum((y_true_f- y_pred_f) ** 2)
    denominator_nse = np.sum((y_true_f- y_mean) ** 2)
    nse = 1 - (numerator_nse / denominator_nse)

    # R2 Calculation
    numerator_r2 = np.sum((y_true_f- y_mean) * (y_pred_f- y_pred_mean))
    denominator_r2 = np.sqrt(np.sum((y_true_f- y_mean) ** 2) * np.sum((y_pred_f- y_pred_mean) ** 2))
    r2 = numerator_r2 / denominator_r2

    return nse, r2

# Calculate the metrics for the combined data
nse_combined, r2_combined = calculate_metrics(y_combined, y_pred_combined)

# Additional performance metrics
rmse_combined = np.sqrt(mean_squared_error(y_combined_f, y_pred_combined_f))
mae_combined = mean_absolute_error(y_combined_f, y_pred_combined_f)
nrmse_combined = rmse_combined / (np.max(y_combined_f) - np.min(y_combined_f))  # Normalize by range of observed values

# Display combined metrics
print(f" Metrics:")
print(f"  Nash-Sutcliffe Efficiency (NSE): {nse_combined}")
print(f"  Coefficient de Détermination (R²): {r2_combined}")
print(f"  Mean Absolute Error (MAE): {mae_combined}")
print(f"  Root Mean Squared Error (RMSE): {rmse_combined}")
print(f"  Normalized RMSE (NRMSE): {nrmse_combined}")
print('best params: ', best_params)


# ... (votre code existant)

# Préparer les données pour la dernière prédiction
last_sequence = data_scaled[-best_seq_length:]  # Les 8 dernières valeurs
last_sequence = last_sequence.reshape(1, best_seq_length, -1)  # Reshape pour XGBoost

# Faire les prédictions
y_pred_next = best_model.predict(last_sequence.reshape(1, -1))  # Prédictions pour les 4 prochaines heures

# Générer les timestamps pour les prédictions
last_timestamp = inflow_data.index[-1]  # Dernier timestamp dans les données
predicted_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=3), periods=jour_pred, freq='3H')

# Créer un DataFrame pour les prédictions
predictions_df = pd.DataFrame({
    'Timestamp': predicted_timestamps,
    'Predictions': y_pred_next.flatten()
})

# Afficher les prédictions
print(predictions_df)

# Optionnel : Enregistrer les prédictions dans un fichier CSV
#predictions_df.to_csv('predictions_next_4_hours.csv', index=False)
actual_data = pd.read_csv('Inflow_Data_org.csv', parse_dates=['TimeStample'], index_col='TimeStample')
actual_values_df = actual_data.loc[predictions_df['Timestamp']]

# Étape 5 : Créer un DataFrame de comparaison
comparison_df = pd.merge(predictions_df, actual_values_df, left_on='Timestamp', right_index=True, how='inner')
comparison_df.rename(columns={'Inflows': 'Actual'}, inplace=True)  # Remplacez 'Inflows' par le nom correct de la colonne

# Étape 6 : Affichage du graphique
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Timestamp'], comparison_df['Predictions'], label='Actual Values', color='red')
plt.plot(comparison_df['Timestamp'], comparison_df['Actual'], label='Actual inflow', marker='o', color='blue')
plt.title('A comparison between the actual and predicted inflows')
plt.xlabel('Date')
plt.ylabel('Inflows')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()