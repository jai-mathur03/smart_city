import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet

# Load the dataset
print("Loading dataset...")
try:
    df = pd.read_csv('jaipur_waste_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    print(f"Dataset loaded with {len(df)} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
except FileNotFoundError:
    print("Error: Dataset file not found. Please run dataset_generator.py first.")
    exit()

# Feature Engineering with improved lag feature handling
def create_features(df):
    """Create features for modeling while preserving maximum data"""
    # Make copy of original columns
    original_cols = df.columns
    
    # Create lag features
    print("Creating lag features...")
    for lag in [1, 2, 3, 7]:
        df[f'waste_lag{lag}'] = df['total_waste_kg'].shift(lag)
    
    # Rolling features
    df['waste_rolling_avg7'] = df['total_waste_kg'].rolling(window=7).mean()
    
    # Binary festival flag
    df['is_festival'] = df['festival'].apply(lambda x: 1 if x != '' else 0)
    
    # Season dummies
    if 'season' in df.columns:
        df = pd.get_dummies(df, columns=['season'], prefix='season')
    
    # Only drop NA rows created by our feature engineering
    initial_count = len(df)
    df.dropna(subset=['waste_lag7', 'waste_rolling_avg7'], inplace=True)
    print(f"Kept {len(df)}/{initial_count} rows after feature engineering")
    
    return df

print("\nCreating features...")
df = create_features(df)

# Validate we have sufficient data
if len(df) < 100:
    print("\nERROR: Insufficient data after feature engineering!")
    print("Only {len(df)} rows remaining - check dataset generation")
    exit()

# Basic time series plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['total_waste_kg'])
plt.title('Waste Generation Time Series')
plt.xlabel('Date')
plt.ylabel('Waste (kg)')
plt.grid(True)
plt.savefig('waste_time_series.png')
plt.close()

# Prepare data for ML
target = 'total_waste_kg'
features = ['domestic_tourists', 'foreign_tourists', 'temperature', 'rainfall',
            'humidity', 'festival_impact', 'is_weekend', 'waste_lag1',
            'waste_lag2', 'waste_lag3', 'waste_lag7', 'waste_rolling_avg7',
            'is_festival']

# Add season columns if they exist
season_cols = [col for col in df.columns if col.startswith('season_')]
features.extend(season_cols)

X = df[features]
y = df[target]

# Time-series aware split (20% for testing, minimum 30 days)
test_size = max(int(len(df) * 0.2), 30)
split_point = len(df) - test_size
split_date = df.index[split_point]

X_train = X.iloc[:split_point]
X_test = X.iloc[split_point:]
y_train = y.iloc[:split_point]
y_test = y.iloc[split_point:]

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.index.min()} to {X_train.index.max()})")
print(f"Test set: {X_test.shape[0]} samples ({X_test.index.min()} to {X_test.index.max()})")

# Scale features
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

# 1. Random Forest Model
print("\nTraining Random Forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=7, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print(f"Random Forest Performance:")
print(f"RMSE: {rf_rmse:.2f} kg")
print(f"MAE: {rf_mae:.2f} kg")
print(f"R2 Score: {rf_r2:.2f}")

# 2. XGBoost Model
print("\nTraining XGBoost...")
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scaled, y_train)

# Predictions
y_pred_xgb = xgb.predict(X_test_scaled)

# Evaluate
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_r2 = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Performance:")
print(f"RMSE: {xgb_rmse:.2f} kg")
print(f"MAE: {xgb_mae:.2f} kg")
print(f"R2 Score: {xgb_r2:.2f}")

# 3. Prophet Model (only if sufficient data)
has_prophet = False
prophet_rmse = np.nan
prophet_mae = np.nan
prophet_r2 = np.nan
y_pred_prophet = np.zeros(len(y_test))

if len(df) >= 365:  # At least 1 year of data
    print("\nPreparing Prophet model...")
    try:
        prophet_df = df.reset_index()[['date', 'total_waste_kg'] + features]
        prophet_df.columns = ['ds', 'y'] + features
        
        # Split for Prophet
        prophet_train = prophet_df.iloc[:split_point].copy()
        prophet_test = prophet_df.iloc[split_point:].copy()
        
        # Configure Prophet
        prophet_model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add important regressors
        for feature in ['domestic_tourists', 'foreign_tourists', 'temperature', 'rainfall']:
            prophet_model.add_regressor(feature)
        
        # Fit model
        prophet_model.fit(prophet_train)
        
        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=len(prophet_test))
        future = future.merge(prophet_df[['ds'] + features], on='ds', how='left')
        
        # Forecast
        forecast = prophet_model.predict(future)
        y_pred_prophet = forecast.iloc[-len(prophet_test):]['yhat'].values
        
        # Evaluate
        prophet_rmse = np.sqrt(mean_squared_error(y_test, y_pred_prophet))
        prophet_mae = mean_absolute_error(y_test, y_pred_prophet)
        prophet_r2 = r2_score(y_test, y_pred_prophet)
        
        print(f"Prophet Performance:")
        print(f"RMSE: {prophet_rmse:.2f} kg")
        print(f"MAE: {prophet_mae:.2f} kg")
        print(f"R2 Score: {prophet_r2:.2f}")
        has_prophet = True
        
    except Exception as e:
        print(f"Prophet model failed: {str(e)}")
        has_prophet = False
else:
    print("\nSkipping Prophet - insufficient data (needs at least 1 year)")

# 4. Ensemble Model - only include models with valid R2 scores
# 4. Ensemble Model - only include models with valid R2 scores
print("\nCreating Ensemble Model...")
models_to_ensemble = []
model_r2_scores = []

if not np.isnan(rf_r2) and rf_r2 > 0:
    models_to_ensemble.append(('RF', y_pred_rf))
    model_r2_scores.append(rf_r2)
if not np.isnan(xgb_r2) and xgb_r2 > 0:
    models_to_ensemble.append(('XGB', y_pred_xgb))
    model_r2_scores.append(xgb_r2)
if has_prophet and not np.isnan(prophet_r2) and prophet_r2 > 0:
    models_to_ensemble.append(('Prophet', y_pred_prophet))
    model_r2_scores.append(prophet_r2)

if not models_to_ensemble:
    print("No models with positive R2 scores. Using best individual model.")
    best_r2 = max(rf_r2, xgb_r2, prophet_r2 if has_prophet else -np.inf)
    if best_r2 == rf_r2:
        y_pred_ensemble = y_pred_rf
        best_model_name = "Random Forest"
    elif best_r2 == xgb_r2:
        y_pred_ensemble = y_pred_xgb
        best_model_name = "XGBoost"
    else:
        y_pred_ensemble = y_pred_prophet
        best_model_name = "Prophet"
    print(f"Using {best_model_name} as ensemble model")
else:
    print(f"Ensemble combining {[name for name, _ in models_to_ensemble]}")
    # Calculate weights based on R2 scores
    weights = np.array(model_r2_scores) / sum(model_r2_scores)
    y_pred_ensemble = np.average([pred for _, pred in models_to_ensemble], axis=0, weights=weights)

# Evaluate Ensemble
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
ensemble_mae = mean_absolute_error(y_test, y_pred_ensemble)
ensemble_r2 = r2_score(y_test, y_pred_ensemble)

print(f"Ensemble Performance:")
print(f"RMSE: {ensemble_rmse:.2f} kg")
print(f"MAE: {ensemble_mae:.2f} kg")
print(f"R2 Score: {ensemble_r2:.2f}")
# Compare all models
results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Prophet', 'Ensemble'],
    'RMSE': [rf_rmse, xgb_rmse, prophet_rmse if has_prophet else np.nan, ensemble_rmse],
    'MAE': [rf_mae, xgb_mae, prophet_mae if has_prophet else np.nan, ensemble_mae],
    'R2': [rf_r2, xgb_r2, prophet_r2 if has_prophet else np.nan, ensemble_r2]
})
print("\nModel Comparison:")
print(results)

# Plot feature importance
plt.figure(figsize=(10, 6))
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
sns.barplot(x=importances[sorted_idx], y=np.array(features)[sorted_idx])
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred_ensemble, label='Ensemble Predicted', color='red', alpha=0.7)
plt.title('Actual vs Predicted Waste Generation')
plt.xlabel('Date')
plt.ylabel('Waste (kg)')
plt.legend()
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.close()

# Save predictions
predictions = pd.DataFrame({
    'Date': y_test.index,
    'Actual': y_test.values,
    'RF_Predicted': y_pred_rf,
    'XGB_Predicted': y_pred_xgb
})

if has_prophet:
    predictions['Prophet_Predicted'] = y_pred_prophet
predictions['Ensemble_Predicted'] = y_pred_ensemble
predictions.to_csv('waste_predictions.csv', index=False)

# Future predictions
print("\nGenerating future predictions...")
future_steps = 30

def predict_future(model, last_features, steps=30):
    """Generate future predictions with realistic variability"""
    future_preds = []
    current_features = last_features.copy()
    
    # Get variation from training data
    y_std = np.std(y_train) * 0.1  # 10% of std dev
    
    for _ in range(steps):
        # Predict next value
        next_pred = model.predict(current_features.values.reshape(1, -1))[0]
        
        # Add realistic variation
        next_pred += np.random.normal(0, y_std)
        next_pred = max(next_pred, 0)  # Ensure non-negative
        future_preds.append(next_pred)
        
        # Update features for next prediction
        if 'waste_lag1' in current_features.index:
            current_features['waste_lag1'] = next_pred
        if 'waste_lag2' in current_features.index:
            current_features['waste_lag2'] = current_features['waste_lag1']
        if 'waste_lag3' in current_features.index:
            current_features['waste_lag3'] = current_features['waste_lag2']
        if 'waste_lag7' in current_features.index and len(future_preds) >= 7:
            current_features['waste_lag7'] = future_preds[-7]
            
        # Update rolling average
        if 'waste_rolling_avg7' in current_features.index:
            if len(future_preds) < 7:
                # Combine historical and predicted values
                hist = y_train.iloc[-(7-len(future_preds)):].values
                combined = np.concatenate([hist, np.array(future_preds)])
                current_features['waste_rolling_avg7'] = combined.mean()
            else:
                current_features['waste_rolling_avg7'] = np.mean(future_preds[-7:])
    
    return future_preds

# Generate future predictions using best model
best_r2 = max(rf_r2, xgb_r2, ensemble_r2)
if best_r2 == rf_r2:
    print("Using Random Forest for future predictions")
    future_preds = predict_future(rf, X_test_scaled.iloc[-1], future_steps)
elif best_r2 == xgb_r2:
    print("Using XGBoost for future predictions")
    future_preds = predict_future(xgb, X_test_scaled.iloc[-1], future_steps)
else:
    print("Using Ensemble for future predictions")
    # For ensemble, average predictions from all good models
    all_preds = []
    if not np.isnan(rf_r2) and rf_r2 > 0:
        all_preds.append(predict_future(rf, X_test_scaled.iloc[-1], future_steps))
    if not np.isnan(xgb_r2) and xgb_r2 > 0:
        all_preds.append(predict_future(xgb, X_test_scaled.iloc[-1], future_steps))
    if has_prophet and not np.isnan(prophet_r2) and prophet_r2 > 0:
        # Prophet future handling would be different
        pass
    
    if all_preds:
        future_preds = np.mean(all_preds, axis=0)
    else:
        print("No valid models for future prediction. Using XGBoost as fallback")
        future_preds = predict_future(xgb, X_test_scaled.iloc[-1], future_steps)

# Create future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)

# Save future predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Waste_kg': future_preds
})
future_df.to_csv('future_waste_predictions.csv', index=False)

# Plot future predictions
plt.figure(figsize=(12, 6))
# Last 60 days of actual data
plt.plot(df.index[-60:], df['total_waste_kg'][-60:], label='Historical', color='blue')
# Future predictions
plt.plot(future_dates, future_preds, label='Predicted', color='red', linestyle='--')
plt.title('Future Waste Generation Forecast')
plt.xlabel('Date')
plt.ylabel('Waste (kg)')
plt.legend()
plt.grid(True)
plt.savefig('future_predictions.png')
plt.close()

print("\nNext 7 days predictions:")
print(future_df.head(7))
print("\nAll models trained and evaluated successfully!")
print("Visualizations and predictions saved to files.")