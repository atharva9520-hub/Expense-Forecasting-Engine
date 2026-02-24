import pandas as pd
import numpy as np
import json
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def load_and_prepare_data(json_file):
    print("Loading and cleaning JSON data...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    if 'date' not in df.columns or 'total_amount' not in df.columns:
        print("Error: Required columns not found in data.")
        return None

    df = df[['date', 'total_amount']].dropna()
    df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # --- FIX 1 & 3: Tighter Date & Amount Sanity Checks ---
    # Cut off 2019 because the SROIE dataset is mostly empty there
    df = df[(df['date'] >= '2015-01-01') & (df['date'] <= '2018-12-31')]
    
    # Cap outliers at 3000 to prevent massive single-day spikes from confusing the AI
    df = df[df['total_amount'] < 3000] 
    
    df = df.dropna(subset=['date', 'total_amount'])
    
    if df.empty:
        print("Error: No valid data left after cleaning!")
        return None

    # Aggregate Month-on-Month
    monthly_expenses = df.set_index('date').resample('MS')['total_amount'].sum().reset_index()
    monthly_expenses.rename(columns={'date': 'ds', 'total_amount': 'y'}, inplace=True)
    
    # Drop months with $0 (no receipts scanned)
    monthly_expenses = monthly_expenses[monthly_expenses['y'] > 0].reset_index(drop=True)
    
    print(f"Success! Aggregated data into {len(monthly_expenses)} distinct months.")
    return monthly_expenses


def train_and_evaluate(df):
    print("Splitting data into Train and Test sets...")
    
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print("Initializing Prophet Time-Series model (Monthly, tuned seasonality)...")
    # THE GOLDILOCKS FIX: yearly_seasonality is True, but scaled down to 0.1 to prevent wild spikes
    model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, seasonality_prior_scale=0.1)
    model.fit(train_df)
    
    print(f"Predicting on the unseen {len(test_df)} months of test data...")
    future_dates = pd.DataFrame(test_df['ds'])
    forecast = model.predict(future_dates)
    
    # Clip negative predictions to zero
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
    
    # --- CALCULATE RESUME METRICS ---
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    # 1. Absolute Errors
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 2. Percentage Errors
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100 # MAE as a %
    
    # Calculate RMSPE (RMSE as a %) safely to avoid dividing by zero
    #  add a tiny number (1e-10) to the denominator just in case a true value is exactly 0
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-10)))) * 100
    
    # 3. Normalized back to daily equivalent for fair comparison
    daily_equiv_mae = mae / 30.4
    daily_equiv_rmse = rmse / 30.4
    
    print("\n" + "="*50)
    print(" METRICS (Monthly Performance)")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}/mo (Equivalent to ${daily_equiv_mae:.2f}/day)")
    print(f"Root Mean Sq Error (RMSE): ${rmse:.2f}/mo (Equivalent to ${daily_equiv_rmse:.2f}/day)")
    print("-" * 50)
    print(f"Percentage MAE (MAPE):  {mape:.2f}%")
    print(f"Percentage RMSE (RMSPE): {rmspe:.2f}%")
    print("Confidence Interval: 95%")
    print("="*50 + "\n")
    
    # --- PLOT MONTH-ON-MONTH GRAPH ---
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['ds'], train_df['y'], label='Training Data (Past)', color='lightgray', marker='o', linewidth=2)
    plt.plot(test_df['ds'], test_df['y'], label='Actual Monthly Spending', color='blue', marker='o', linewidth=2)
    plt.plot(forecast['ds'], forecast['yhat'], label='AI Prediction', color='red', linestyle='--', linewidth=2, marker='X')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.15, label='95% Confidence Interval')

    plt.title('Month-on-Month Expense Forecasting (Cleaned & Tuned)', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Spent (MYR)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return forecast

if __name__ == "__main__":
    json_path = "/Users/atharvaaserkar/Documents/pp/financial_document_analysis/data/extracted_receipts.json"
    
    df_clean = load_and_prepare_data(json_path)
    
    if df_clean is not None:
        forecast_data = train_and_evaluate(df_clean)
