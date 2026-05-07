"""
Forecasting Inference Module
Loads saved best model per state and generates 8-week future forecasts.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

LOOKBACK = 13
FEATURE_COLS = [
    "lag_1", "lag_4", "lag_13",
    "roll_mean_4", "roll_std_4",
    "roll_mean_8", "roll_std_8",
    "week_of_year", "month", "quarter",
    "day_of_week", "holiday_flag",
]


def get_future_dates(last_date: pd.Timestamp, n_weeks: int = 8) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq="W-MON")


def forecast_sarima(model, n_weeks: int = 8) -> np.ndarray:
    preds = model.forecast(steps=n_weeks)
    return np.maximum(preds.values, 0)


def forecast_prophet(model, last_date: pd.Timestamp, n_weeks: int = 8) -> np.ndarray:
    future = model.make_future_dataframe(periods=n_weeks, freq="W")
    forecast = model.predict(future)
    return np.maximum(forecast["yhat"].tail(n_weeks).values, 0)


def forecast_xgboost(model, state_df: pd.DataFrame, n_weeks: int = 8) -> np.ndarray:
    import holidays as hols
    us_holidays = hols.US(years=range(2018, 2026))

    history = state_df.copy().sort_values("Date")
    preds = []

    for i in range(n_weeks):
        next_date = history["Date"].max() + pd.Timedelta(weeks=1)
        sales_series = history["Sales"]

        # Build feature row
        feat = {
            "lag_1":         sales_series.iloc[-1],
            "lag_4":         sales_series.iloc[-4] if len(sales_series) >= 4 else sales_series.mean(),
            "lag_13":        sales_series.iloc[-13] if len(sales_series) >= 13 else sales_series.mean(),
            "roll_mean_4":   sales_series.iloc[-4:].mean(),
            "roll_std_4":    sales_series.iloc[-4:].std(),
            "roll_mean_8":   sales_series.iloc[-8:].mean(),
            "roll_std_8":    sales_series.iloc[-8:].std(),
            "week_of_year":  next_date.isocalendar()[1],
            "month":         next_date.month,
            "quarter":       (next_date.month - 1) // 3 + 1,
            "day_of_week":   next_date.dayofweek,
            "holiday_flag":  int(any(
                (next_date + pd.Timedelta(days=j)) in us_holidays for j in range(7)
            )),
        }

        X = pd.DataFrame([feat])
        p = float(model.predict(X)[0])
        p = max(p, 0)
        preds.append(p)

        # Append predicted row to history for next iteration
        new_row = pd.DataFrame({"Date": [next_date], "Sales": [p]})
        history = pd.concat([history, new_row], ignore_index=True)

    return np.array(preds)


def forecast_lstm(model, scaler, state_df: pd.DataFrame, n_weeks: int = 8) -> np.ndarray:
    sales = state_df["Sales"].values
    scaled = scaler.transform(sales.reshape(-1, 1)).flatten()
    history = list(scaled[-LOOKBACK:])
    preds_scaled = []

    for _ in range(n_weeks):
        seq = np.array(history[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = model.predict(seq, verbose=0)[0][0]
        preds_scaled.append(p)
        history.append(p)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return np.maximum(preds, 0)


def forecast_state(state: str, df: pd.DataFrame, model_dir: str = "models", n_weeks: int = 8) -> pd.DataFrame:
    """Load best model for a state and return 8-week forecast DataFrame."""
    state_key = state.replace(" ", "_")
    model_path = f"{model_dir}/{state_key}_best_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for state: {state}. Run train_models.py first.")

    artifact = joblib.load(model_path)
    model_name = artifact["model_name"]
    model = artifact["model"]

    state_df = df[df["State"] == state].sort_values("Date").dropna(subset=["Sales"])
    last_date = state_df["Date"].max()
    future_dates = get_future_dates(last_date, n_weeks)

    if model_name == "SARIMA":
        preds = forecast_sarima(model, n_weeks)
    elif model_name == "Prophet":
        preds = forecast_prophet(model, last_date, n_weeks)
    elif model_name == "XGBoost":
        preds = forecast_xgboost(model, state_df, n_weeks)
    elif model_name == "LSTM":
        scaler = artifact["scaler"]
        preds = forecast_lstm(model, scaler, state_df, n_weeks)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    result = pd.DataFrame({
        "state": state,
        "forecast_date": future_dates,
        "predicted_sales": np.round(preds, 2),
        "model_used": model_name,
        "MAE": artifact["metrics"]["MAE"],
        "RMSE": artifact["metrics"]["RMSE"],
        "MAPE": artifact["metrics"]["MAPE"],
    })
    return result


def forecast_all_states(df: pd.DataFrame, model_dir: str = "models", n_weeks: int = 8) -> pd.DataFrame:
    """Generate 8-week forecasts for all states with trained models."""
    model_files = [f for f in os.listdir(model_dir) if f.endswith("_best_model.pkl")]
    if not model_files:
        raise RuntimeError("No trained models found. Run train_models.py first.")

    all_forecasts = []
    for model_file in sorted(model_files):
        state = model_file.replace("_best_model.pkl", "").replace("_", " ")
        try:
            result = forecast_state(state, df, model_dir, n_weeks)
            all_forecasts.append(result)
        except Exception as e:
            print(f"  Warning: Could not forecast {state}: {e}")

    return pd.concat(all_forecasts, ignore_index=True)


if __name__ == "__main__":
    from utils.data_preprocessing import prepare_data
    df = prepare_data("data/sales_data.xlsx")
    forecasts = forecast_all_states(df)
    print(forecasts.head(20))
    forecasts.to_csv("models/all_forecasts.csv", index=False)
    print("Forecasts saved to models/all_forecasts.csv")
