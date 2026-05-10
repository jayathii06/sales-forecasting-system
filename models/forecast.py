"""
Forecasting Inference Module - Fixed for deployment
Falls back to XGBoost if LSTM fails to load
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


def get_future_dates(last_date, n_weeks=8):
    return pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_weeks, freq="W")


def forecast_sarima(model, n_weeks=8):
    preds = model.forecast(steps=n_weeks)
    return np.maximum(np.array(preds), 0)


def forecast_prophet(model, last_date, n_weeks=8):
    future = model.make_future_dataframe(periods=n_weeks, freq="W")
    forecast = model.predict(future)
    return np.maximum(forecast["yhat"].tail(n_weeks).values, 0)


def forecast_xgboost(model, state_df, n_weeks=8):
    import holidays as hols
    us_holidays = hols.US(years=range(2018, 2030))
    history = state_df.copy().sort_values("Date")
    preds = []
    for i in range(n_weeks):
        next_date = history["Date"].max() + pd.Timedelta(weeks=1)
        s = history["Sales"]
        feat = {
            "lag_1":        s.iloc[-1],
            "lag_4":        s.iloc[-4]  if len(s) >= 4  else s.mean(),
            "lag_13":       s.iloc[-13] if len(s) >= 13 else s.mean(),
            "roll_mean_4":  s.iloc[-4:].mean(),
            "roll_std_4":   s.iloc[-4:].std() if len(s) >= 4 else 0,
            "roll_mean_8":  s.iloc[-8:].mean(),
            "roll_std_8":   s.iloc[-8:].std() if len(s) >= 8 else 0,
            "week_of_year": int(next_date.isocalendar()[1]),
            "month":        next_date.month,
            "quarter":      (next_date.month - 1) // 3 + 1,
            "day_of_week":  next_date.dayofweek,
            "holiday_flag": int(any(
                (next_date + pd.Timedelta(days=j)) in us_holidays for j in range(7)
            )),
        }
        p = max(float(model.predict(pd.DataFrame([feat]))[0]), 0)
        preds.append(p)
        history = pd.concat([history, pd.DataFrame({"Date": [next_date], "Sales": [p]})],
                           ignore_index=True)
    return np.array(preds)


def forecast_lstm(model, scaler, state_df, n_weeks=8):
    sales = state_df["Sales"].values
    scaled = scaler.transform(sales.reshape(-1, 1)).flatten()
    history = list(scaled[-LOOKBACK:])
    preds_scaled = []
    for _ in range(n_weeks):
        seq = np.array(history[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = float(model.predict(seq, verbose=0)[0][0])
        preds_scaled.append(p)
        history.append(p)
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return np.maximum(preds, 0)


def forecast_state(state, df, model_dir="models", n_weeks=8):
    """Load best model and forecast. Falls back to XGBoost if LSTM fails."""
    state_key = state.replace(" ", "_")
    model_path = f"{model_dir}/{state_key}_best_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for state: {state}")

    artifact = joblib.load(model_path)
    model_name = artifact["model_name"]
    model = artifact["model"]

    state_df = df[df["State"] == state].sort_values("Date").copy()
    state_df["Sales"] = state_df["Sales"].ffill().bfill().fillna(0)
    last_date = state_df["Date"].max()
    future_dates = get_future_dates(last_date, n_weeks)

    # Try the best model first, fall back to XGBoost if it fails
    try:
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
            raise ValueError(f"Unknown model: {model_name}")

    except Exception as e:
        print(f"  {model_name} failed for {state}: {e}. Falling back to XGBoost...")
        # Fall back to XGBoost
        from xgboost import XGBRegressor
        feat_cols = FEATURE_COLS
        sdf = df[df["State"] == state].sort_values("Date").copy()
        sdf["Sales"] = sdf["Sales"].ffill().bfill().fillna(0)
        fill_vals = sdf[feat_cols].mean().fillna(0)
        sdf[feat_cols] = sdf[feat_cols].fillna(fill_vals)
        train = sdf.iloc[:-8] if len(sdf) > 8 else sdf
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                           max_depth=4, random_state=42, verbosity=0)
        xgb.fit(train[feat_cols].fillna(0), train["Sales"])
        preds = forecast_xgboost(xgb, state_df, n_weeks)
        model_name = "XGBoost (fallback)"

    result = pd.DataFrame({
        "state":           state,
        "forecast_date":   future_dates,
        "predicted_sales": np.round(preds, 2),
        "model_used":      model_name,
        "MAE":             artifact["metrics"]["MAE"],
        "RMSE":            artifact["metrics"]["RMSE"],
        "MAPE":            artifact["metrics"]["MAPE"],
    })
    return result


def forecast_all_states(df, model_dir="models", n_weeks=8):
    model_files = [f for f in os.listdir(model_dir) if f.endswith("_best_model.pkl")]
    if not model_files:
        raise RuntimeError("No trained models found.")
    all_forecasts = []
    for model_file in sorted(model_files):
        state = model_file.replace("_best_model.pkl", "").replace("_", " ")
        try:
            result = forecast_state(state, df, model_dir, n_weeks)
            all_forecasts.append(result)
        except Exception as e:
            print(f"  Warning: Could not forecast {state}: {e}")
    return pd.concat(all_forecasts, ignore_index=True)