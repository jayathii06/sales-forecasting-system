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


def forecast_xgboost_walk(model, state_df, n_weeks=8):
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


def retrain_xgboost(df, state, n_weeks=8):
    from xgboost import XGBRegressor
    sdf = df[df["State"] == state].sort_values("Date").copy()
    sdf["Sales"] = sdf["Sales"].ffill().bfill().fillna(0)
    fill_vals = sdf[FEATURE_COLS].mean().fillna(0)
    sdf[FEATURE_COLS] = sdf[FEATURE_COLS].fillna(fill_vals)
    train = sdf.iloc[:-n_weeks] if len(sdf) > n_weeks else sdf
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05,
                       max_depth=4, random_state=42, verbosity=0)
    xgb.fit(train[FEATURE_COLS].fillna(0), train["Sales"])
    return xgb, sdf


def forecast_state(state, df, model_dir="models", n_weeks=8):
    state_key = state.replace(" ", "_")
    model_path = f"{model_dir}/{state_key}_best_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for state: {state}")

    artifact   = joblib.load(model_path)
    model_name = artifact["model_name"]
    model      = artifact["model"]

    state_df = df[df["State"] == state].sort_values("Date").copy()
    state_df["Sales"] = state_df["Sales"].ffill().bfill().fillna(0)
    last_date    = state_df["Date"].max()
    future_dates = get_future_dates(last_date, n_weeks)

    preds = None

    try:
        if model_name == "SARIMA":
            preds = np.maximum(np.array(model.forecast(steps=n_weeks)), 0)

        elif model_name == "Prophet":
            future = model.make_future_dataframe(periods=n_weeks, freq="W")
            fc     = model.predict(future)
            preds  = np.maximum(fc["yhat"].tail(n_weeks).values, 0)

        elif model_name == "XGBoost":
            preds = forecast_xgboost_walk(model, state_df, n_weeks)

        elif model_name == "LSTM":
            scaler = artifact["scaler"]
            sales  = state_df["Sales"].values
            scaled = scaler.transform(sales.reshape(-1, 1)).flatten()
            history = list(scaled[-LOOKBACK:])
            preds_scaled = []
            for _ in range(n_weeks):
                seq = np.array(history[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
                p   = float(model.predict(seq, verbose=0)[0][0])
                preds_scaled.append(p)
                history.append(p)
            preds = np.maximum(
                scaler.inverse_transform(
                    np.array(preds_scaled).reshape(-1, 1)
                ).flatten(), 0
            )

    except Exception as e:
        print(f"Model {model_name} failed: {str(e)} - using XGBoost fallback")
        preds = None

    if preds is None:
        xgb, sdf = retrain_xgboost(df, state, n_weeks)
        preds     = forecast_xgboost_walk(xgb, sdf, n_weeks)
        model_name = "XGBoost"

    return pd.DataFrame({
        "state":           state,
        "forecast_date":   future_dates,
        "predicted_sales": np.round(preds, 2),
        "model_used":      model_name,
        "MAE":             artifact["metrics"]["MAE"],
        "RMSE":            artifact["metrics"]["RMSE"],
        "MAPE":            artifact["metrics"]["MAPE"],
    })


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
            print(f"Warning: {state}: {str(e)}")
    return pd.concat(all_forecasts, ignore_index=True)
