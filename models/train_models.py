"""
Model Training Module - Clean Rewrite
Trains SARIMA, Prophet, XGBoost, LSTM per state
Picks best model by MAPE on last 8 weeks validation
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "lag_1", "lag_4", "lag_13",
    "roll_mean_4", "roll_std_4",
    "roll_mean_8", "roll_std_8",
    "week_of_year", "month", "quarter",
    "day_of_week", "holiday_flag",
]

LOOKBACK = 13


# ── Metrics ──────────────────────────────────────────────

def mape_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 999.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred):
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAPE": round(mape_score(y_true, y_pred), 2),
    }


# ── Data prep for one state ───────────────────────────────

def get_state_data(df, state, val_weeks=8):
    sdf = df[df["State"] == state].sort_values("Date").copy()
    # Ensure Sales has no nulls
    sdf["Sales"] = pd.to_numeric(sdf["Sales"], errors="coerce")
    sdf["Sales"] = sdf["Sales"].ffill().bfill().fillna(0)
    # Fill feature columns
    fill_vals = sdf[FEATURE_COLS].mean().fillna(0)
    sdf[FEATURE_COLS] = sdf[FEATURE_COLS].fillna(fill_vals)
    train = sdf.iloc[:-val_weeks].copy()
    val   = sdf.iloc[-val_weeks:].copy()
    return sdf, train, val


# ── SARIMA ───────────────────────────────────────────────

def train_sarima(train_series, val_steps=8):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    preds = fit.forecast(steps=val_steps)
    return fit, np.maximum(np.array(preds), 0)


# ── Prophet ──────────────────────────────────────────────

def train_prophet(train_df, val_steps=8):
    from prophet import Prophet
    pdf = train_df[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="multiplicative")
    m.fit(pdf)
    future = m.make_future_dataframe(periods=val_steps, freq="W")
    fc = m.predict(future)
    preds = fc["yhat"].tail(val_steps).values
    return m, np.maximum(preds, 0)


# ── XGBoost ──────────────────────────────────────────────

def train_xgboost(train, val):
    from xgboost import XGBRegressor
    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["Sales"].fillna(0)
    X_val   = val[FEATURE_COLS].fillna(0)

    m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    m.fit(X_train, y_train)
    preds = m.predict(X_val)
    return m, np.maximum(preds, 0)


# ── LSTM ─────────────────────────────────────────────────

def train_lstm(train_series, val_steps=8):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    if len(X) < 10:
        raise ValueError("Not enough data for LSTM")

    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=16,
              validation_split=0.1, callbacks=[es], verbose=0)

    history = list(scaled[-LOOKBACK:])
    preds_scaled = []
    for _ in range(val_steps):
        seq = np.array(history[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        p = float(model.predict(seq, verbose=0)[0][0])
        preds_scaled.append(p)
        history.append(p)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return model, scaler, np.maximum(preds, 0)


# ── Main training loop ────────────────────────────────────

def train_all_models(df, model_dir="models", val_weeks=8):
    os.makedirs(model_dir, exist_ok=True)
    summary_rows = []
    states = sorted(df["State"].unique())
    print(f"\nTraining models for {len(states)} states...\n")

    for idx, state in enumerate(states, 1):
        print(f"[{idx}/{len(states)}] {state}")

        try:
            sdf, train, val = get_state_data(df, state, val_weeks)
        except Exception as e:
            print(f"  ⚠ Data prep failed: {e}")
            continue

        print(f"  Data: {len(train)} train rows, {len(val)} val rows, "
              f"Sales range: {sdf['Sales'].min():.0f} - {sdf['Sales'].max():.0f}")

        if len(train) < 20 or len(val) < val_weeks:
            print(f"  ⚠ Skipping: not enough rows")
            continue

        y_val = val["Sales"].values
        results = {}

        # 1. SARIMA
        try:
            fit, preds = train_sarima(train["Sales"].values, val_weeks)
            results["SARIMA"] = {"metrics": evaluate(y_val, preds), "model": fit, "preds": preds}
            print(f"  SARIMA  → MAPE: {results['SARIMA']['metrics']['MAPE']:.1f}%")
        except Exception as e:
            print(f"  SARIMA  → FAILED: {e}")

        # 2. Prophet
        try:
            m, preds = train_prophet(train, val_weeks)
            results["Prophet"] = {"metrics": evaluate(y_val, preds), "model": m, "preds": preds}
            print(f"  Prophet → MAPE: {results['Prophet']['metrics']['MAPE']:.1f}%")
        except Exception as e:
            print(f"  Prophet → FAILED: {e}")

        # 3. XGBoost
        try:
            m, preds = train_xgboost(train, val)
            results["XGBoost"] = {"metrics": evaluate(y_val, preds), "model": m, "preds": preds}
            print(f"  XGBoost → MAPE: {results['XGBoost']['metrics']['MAPE']:.1f}%")
        except Exception as e:
            print(f"  XGBoost → FAILED: {e}")

        # 4. LSTM
        try:
            m, scaler, preds = train_lstm(train["Sales"].values, val_weeks)
            results["LSTM"] = {"metrics": evaluate(y_val, preds), "model": m,
                               "scaler": scaler, "preds": preds}
            print(f"  LSTM    → MAPE: {results['LSTM']['metrics']['MAPE']:.1f}%")
        except Exception as e:
            print(f"  LSTM    → FAILED: {e}")

        if not results:
            print(f"  ⚠ All models failed for {state}")
            continue

        # Pick best by MAPE
        best_name = min(results, key=lambda k: results[k]["metrics"]["MAPE"])
        best = results[best_name]
        print(f"  ✅ Best: {best_name} (MAPE {best['metrics']['MAPE']:.1f}%)\n")

        # Save
        state_key = state.replace(" ", "_")
        artifact = {
            "state": state,
            "model_name": best_name,
            "metrics": best["metrics"],
            "model": best["model"],
        }
        if best_name == "LSTM":
            artifact["scaler"] = best["scaler"]
        joblib.dump(artifact, f"{model_dir}/{state_key}_best_model.pkl")

        for model_name, res in results.items():
            summary_rows.append({
                "state": state, "model": model_name,
                "best": model_name == best_name,
                **res["metrics"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{model_dir}/model_comparison.csv", index=False)
    print(f"\nModel comparison saved to {model_dir}/model_comparison.csv")
    return summary_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_preprocessing import prepare_data
    df = prepare_data("data/sales_data.xlsx")
    summary = train_all_models(df)
    if not summary.empty and "best" in summary.columns:
        print(summary[summary["best"] == True][["state","model","MAPE"]].to_string(index=False))