"""
Quick Demo Script — trains XGBoost for California & Texas, runs 8-week forecast.
Run: python demo.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mape(y_true, y_pred):
    mask = np.array(y_true) != 0
    return np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100


FEAT = ["lag_1","lag_4","lag_13","roll_mean_4","roll_std_4",
        "roll_mean_8","roll_std_8","week_of_year","month",
        "quarter","day_of_week","holiday_flag"]
VAL_WEEKS = 8


def prep_state(df, state):
    """Get state data, fill NaN lag features with column mean (no leakage on test)."""
    sdf = df[df["State"] == state].sort_values("Date").copy()
    # Fill NaN Sales first
    sdf["Sales"] = sdf["Sales"].ffill().bfill().fillna(0)
    train = sdf.iloc[:-VAL_WEEKS].copy()
    val   = sdf.iloc[-VAL_WEEKS:].copy()
    # Fill NaN in feature cols with mean of training portion
    fill_vals = train[FEAT].mean().fillna(0)
    train[FEAT] = train[FEAT].fillna(fill_vals)
    val[FEAT]   = val[FEAT].fillna(fill_vals)
    # Drop any remaining NaN rows in training Sales
    train = train.dropna(subset=["Sales"])
    val   = val.dropna(subset=["Sales"])
    return train, val, sdf


def demo():
    print("=" * 60)
    print("  SALES FORECASTING SYSTEM — QUICK DEMO")
    print("=" * 60)

    # ── Step 1: Load & preprocess
    print("\n[1] Loading and preprocessing data...")
    from utils.data_preprocessing import prepare_data
    df = prepare_data("data/sales_data.xlsx")
    print(f"    Shape: {df.shape}")
    print(f"    States: {df['State'].nunique()}")
    print(f"    Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # ── Step 2: Feature check
    print("\n[2] Feature engineering check (California sample)...")
    ca = df[df["State"] == "California"].sort_values("Date")
    sample = ca[["Date","Sales","lag_1","lag_4","month","holiday_flag"]].iloc[14:17]
    print(sample.to_string(index=False))

    # ── Step 3: Train XGBoost for 2 demo states
    print("\n[3] Training XGBoost for California & Texas...")
    from xgboost import XGBRegressor

    demo_results = {}
    for state in ["California", "Texas"]:
        train, val, sdf = prep_state(df, state)

        if len(train) < 10:
            print(f"    {state}: not enough data, skipping")
            continue

        model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                             max_depth=4, random_state=42, verbosity=0)
        model.fit(train[FEAT], train["Sales"])
        preds = np.maximum(model.predict(val[FEAT]), 0)

        mae   = mean_absolute_error(val["Sales"], preds)
        _mape = mape(val["Sales"].values, preds)
        print(f"    {state:15s} → MAE: ${mae:,.0f}  |  MAPE: {_mape:.2f}%")
        demo_results[state] = {"model": model, "sdf": sdf, "mae": mae, "mape": _mape}

    # ── Step 4: Future 8-week forecast
    print("\n[4] Generating 8-week future forecasts...")
    import holidays as hols
    us_holidays = hols.US(years=range(2018, 2027))

    for state, res in demo_results.items():
        history = res["sdf"].copy()
        fill_vals = history[FEAT].mean()
        future_preds = []

        for _ in range(VAL_WEEKS):
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
            p = max(float(res["model"].predict(pd.DataFrame([feat]))[0]), 0)
            future_preds.append((next_date, p))
            history = pd.concat([history, pd.DataFrame({"Date": [next_date], "Sales": [p]})],
                                ignore_index=True)

        print(f"\n    {state} — 8-Week Forecast:")
        print(f"    {'Week':<6} {'Date':<14} {'Predicted Sales':>18}")
        print(f"    {'-'*40}")
        for i, (d, p) in enumerate(future_preds, 1):
            print(f"    {i:<6} {str(d.date()):<14} ${p:>17,.0f}")

    # ── Step 5: Save chart
    print("\n[5] Saving forecast chart to output/demo_forecast.png ...")
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(1, len(demo_results), figsize=(16, 5))
    if len(demo_results) == 1:
        axes = [axes]

    for ax, (state, res) in zip(axes, demo_results.items()):
        hist = res["sdf"].tail(24)
        future_dates, future_vals = [], []
        history = res["sdf"].copy()

        for i in range(VAL_WEEKS):
            nd = history["Date"].max() + pd.Timedelta(weeks=1)
            s  = history["Sales"]
            feat = {
                "lag_1": s.iloc[-1], "lag_4": s.iloc[-4] if len(s)>=4 else s.mean(),
                "lag_13": s.iloc[-13] if len(s)>=13 else s.mean(),
                "roll_mean_4": s.iloc[-4:].mean(),
                "roll_std_4":  s.iloc[-4:].std() if len(s)>=4 else 0,
                "roll_mean_8": s.iloc[-8:].mean(),
                "roll_std_8":  s.iloc[-8:].std() if len(s)>=8 else 0,
                "week_of_year": int(nd.isocalendar()[1]),
                "month": nd.month, "quarter": (nd.month-1)//3+1,
                "day_of_week": nd.dayofweek,
                "holiday_flag": int(any((nd+pd.Timedelta(days=j)) in us_holidays for j in range(7))),
            }
            p = max(float(res["model"].predict(pd.DataFrame([feat]))[0]), 0)
            future_dates.append(nd)
            future_vals.append(p)
            history = pd.concat([history, pd.DataFrame({"Date":[nd],"Sales":[p]})], ignore_index=True)

        ax.plot(hist["Date"], hist["Sales"]/1e6, color="#1565C0",
                linewidth=2, label="Historical", marker="o", markersize=3)
        ax.plot(future_dates, np.array(future_vals)/1e6, color="#E53935",
                linewidth=2.5, linestyle="--", label="8-wk Forecast", marker="s", markersize=5)
        ax.fill_between(future_dates,
                        np.array(future_vals)*0.9/1e6,
                        np.array(future_vals)*1.1/1e6,
                        alpha=0.2, color="#E53935")
        ax.axvline(res["sdf"]["Date"].max(), color="gray", linestyle=":", linewidth=1.5)
        ax.set_title(f"{state} | XGBoost | MAPE: {res['mape']:.1f}%", fontweight="bold")
        ax.set_ylabel("Sales (Million USD)")
        ax.legend(fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle("8-Week Sales Forecast — Demo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("output/demo_forecast.png", dpi=150)
    plt.close()
    print("    Saved: output/demo_forecast.png")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("  Next steps:")
    print("  1. Full training:  python run.py --train-only")
    print("  2. Start API:      python run.py --api-only")
    print("  3. Test API:       python test_api.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo()