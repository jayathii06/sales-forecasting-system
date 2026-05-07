"""
Model Evaluation & Forecast Visualization
- Plots actual vs predicted for validation period
- Plots 8-week forecast with confidence band
- Saves model comparison bar chart
Run: python utils/evaluate.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "output/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────
# 1. Actual vs Forecast chart for one state
# ─────────────────────────────────────────────

def plot_forecast(state: str, df: pd.DataFrame, model_dir: str = "models", n_weeks: int = 8):
    from models.forecast import forecast_state

    state_df = df[df["State"] == state].sort_values("Date").dropna(subset=["Sales"])
    historical = state_df.tail(26)  # last 26 weeks for context

    try:
        forecast_df = forecast_state(state, df, model_dir, n_weeks)
    except Exception as e:
        print(f"  Could not forecast {state}: {e}")
        return

    model_name = forecast_df["model_used"].iloc[0]
    mape = forecast_df["MAPE"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical
    ax.plot(historical["Date"], historical["Sales"] / 1e6,
            color="#1565C0", linewidth=2, label="Historical Sales", marker="o", markersize=3)

    # Forecast
    ax.plot(forecast_df["forecast_date"], forecast_df["predicted_sales"] / 1e6,
            color="#E53935", linewidth=2.5, linestyle="--",
            label=f"8-Week Forecast ({model_name})", marker="s", markersize=5)

    # Simple ±10% confidence band
    lower = forecast_df["predicted_sales"] * 0.90 / 1e6
    upper = forecast_df["predicted_sales"] * 1.10 / 1e6
    ax.fill_between(forecast_df["forecast_date"], lower, upper,
                    alpha=0.2, color="#E53935", label="±10% band")

    # Vertical divider
    ax.axvline(state_df["Date"].max(), color="gray", linestyle=":", linewidth=1.5, label="Forecast start")

    ax.set_title(f"{state} — 8-Week Sales Forecast  |  Model: {model_name}  |  Val MAPE: {mape:.1f}%",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (Million USD)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    plt.tight_layout()

    safe_state = state.replace(" ", "_")
    path = f"{OUTPUT_DIR}/forecast_{safe_state}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 2. Model comparison bar chart (MAPE by model)
# ─────────────────────────────────────────────

def plot_model_comparison(model_dir: str = "models"):
    csv_path = f"{model_dir}/model_comparison.csv"
    if not os.path.exists(csv_path):
        print("  model_comparison.csv not found — run training first.")
        return

    df = pd.read_csv(csv_path)
    avg = df.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().reset_index()
    avg = avg.sort_values("MAPE")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
        axes[i].bar(avg["model"], avg[metric], color=palette[:len(avg)])
        axes[i].set_title(f"Avg {metric} by Model", fontweight="bold")
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("")
        for j, (_, row) in enumerate(avg.iterrows()):
            suffix = "%" if metric == "MAPE" else ""
            axes[i].text(j, row[metric] * 1.01, f"{row[metric]:.1f}{suffix}",
                         ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Model Comparison — Average Metrics Across All States",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 3. Best model distribution pie chart
# ─────────────────────────────────────────────

def plot_best_model_distribution(model_dir: str = "models"):
    csv_path = f"{model_dir}/model_comparison.csv"
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    best = df[df["best"] == True]
    counts = best["model"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    explode = [0.05] * len(counts)
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.0f%%",
        startangle=140, explode=explode,
        colors=sns.color_palette("tab10", len(counts)),
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("Best Model Distribution Across States", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/best_model_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 4. MAPE heatmap — all models x top states
# ─────────────────────────────────────────────

def plot_mape_heatmap(model_dir: str = "models", top_n: int = 20):
    csv_path = f"{model_dir}/model_comparison.csv"
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    top_states = (
        df[df["best"] == True]
        .sort_values("MAPE")
        .head(top_n)["state"]
        .tolist()
    )
    pivot = df[df["state"].isin(top_states)].pivot_table(
        index="state", columns="model", values="MAPE"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "MAPE (%)"})
    ax.set_title(f"MAPE (%) by Model & State (Top {top_n} States)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("State")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/mape_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────

def run_evaluation(
    filepath: str = "data/sales_data.xlsx",
    model_dir: str = "models",
    sample_states: list = None,
):
    from utils.data_preprocessing import prepare_data

    print("Loading data...")
    df = prepare_data(filepath)

    print("\nGenerating model comparison charts:")
    plot_model_comparison(model_dir)
    plot_best_model_distribution(model_dir)
    plot_mape_heatmap(model_dir)

    # Forecast plots for a sample of states
    if sample_states is None:
        sample_states = ["California", "Texas", "Florida", "New York", "Illinois"]

    print("\nGenerating forecast plots:")
    for state in sample_states:
        plot_forecast(state, df, model_dir)

    print(f"\n✅ All evaluation charts saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    run_evaluation()
