"""
Exploratory Data Analysis (EDA) + Visualization
Run: python utils/eda.py
Generates plots in output/eda/ folder
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")
FIGSIZE = (14, 5)


# ─────────────────────────────────────────────
# 1. National weekly sales trend
# ─────────────────────────────────────────────

def plot_national_trend(df: pd.DataFrame):
    national = df.groupby("Date")["Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(national["Date"], national["Sales"] / 1e9, linewidth=2, color="#2196F3")
    ax.fill_between(national["Date"], national["Sales"] / 1e9, alpha=0.15, color="#2196F3")
    ax.set_title("National Weekly Sales Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (Billion USD)")
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_national_trend.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 2. Top 10 states by total sales
# ─────────────────────────────────────────────

def plot_top_states(df: pd.DataFrame):
    top = (
        df.groupby("State")["Sales"].sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top["State"][::-1], top["Sales"][::-1] / 1e9, color=sns.color_palette("tab10", 10))
    ax.set_xlabel("Total Sales (Billion USD)")
    ax.set_title("Top 10 States by Total Sales", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, top["Sales"][::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"${val/1e9:.1f}B", va="center", fontsize=9)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_top_states.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 3. Monthly seasonality
# ─────────────────────────────────────────────

def plot_seasonality(df: pd.DataFrame):
    df2 = df.copy()
    df2["Month"] = df2["Date"].dt.month
    monthly = df2.groupby("Month")["Sales"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["MonthName"] = monthly["Month"].apply(lambda m: month_names[m-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(monthly["MonthName"], monthly["Sales"] / 1e6,
           color=sns.color_palette("coolwarm", 12))
    ax.set_title("Average Weekly Sales by Month (Seasonality)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Sales (Million USD)")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_seasonality.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 4. Multi-state sales trends (top 6)
# ─────────────────────────────────────────────

def plot_multi_state(df: pd.DataFrame):
    top6 = (
        df.groupby("State")["Sales"].sum()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)
    axes = axes.flatten()
    palette = sns.color_palette("tab10", 6)

    for i, state in enumerate(top6):
        s = df[df["State"] == state].sort_values("Date")
        axes[i].plot(s["Date"], s["Sales"] / 1e6, color=palette[i], linewidth=1.5)
        axes[i].set_title(state, fontweight="bold")
        axes[i].set_ylabel("Sales (M USD)")
        axes[i].tick_params(axis="x", rotation=30, labelsize=7)

    fig.suptitle("Sales Trends — Top 6 States", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_multi_state_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 5. Missing data heatmap
# ─────────────────────────────────────────────

def plot_missing_data(df: pd.DataFrame):
    pivot = df.pivot_table(index="Date", columns="State", values="Sales", aggfunc="sum")
    missing = pivot.isnull().astype(int)

    if missing.sum().sum() == 0:
        print("  No missing data detected — skipping missing-data heatmap.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(missing.T, cmap="Reds", ax=ax, cbar=False,
                xticklabels=False, yticklabels=True)
    ax.set_title("Missing Data Heatmap (red = missing)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time →")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_missing_data.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 6. YoY growth heatmap
# ─────────────────────────────────────────────

def plot_yoy_heatmap(df: pd.DataFrame):
    df2 = df.copy()
    df2["Year"] = df2["Date"].dt.year
    yearly = df2.groupby(["State", "Year"])["Sales"].sum().unstack("Year")
    years = sorted(yearly.columns)
    if len(years) < 2:
        return
    yoy = pd.DataFrame(index=yearly.index)
    for i in range(1, len(years)):
        col = f"{years[i-1]}→{years[i]}"
        yoy[col] = ((yearly[years[i]] - yearly[years[i-1]]) / yearly[years[i-1]]) * 100

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(yoy, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                ax=ax, linewidths=0.5, cbar_kws={"label": "YoY Growth %"})
    ax.set_title("Year-over-Year Sales Growth by State (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_yoy_growth.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 7. Distribution of sales (log scale)
# ─────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    log_sales = np.log1p(df["Sales"])
    ax.hist(log_sales, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.set_xlabel("log(Sales + 1)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Weekly Sales (log scale)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/07_sales_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Run all plots
# ─────────────────────────────────────────────

def run_eda(filepath: str = "data/sales_data.xlsx"):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_preprocessing import prepare_data

    print("Running EDA...")
    df = prepare_data(filepath)

    print("\nGenerating plots:")
    plot_national_trend(df)
    plot_top_states(df)
    plot_seasonality(df)
    plot_multi_state(df)
    plot_missing_data(df)
    plot_yoy_heatmap(df)
    plot_distribution(df)

    print(f"\n✅ All EDA plots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    run_eda()
