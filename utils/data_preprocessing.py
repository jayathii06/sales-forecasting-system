"""
Data Preprocessing Module - Final Working Version
"""
import pandas as pd
import numpy as np
import holidays


def prepare_data(filepath: str) -> pd.DataFrame:
    print("[1/4] Loading and cleaning data...")
    df = pd.read_excel(filepath)
    df.columns = [c.strip() for c in df.columns]
    df = df[["State", "Date", "Total"]].copy()
    df = df.rename(columns={"Total": "Sales"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "State", "Sales"])
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    print(f"      Rows: {len(df)}, States: {df['State'].nunique()}")

    print("[2/4] Aggregating to weekly frequency...")
    df["Date"] = df["Date"].dt.to_period("W").dt.start_time
    df = df.groupby(["State", "Date"], as_index=False)["Sales"].sum()
    print(f"      Weekly rows: {len(df)}")
    print(f"      Date sample: {df['Date'].head(3).tolist()}")
    print(f"      Sales sample: {df['Sales'].head(3).tolist()}")

    print("[3/4] Filling missing dates...")
    frames = []
    for state, grp in df.groupby("State"):
        grp = grp.sort_values("Date").set_index("Date")
        # Use the data's own frequency
        freq = pd.infer_freq(grp.index)
        if freq is None:
            freq = "7D"
        full_idx = pd.date_range(start=grp.index.min(), end=grp.index.max(), freq=freq)
        grp = grp.reindex(full_idx)
        grp["Sales"] = grp["Sales"].ffill().bfill()
        grp["State"] = state
        grp.index.name = "Date"
        frames.append(grp.reset_index())
    df = pd.concat(frames, ignore_index=True)
    print(f"      Rows: {len(df)}, Sales null: {df['Sales'].isnull().sum()}")
    print(f"      Sales sample: {df['Sales'].head(3).tolist()}")

    print("[4/4] Engineering features...")
    us_holidays = holidays.US(years=range(2018, 2027))
    result_frames = []
    for state, grp in df.groupby("State"):
        grp = grp.sort_values("Date").copy()
        grp["Sales"] = grp["Sales"].ffill().bfill().fillna(0)
        grp["lag_1"]        = grp["Sales"].shift(1)
        grp["lag_4"]        = grp["Sales"].shift(4)
        grp["lag_13"]       = grp["Sales"].shift(13)
        grp["roll_mean_4"]  = grp["Sales"].shift(1).rolling(4).mean()
        grp["roll_std_4"]   = grp["Sales"].shift(1).rolling(4).std()
        grp["roll_mean_8"]  = grp["Sales"].shift(1).rolling(8).mean()
        grp["roll_std_8"]   = grp["Sales"].shift(1).rolling(8).std()
        grp["week_of_year"] = grp["Date"].dt.isocalendar().week.astype(int)
        grp["month"]        = grp["Date"].dt.month
        grp["quarter"]      = grp["Date"].dt.quarter
        grp["day_of_week"]  = grp["Date"].dt.dayofweek
        grp["holiday_flag"] = grp["Date"].apply(
            lambda d: int(any((d + pd.Timedelta(days=i)) in us_holidays for i in range(7)))
        )
        result_frames.append(grp)

    df = pd.concat(result_frames, ignore_index=True)
    print(f"      Done. Shape: {df.shape}, States: {df['State'].nunique()}")
    print(f"      Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"      Sales null: {df['Sales'].isnull().sum()}")
    print(f"      Sales range: {df['Sales'].min():.0f} to {df['Sales'].max():.0f}")
    return df
