"""Run: python debug_dates.py"""
import pandas as pd

df = pd.read_excel("data/sales_data.xlsx")
df = df.rename(columns={"Total": "Sales"})
df = df[["State", "Date", "Sales"]].copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df = df.dropna(subset=["Date", "State", "Sales"])

# Weekly aggregation
df["Date"] = df["Date"].dt.to_period("W-MON").dt.start_time
df = df.groupby(["State", "Date"], as_index=False)["Sales"].sum()

ca = df[df["State"] == "California"]
print("=== AFTER WEEKLY AGG ===")
print(f"Rows: {len(ca)}")
print(f"Date type: {ca['Date'].dtype}")
print(f"Date sample: {ca['Date'].head(3).tolist()}")
print(f"Sales sample: {ca['Sales'].head(3).tolist()}")

# Build full calendar
all_weeks = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="W-MON")
print(f"\n=== CALENDAR DATES ===")
print(f"Calendar date type: {all_weeks.dtype}")
print(f"Calendar sample: {all_weeks[:3].tolist()}")

# Check if types match
print(f"\nDate dtype: {df['Date'].dtype}")
print(f"Calendar dtype: {all_weeks.dtype}")

# Try a simple filter
ca_state = df[df["State"] == "California"].copy()
ca_state = ca_state.set_index("Date").sort_index()
print(f"\nIndex type after set_index: {ca_state.index.dtype}")
