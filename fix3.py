# Debug script - run this to see exactly what's happening
import pandas as pd
import sys
sys.path.insert(0, '.')
from utils.data_preprocessing import prepare_data

df = prepare_data("data/sales_data.xlsx")

# Check California specifically
ca = df[df["State"] == "California"].copy()
print("California rows:", len(ca))
print("Sales null count:", ca["Sales"].isnull().sum())
print("Sales sample:", ca["Sales"].head(10).tolist())
print("Sales > 0:", (ca["Sales"] > 0).sum())

# Simulate exactly what train_models does
ca["Sales"] = ca["Sales"].ffill().bfill().fillna(0)
print("\nAfter fill:")
print("Sales null count:", ca["Sales"].isnull().sum())
print("Sales > 0:", (ca["Sales"] > 0).sum())
print("Total rows:", len(ca))
