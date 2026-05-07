with open('models/train_models.py', 'r') as f:
    c = f.read()

c = c.replace(
    'state_df = state_df.dropna(subset=["Sales"])',
    'state_df["Sales"] = state_df["Sales"].ffill().bfill().fillna(0)\n        state_df = state_df[state_df["Sales"] > 0]'
)
c = c.replace(
    'val[FEATURE_COLS].fillna(method="bfill").fillna(0)',
    'val[FEATURE_COLS].fillna(0)'
)

with open('models/train_models.py', 'w') as f:
    f.write(c)

print("Done! train_models.py is fixed.")
