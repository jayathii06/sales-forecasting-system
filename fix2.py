# Run this from inside your project folder: python fix2.py

with open('models/train_models.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix 1: replace the dropna Sales line with ffill
    if 'state_df = state_df.dropna(subset=["Sales"])' in line:
        indent = '        '
        new_lines.append(indent + 'state_df["Sales"] = state_df["Sales"].ffill().bfill().fillna(0)\n')
        # skip the original dropna line
        continue
    # Fix 2: fix the val fillna deprecation warning
    if 'val[FEATURE_COLS].fillna(method="bfill")' in line:
        line = line.replace('fillna(method="bfill").fillna(0)', 'bfill().ffill().fillna(0)')
    new_lines.append(line)

with open('models/train_models.py', 'w') as f:
    f.writelines(new_lines)

# Verify
with open('models/train_models.py', 'r') as f:
    content = f.read()

if 'ffill().bfill().fillna(0)' in content and 'state_df = state_df.dropna(subset=["Sales"])' not in content:
    print("SUCCESS! train_models.py is correctly fixed.")
else:
    print("WARNING: Check the file manually.")
    print("dropna still present:", 'state_df = state_df.dropna(subset=["Sales"])' in content)
    print("ffill present:", 'ffill' in content)
