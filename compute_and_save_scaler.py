# compute_and_save_scaler.py
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import joblib

CSV_PATH = "brain_mri_testdata.csv"
SCALER_OUT = "scaler.pkl"
FEATURE_ORDER_OUT = "feature_order.json"

# load
df = pd.read_csv(CSV_PATH)
# assume label column is named 'tumor' and is last column
if 'tumor' not in df.columns:
    raise SystemExit("CSV must contain 'tumor' column")

feature_cols = [c for c in df.columns if c != 'tumor']
X = df[feature_cols].values

scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, SCALER_OUT)

# save feature order to json
with open(FEATURE_ORDER_OUT, 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("Saved:", SCALER_OUT, FEATURE_ORDER_OUT)