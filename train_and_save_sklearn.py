# train_and_save_sklearn.py
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ---------- CONFIG ----------
CSV_PATH = "brain_mri_testdata.csv"   # produced earlier
MODEL_OUT = "model.pkl"               # scikit-learn RandomForest
SCALER_OUT = "scaler.pkl"
FEATURE_ORDER_OUT = "feature_order.json"
N_TREES = 100
RANDOM_STATE = 42
# ----------------------------

# load dataset
df = pd.read_csv(CSV_PATH)
if 'tumor' not in df.columns:
    raise SystemExit("CSV must contain 'tumor' column")

# features and labels
feature_cols = [c for c in df.columns if c != 'tumor']
X = df[feature_cols].values
y = df['tumor'].values

# scaler (z-score) — matches your Orange preprocess
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# train Random Forest
clf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE)
clf.fit(Xs, y)

# optional: quick CV score check
cv_scores = cross_val_score(clf, Xs, y, cv=5, scoring='roc_auc')
print(f"Cross-val AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# save model + scaler + feature order
joblib.dump(clf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
with open(FEATURE_ORDER_OUT, 'w') as f:
    json.dump(feature_cols, f, indent=2)

print("Saved:", MODEL_OUT, SCALER_OUT, FEATURE_ORDER_OUT)
