# model_train.py (overwrite)
import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATHS = [
    r"C:\Users\akash\OneDrive\Desktop\farmers_dataset.csv",
    "/mnt/data/b799548b-bdbf-40ab-bbff-2f6c7a23c1f0.csv"
]
OUT_DIR = os.path.dirname(__file__)
OUT_MODEL = os.path.join(OUT_DIR, "subsidy_model_v1.pkl")
META_JSON = os.path.join(OUT_DIR, "model_metadata.json")

DATA_PATH = next((p for p in DATA_PATHS if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("Dataset not found in expected paths.")

df = pd.read_csv(DATA_PATH)
df['acres'] = pd.to_numeric(df.get('acres', 0), errors='coerce').fillna(0)

if 'income' not in df.columns:
    rng = np.random.default_rng(42)
    df['income'] = (df['acres'] * rng.normal(3000, 500, size=len(df))).clip(5000,200000).astype(int)
if 'crop' not in df.columns:
    df['crop'] = np.random.choice(['rice','wheat','maize','cotton','banana','mango'], size=len(df))

def compute_label(r):
    base = r['acres'] * 8000.0
    income = r['income']
    if income < 50000: base *= 1.25
    elif income < 100000: base *= 1.10
    else: base *= 0.95
    soil = str(r.get('soil_type','')).lower()
    if 'sandy' in soil: base *= 1.10
    elif 'clay' in soil: base *= 0.95
    irr = str(r.get('irrigation_type','')).lower()
    if 'rain' in irr: base *= 1.20
    crop = str(r.get('crop','')).lower()
    if 'rice' in crop: base *= 1.18
    base = max(0, base + np.random.normal(0, base*0.05))
    return round(base, 0)

if 'subsidy_amount' not in df.columns:
    df['subsidy_amount'] = df.apply(compute_label, axis=1)

FEATURES = ['acres','soil_type','irrigation_type','location','crop','income']
X = df[FEATURES].copy()
y = df['subsidy_amount'].astype(float)

cat_cols = ['soil_type','irrigation_type','location','crop']
num_cols = ['acres','income']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
    ('num', StandardScaler(), num_cols)
], remainder='drop')

model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
pipeline = Pipeline([('pre', preprocessor), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
print("CV MAE (neg):", cv_scores, "mean:", cv_scores.mean())

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Test MAE: {mae:.2f}, R2: {r2:.3f}")

joblib.dump(pipeline, OUT_MODEL)
meta = {
    "model_file": os.path.basename(OUT_MODEL),
    "features": FEATURES,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "mae_test": float(mae),
    "r2_test": float(r2),
    "cv_neg_mae_mean": float(cv_scores.mean())
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print("Model saved:", OUT_MODEL)
print("Metadata saved:", META_JSON)
