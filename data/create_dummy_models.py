import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

# --- Create a short dummy dataset ---
data = {
    "climate_model": ["GISS", "GFDL", "UKMO", "HadCM3", "GISS", "GFDL", "UKMO", "HadCM3"],
    "co2": [330, 400, 450, 500, 550, 370, 420, 480],
    "adaptation": ["none", "level1", "level2", "none", "level1", "level2", "none", "level1"],
    "region": ["Asia", "Europe", "Africa", "Asia", "Europe", "Africa", "Asia", "Europe"],
    "wheat_yield": [-12, 3, 8, -5, 5, 10, -8, 6],
    "rice_yield": [-10, 4, 9, -3, 6, 11, -6, 7],
    "coarse_yield": [-7, 2, 6, -4, 4, 9, -5, 5],
    "soybean_yield": [-5, 3, 7, -2, 5, 8, -3, 6],
}

df = pd.DataFrame(data)

# --- Encode categorical columns ---
for col in ["climate_model", "adaptation", "region"]:
    df[col] = df[col].astype("category").cat.codes

# --- Prepare feature matrix ---
X = df[["climate_model", "co2", "adaptation", "region"]]

# --- Create folder to save models ---
os.makedirs("models", exist_ok=True)

# --- Train & save one model per crop ---
for crop in ["wheat", "rice", "coarse", "soybean"]:
    y = df[f"{crop}_yield"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, f"models/{crop}_model.joblib")
    print(f"✅ Saved: models/{crop}_model.joblib")

print("\nAll dummy models created successfully!")
