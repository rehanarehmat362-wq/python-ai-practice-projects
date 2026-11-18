"""
House Price Prediction — Linear Regression
IMPORTANT: This package uses a synthetic demonstration dataset, not verified market data.
Run: python train_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "house_prices_synthetic.csv"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "model"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = ["area_sqft", "bedrooms", "bathrooms", "house_age_years",
            "location_score", "parking_spaces"]
X, y = df[features], df["price_million_pkr"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
results = {
    "MAE_million_PKR": mean_absolute_error(y_test, pred),
    "MSE_(million_PKR)^2": mse,
    "RMSE_million_PKR": mse ** 0.5,
    "R2": r2_score(y_test, pred),
}
pd.DataFrame(list(results.items()), columns=["metric", "value"]).to_csv(
    OUT / "model_evaluation_reproduced.csv", index=False
)
joblib.dump(model, MODEL_DIR / "linear_regression_model.pkl")
print("Training complete.")
print("Evaluation:", results)
print("Reminder: metrics describe performance on this synthetic dataset only.")
