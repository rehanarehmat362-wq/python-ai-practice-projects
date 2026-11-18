# House Price Prediction — Professional Project Package

## Project status
**Portfolio / technical demonstration only. Not production-ready for real property valuation.**

The supplied data are synthetic and were generated programmatically for learning. No real property transaction or listing data were provided. The model outputs must not be presented as verified market prices.

## What is included
- `notebook/House_Price_Prediction_Professional.ipynb` — cleaned, structured, reproducible notebook with explicit limitations.
- `notebook/Original_Uploaded_Notebook.ipynb` — original notebook uploaded for review.
- `data/house_prices_synthetic.csv` — 180 generated records; six input features and one target.
- `figures/` — price distribution, actual-vs-predicted, residual, and correlation plots.
- `outputs/model_evaluation.csv` — held-out test metrics.
- `outputs/test_set_predictions.csv` — actual/predicted comparison and absolute error.
- `outputs/permutation_importance.csv` — permutation-based model interpretation.
- `outputs/example_prediction.csv` — illustrative example input and output.
- `model/linear_regression_model.pkl` — trained scikit-learn model.
- `train_model.py` — reproducible training script.

## Dataset fields
- `area_sqft`: area in square feet
- `bedrooms`: bedroom count
- `bathrooms`: bathroom count
- `house_age_years`: age in years
- `location_score`: synthetic ordinal score from 1 to 10 (not a defined real-world location index)
- `parking_spaces`: number of parking spaces
- `price_million_pkr`: generated target in million PKR

## Reproduced evaluation
| Metric | Value |
|---|---:|
| MAE | 2.2178 million PKR |
| MSE | 8.3129 (million PKR)^2 |
| RMSE | 2.8832 million PKR |
| R² | 0.9737 |

These figures are based only on a random 20% holdout from the generated dataset. They are not estimates of real-world market performance.

## Environment
Python 3.10+ recommended. Install dependencies:
```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib jupyter
```
Open the notebook with:
```bash
jupyter notebook notebook/House_Price_Prediction_Professional.ipynb
```
Run the training script from the project root:
```bash
python train_model.py
```

## Example model use
```python
import joblib
import pandas as pd

model = joblib.load("model/linear_regression_model.pkl")
house = pd.DataFrame([{
    "area_sqft": 2000,
    "bedrooms": 3,
    "bathrooms": 2,
    "house_age_years": 5,
    "location_score": 8,
    "parking_spaces": 1
}])
print(model.predict(house)[0], "million PKR")
```
The output is illustrative only.

## Required work before a real client engagement
1. Replace the synthetic dataset with authorized, verified property data.
2. Define scope and target geography/property segment.
3. Validate data quality and prevent leakage.
4. Establish a defensible validation design and benchmark.
5. Evaluate segment-wise error and out-of-time performance.
6. Review fairness, privacy, data licensing, and applicable legal requirements.
7. Have a real-estate domain expert review the methodology and outputs.
8. Agree on acceptance criteria and disclaimers with the client.
