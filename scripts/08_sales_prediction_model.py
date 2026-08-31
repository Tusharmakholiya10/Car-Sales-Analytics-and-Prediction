from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "car_sales_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")


# ==========================================
# SELECT FEATURES AND TARGET
# ==========================================

target = "sales_in_thousands"

features = [
    "manufacturer",
    "vehicle_type",
    "price_in_thousands",
    "engine_size",
    "horsepower",
    "wheelbase",
    "width",
    "length",
    "curb_weight",
    "fuel_capacity",
    "fuel_efficiency",
    "power_perf_factor"
]

X = df[features]
y = df[target]


# ==========================================
# DEFINE FEATURE TYPES
# ==========================================

categorical_features = [
    "manufacturer",
    "vehicle_type"
]

numerical_features = [
    "price_in_thousands",
    "engine_size",
    "horsepower",
    "wheelbase",
    "width",
    "length",
    "curb_weight",
    "fuel_capacity",
    "fuel_efficiency",
    "power_perf_factor"
]


# ==========================================
# PREPROCESSING
# ==========================================

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# MODEL PIPELINE
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ==========================================
# TRAIN MODEL
# ==========================================

pipeline.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("=" * 40)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# ==========================================
# SAVE MODEL
# ==========================================

model_path = MODEL_DIR / "sales_prediction_model.joblib"

joblib.dump(pipeline, model_path)

print(f"\nModel saved successfully: {model_path}")


# ==========================================
# ACTUAL VS PREDICTED VISUALIZATION
# ==========================================

results_df = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

results_df.to_csv(
    OUTPUT_DIR / "sales_predictions.csv",
    index=False
)

plt.figure(figsize=(10, 7))

plt.scatter(
    results_df["Actual Sales"],
    results_df["Predicted Sales"],
    alpha=0.7
)

min_value = min(
    results_df["Actual Sales"].min(),
    results_df["Predicted Sales"].min()
)

max_value = max(
    results_df["Actual Sales"].max(),
    results_df["Predicted Sales"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Car Sales")
plt.xlabel("Actual Sales (Thousands)")
plt.ylabel("Predicted Sales (Thousands)")

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPrediction results saved to output/sales_predictions.csv"
)

print(
    "Visualization saved to output/actual_vs_predicted_sales.png"
)