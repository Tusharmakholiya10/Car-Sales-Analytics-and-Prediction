from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
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

print("=" * 50)
print("CAR SALES PREDICTION MODEL")
print("=" * 50)

print(f"\nDataset shape: {df.shape}")


# ==========================================
# DEFINE TARGET AND FEATURES
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
# PREPROCESSING PIPELINE
# ==========================================

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
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

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# CREATE MODEL PIPELINE
# ==========================================

model = LinearRegression()

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# TRAIN MODEL
# ==========================================

pipeline.fit(X_train, y_train)

print("\n✓ Model training completed successfully!")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

predictions = pipeline.predict(X_test)


# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\nMODEL PERFORMANCE")
print("=" * 50)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# ==========================================
# SAVE MODEL
# ==========================================

model_path = MODEL_DIR / "best_sales_prediction_model.joblib"

joblib.dump(
    pipeline,
    model_path
)

print(f"\n✓ Best model saved to:\n{model_path}")


# ==========================================
# SAVE PREDICTION RESULTS
# ==========================================

results_df = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions,
    "Difference": y_test.values - predictions
})

results_path = OUTPUT_DIR / "best_model_predictions.csv"

results_df.to_csv(
    results_path,
    index=False
)

print(f"\n✓ Predictions saved to:\n{results_path}")


# ==========================================
# ACTUAL VS PREDICTED CHART
# ==========================================

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

plt.title("Best Model: Actual vs Predicted Sales")
plt.xlabel("Actual Sales (Thousands)")
plt.ylabel("Predicted Sales (Thousands)")

chart_path = OUTPUT_DIR / "best_model_actual_vs_predicted.png"

plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\n✓ Visualization saved to:\n{chart_path}")

print("\n" + "=" * 50)
print("BEST MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)