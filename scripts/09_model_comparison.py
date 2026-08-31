from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

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
# FEATURE TYPES
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
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# MODELS TO COMPARE
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_split=3,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}


# ==========================================
# TRAIN AND EVALUATE MODELS
# ==========================================

results = []

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

    print("\n" + "=" * 50)
    print(model_name.upper())
    print("=" * 50)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")


# ==========================================
# SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
)

results_path = OUTPUT_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)


print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {results_path}")