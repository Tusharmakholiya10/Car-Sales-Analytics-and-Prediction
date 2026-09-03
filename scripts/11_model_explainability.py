from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_sales_prediction_model.joblib"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ---------------------------------------------------------
# EXTRACT MODEL COMPONENTS
# ---------------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
linear_model = model.named_steps["model"]


# ---------------------------------------------------------
# GET FEATURE NAMES
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

coefficients = linear_model.coef_


# ---------------------------------------------------------
# CREATE FEATURE IMPORTANCE DATAFRAME
# ---------------------------------------------------------

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_importance": abs(coefficients),
    }
)

importance_df = importance_df.sort_values(
    by="absolute_importance",
    ascending=False,
)


# ---------------------------------------------------------
# CLEAN FEATURE NAMES
# ---------------------------------------------------------

importance_df["feature"] = (
    importance_df["feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

output_csv = OUTPUT_DIR / "model_feature_importance.csv"

importance_df.to_csv(output_csv, index=False)

print("\nFeature importance results:")
print(importance_df.head(15).to_string(index=False))

print(f"\nFeature importance saved to:")
print(output_csv)


# ---------------------------------------------------------
# TOP 15 FEATURES VISUALIZATION
# ---------------------------------------------------------

top_features = importance_df.head(15).sort_values(
    by="absolute_importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["coefficient"],
)

plt.axvline(0, linewidth=1)

plt.title("Top 15 Features Influencing Car Sales")
plt.xlabel("Linear Regression Coefficient")
plt.ylabel("Feature")

plt.tight_layout()

output_chart = OUTPUT_DIR / "model_feature_importance.png"

plt.savefig(output_chart, dpi=300, bbox_inches="tight")

plt.close()

print(f"\nFeature importance chart saved to:")
print(output_chart)

print("\nModel explainability analysis completed successfully!")