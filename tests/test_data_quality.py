from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Car_sales.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "car_sales_cleaned.csv"


def test_raw_dataset_exists():
    assert RAW_DATA_PATH.exists(), "Raw dataset is missing"


def test_cleaned_dataset_exists():
    assert CLEANED_DATA_PATH.exists(), "Cleaned dataset is missing"


def test_cleaned_dataset_has_expected_columns():
    df = pd.read_csv(CLEANED_DATA_PATH)

    expected_columns = {
        "manufacturer",
        "model",
        "sales_in_thousands",
        "__year_resale_value",
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
        "latest_launch",
        "power_perf_factor",
    }

    assert expected_columns.issubset(df.columns)


def test_cleaned_dataset_has_no_duplicates():
    df = pd.read_csv(CLEANED_DATA_PATH)

    assert df.duplicated().sum() == 0


def test_cleaned_dataset_has_no_missing_values():
    df = pd.read_csv(CLEANED_DATA_PATH)

    assert df.isnull().sum().sum() == 0


def test_sales_and_price_are_non_negative():
    df = pd.read_csv(CLEANED_DATA_PATH)

    assert (df["sales_in_thousands"] >= 0).all()
    assert (df["price_in_thousands"] >= 0).all()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"


def test_best_model_exists():
    model_path = MODELS_DIR / "best_sales_prediction_model.joblib"

    assert model_path.exists(), "Best sales prediction model is missing"
    assert model_path.stat().st_size > 0, "Best sales prediction model is empty"


def test_model_comparison_output():
    path = OUTPUT_DIR / "model_comparison.csv"

    assert path.exists(), "Model comparison output is missing"

    df = pd.read_csv(path)

    assert list(df.columns) == ["Model", "MAE", "RMSE", "R2 Score"]
    assert len(df) == 3
    assert df["Model"].notna().all()
    assert df[["MAE", "RMSE", "R2 Score"]].notna().all().all()


def test_best_model_predictions_output():
    path = OUTPUT_DIR / "best_model_predictions.csv"

    assert path.exists(), "Best model predictions output is missing"

    df = pd.read_csv(path)

    assert list(df.columns) == [
        "Actual Sales",
        "Predicted Sales",
        "Difference",
    ]
    assert len(df) > 0
    assert df.notna().all().all()


def test_model_feature_importance_output():
    path = OUTPUT_DIR / "model_feature_importance.csv"

    assert path.exists(), "Model feature importance output is missing"

    df = pd.read_csv(path)

    assert list(df.columns) == [
        "feature",
        "coefficient",
        "absolute_importance",
    ]
    assert len(df) > 0
    assert df["feature"].notna().all()
    assert df["coefficient"].notna().all()
    assert df["absolute_importance"].notna().all()


def test_model_comparison_metrics_are_valid():
    path = OUTPUT_DIR / "model_comparison.csv"
    df = pd.read_csv(path)

    assert (df["MAE"] >= 0).all()
    assert (df["RMSE"] >= 0).all()
    assert df["R2 Score"].notna().all()