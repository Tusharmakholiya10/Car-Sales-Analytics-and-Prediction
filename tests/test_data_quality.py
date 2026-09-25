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