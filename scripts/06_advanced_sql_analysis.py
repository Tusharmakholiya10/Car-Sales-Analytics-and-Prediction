import os
from pathlib import Path

import pandas as pd
import mysql.connector
from dotenv import load_dotenv


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")


# ==========================================
# DATABASE CONNECTION
# ==========================================

conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)

print("Connected to MySQL successfully!\n")


# ==========================================
# ADVANCED BUSINESS ANALYSIS QUERIES
# ==========================================

queries = {

    "top_10_best_selling_cars": """
        SELECT
            manufacturer,
            model,
            sales_in_thousands,
            price_in_thousands
        FROM car_sales
        ORDER BY sales_in_thousands DESC
        LIMIT 10;
    """,

    "manufacturer_sales_performance": """
        SELECT
            manufacturer,
            ROUND(SUM(sales_in_thousands), 2) AS total_sales_thousands,
            ROUND(AVG(sales_in_thousands), 2) AS average_sales_thousands,
            COUNT(*) AS number_of_models
        FROM car_sales
        GROUP BY manufacturer
        ORDER BY total_sales_thousands DESC;
    """,

    "average_price_by_vehicle_type": """
        SELECT
            vehicle_type,
            ROUND(AVG(price_in_thousands), 2) AS average_price_thousands,
            ROUND(AVG(sales_in_thousands), 2) AS average_sales_thousands
        FROM car_sales
        GROUP BY vehicle_type
        ORDER BY average_price_thousands DESC;
    """,

    "most_fuel_efficient_manufacturers": """
        SELECT
            manufacturer,
            ROUND(AVG(fuel_efficiency), 2) AS average_fuel_efficiency,
            COUNT(*) AS number_of_models
        FROM car_sales
        GROUP BY manufacturer
        ORDER BY average_fuel_efficiency DESC
        LIMIT 10;
    """,

    "best_value_for_money_cars": """
        SELECT
            manufacturer,
            model,
            ROUND(
                horsepower / NULLIF(price_in_thousands, 0),
                2
            ) AS horsepower_per_price,
            horsepower,
            price_in_thousands
        FROM car_sales
        WHERE price_in_thousands > 0
        ORDER BY horsepower_per_price DESC
        LIMIT 10;
    """,

    "high_sales_low_price_cars": """
        SELECT
            manufacturer,
            model,
            sales_in_thousands,
            price_in_thousands,
            ROUND(
                sales_in_thousands / NULLIF(price_in_thousands, 0),
                2
            ) AS sales_to_price_ratio
        FROM car_sales
        WHERE price_in_thousands > 0
        ORDER BY sales_to_price_ratio DESC
        LIMIT 10;
    """,

    "power_vs_sales": """
        SELECT
            manufacturer,
            model,
            horsepower,
            sales_in_thousands,
            power_perf_factor
        FROM car_sales
        ORDER BY horsepower DESC;
    """

}


# ==========================================
# EXECUTE QUERIES AND SAVE RESULTS
# ==========================================

for analysis_name, query in queries.items():

    print("=" * 60)
    print(analysis_name.replace("_", " ").upper())
    print("=" * 60)

    df = pd.read_sql(query, conn)

    print(df.to_string(index=False))

    output_file = OUTPUT_DIR / f"{analysis_name}.csv"

    df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}\n")


# ==========================================
# CLOSE CONNECTION
# ==========================================

conn.close()

print("All advanced SQL analyses completed successfully!")