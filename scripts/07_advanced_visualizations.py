from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

plt.rcParams["figure.autolayout"] = True


# ==========================================
# 1. TOP 10 BEST-SELLING CARS
# ==========================================

df = pd.read_csv(OUTPUT_DIR / "top_10_best_selling_cars.csv")

df = df.sort_values("sales_in_thousands")

plt.figure(figsize=(12, 7))

labels = df["manufacturer"] + " " + df["model"]

plt.barh(labels, df["sales_in_thousands"])

plt.title("Top 10 Best-Selling Cars")
plt.xlabel("Sales (Thousands)")
plt.ylabel("Car Model")

plt.savefig(
    OUTPUT_DIR / "top_10_best_selling_cars.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 2. TOP 10 MANUFACTURERS BY TOTAL SALES
# ==========================================

df = pd.read_csv(OUTPUT_DIR / "manufacturer_sales_performance.csv")

df = df.head(10).sort_values("total_sales_thousands")

plt.figure(figsize=(12, 7))

plt.barh(df["manufacturer"], df["total_sales_thousands"])

plt.title("Top 10 Manufacturers by Total Sales")
plt.xlabel("Total Sales (Thousands)")
plt.ylabel("Manufacturer")

plt.savefig(
    OUTPUT_DIR / "top_10_manufacturers_total_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 3. AVERAGE PRICE VS SALES BY VEHICLE TYPE
# ==========================================

df = pd.read_csv(OUTPUT_DIR / "average_price_by_vehicle_type.csv")

x = range(len(df))

plt.figure(figsize=(10, 6))

plt.bar(
    [i - 0.2 for i in x],
    df["average_price_thousands"],
    width=0.4,
    label="Average Price (Thousands)"
)

plt.bar(
    [i + 0.2 for i in x],
    df["average_sales_thousands"],
    width=0.4,
    label="Average Sales (Thousands)"
)

plt.xticks(list(x), df["vehicle_type"])

plt.title("Average Price vs Average Sales by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Value (Thousands)")
plt.legend()

plt.savefig(
    OUTPUT_DIR / "price_vs_sales_by_vehicle_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 4. TOP FUEL-EFFICIENT MANUFACTURERS
# ==========================================

df = pd.read_csv(
    OUTPUT_DIR / "most_fuel_efficient_manufacturers.csv"
)

df = df.sort_values("average_fuel_efficiency")

plt.figure(figsize=(12, 7))

plt.barh(
    df["manufacturer"],
    df["average_fuel_efficiency"]
)

plt.title("Top Fuel-Efficient Manufacturers")
plt.xlabel("Average Fuel Efficiency")
plt.ylabel("Manufacturer")

plt.savefig(
    OUTPUT_DIR / "fuel_efficient_manufacturers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 5. HORSEPOWER VS SALES
# ==========================================

df = pd.read_csv(OUTPUT_DIR / "power_vs_sales.csv")

plt.figure(figsize=(10, 7))

plt.scatter(
    df["horsepower"],
    df["sales_in_thousands"],
    alpha=0.7
)

plt.title("Horsepower vs Car Sales")
plt.xlabel("Horsepower")
plt.ylabel("Sales (Thousands)")

plt.savefig(
    OUTPUT_DIR / "horsepower_vs_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 6. BEST VALUE-FOR-MONEY CARS
# ==========================================

df = pd.read_csv(OUTPUT_DIR / "best_value_for_money_cars.csv")

df = df.sort_values("horsepower_per_price")

labels = df["manufacturer"] + " " + df["model"]

plt.figure(figsize=(12, 7))

plt.barh(
    labels,
    df["horsepower_per_price"]
)

plt.title("Top 10 Best Value-for-Money Cars")
plt.xlabel("Horsepower per Price Unit")
plt.ylabel("Car Model")

plt.savefig(
    OUTPUT_DIR / "best_value_for_money_cars.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("=" * 50)
print("ADVANCED VISUALIZATIONS COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("\nGenerated charts:")

charts = [
    "top_10_best_selling_cars.png",
    "top_10_manufacturers_total_sales.png",
    "price_vs_sales_by_vehicle_type.png",
    "fuel_efficient_manufacturers.png",
    "horsepower_vs_sales.png",
    "best_value_for_money_cars.png"
]

for chart in charts:
    print(f"✓ {chart}")