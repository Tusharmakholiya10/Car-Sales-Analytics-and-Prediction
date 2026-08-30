import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car_sales_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "output"

# Create output folder if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

# Basic information
total_rows, total_columns = df.shape
duplicate_rows = df.duplicated().sum()
missing_values = df.isnull().sum()

# Create report
report_lines = []

report_lines.append("=" * 50)
report_lines.append("CAR SALES DATA QUALITY REPORT")
report_lines.append("=" * 50)

report_lines.append(f"\nTotal Rows: {total_rows}")
report_lines.append(f"Total Columns: {total_columns}")
report_lines.append(f"Duplicate Rows: {duplicate_rows}")

report_lines.append("\nMISSING VALUES BY COLUMN")
report_lines.append("-" * 50)

for column, missing in missing_values.items():
    report_lines.append(f"{column}: {missing}")

report_lines.append("\nDATA TYPES")
report_lines.append("-" * 50)

for column, dtype in df.dtypes.items():
    report_lines.append(f"{column}: {dtype}")

report_lines.append("\nDATA QUALITY SUMMARY")
report_lines.append("-" * 50)

if duplicate_rows == 0:
    report_lines.append("✓ No duplicate rows found.")
else:
    report_lines.append(f"⚠ {duplicate_rows} duplicate rows found.")

if missing_values.sum() == 0:
    report_lines.append("✓ No missing values found.")
else:
    report_lines.append(
        f"⚠ Total missing values: {missing_values.sum()}"
    )

report_lines.append("\nReport generated successfully.")

# Save report
report_path = OUTPUT_DIR / "data_quality_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

# Print report
print("\n".join(report_lines))

print(f"\nReport saved to: {report_path}")