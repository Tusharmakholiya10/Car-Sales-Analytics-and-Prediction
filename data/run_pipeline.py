import subprocess
import sys

steps = [
    "scripts/00_download_dataset.py",
    "scripts/01_clean_data.py",
    "scripts/02_load_mysql.py",
    "scripts/03_sql_analysis.py",
    "scripts/04_visualization.py"
]

for step in steps:
    print(f"\n{'='*50}")
    print(f"Running: {step}")
    print(f"{'='*50}\n")

    result = subprocess.run([sys.executable, step], check=False)

    if result.returncode != 0:
        print(f"\nError while executing {step}")
        break

print("\nPipeline execution completed.")
''