import subprocess
import sys
from datetime import datetime, timedelta

from airflow.sdk import DAG, task


PROJECT_ROOT = "/opt/car_sales"


def run_project_script(script_name: str) -> None:
    """Run one existing project script from the mounted project directory."""
    script_path = f"scripts/{script_name}"

    print("=" * 60)
    print(f"Starting pipeline step: {script_path}")
    print(f"Working directory: {PROJECT_ROOT}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        check=False,
    )

    print(f"Finished pipeline step: {script_path}")
    print(f"Exit code: {result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed while running {script_path} "
            f"with exit code {result.returncode}"
        )

    print(f"Pipeline step completed successfully: {script_path}")


with DAG(
    dag_id="car_sales_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["car-sales", "production"],
    description="End-to-end Car Sales analytics and machine learning pipeline",
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=10),
    },
    dagrun_timeout=timedelta(minutes=30),
) as dag:

    @task
    def validate_input_data():
        import os

        required_files = [
            "/opt/car_sales/data/Car_sales.csv",
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Required input file not found: {file_path}"
                )

        print("Input data validation successful.")

    @task
    def clean_data():
        run_project_script("01_clean_data.py")

    @task
    def data_quality():
        run_project_script("05_data_quality_report.py")

    @task
    def load_mysql():
        run_project_script("02_load_mysql.py")

    @task
    def sql_analysis():
        run_project_script("03_sql_analysis.py")

    @task
    def advanced_sql_analysis():
        run_project_script("06_advanced_sql_analysis.py")

    @task
    def visualization():
        run_project_script("04_visualization.py")

    @task
    def advanced_visualization():
        run_project_script("07_advanced_visualizations.py")

    @task
    def model_comparison():
        run_project_script("09_model_comparison.py")

    @task
    def train_best_model():
        run_project_script("10_train_best_model.py")

    @task
    def model_explainability():
        run_project_script("11_model_explainability.py")

    # Pipeline dependency graph
    validate = validate_input_data()
    clean = clean_data()
    quality = data_quality()
    mysql = load_mysql()
    sql = sql_analysis()
    advanced_sql = advanced_sql_analysis()
    visual = visualization()
    advanced_visual = advanced_visualization()
    comparison = model_comparison()
    train = train_best_model()
    explain = model_explainability()

    (
        validate
        >> clean
        >> quality
        >> mysql
        >> sql
        >> advanced_sql
        >> visual
        >> advanced_visual
        >> comparison
        >> train
        >> explain
    )