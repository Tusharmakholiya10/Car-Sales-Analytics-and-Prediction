import subprocess
import sys
from datetime import UTC, datetime, timedelta

from airflow.sdk import DAG, task

PROJECT_ROOT = "/opt/car_sales"

def task_failure_callback(context) -> None:
    """Log useful information when an Airflow task fails."""
    task_instance = context["task_instance"]
    exception = context.get("exception")

    print("=" * 60)
    print("AIRFLOW TASK FAILURE")
    print(f"DAG: {task_instance.dag_id}")
    print(f"Task: {task_instance.task_id}")
    print(f"Run ID: {task_instance.run_id}")
    print(f"Try Number: {task_instance.try_number}")
    print(f"Exception: {exception}")
    print("=" * 60)

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
    max_active_runs=1,
    description=(
        "End-to-end car sales analytics and machine learning pipeline. "
        "Validates and cleans data, loads it into MySQL, runs SQL analysis "
        "and visualizations, compares sales prediction models, trains the "
        "best model, and generates model explainability results."
    ),
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    tags=["car-sales", "production"],
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=10),
        "on_failure_callback": task_failure_callback,
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
    def validate_mysql_data():
        import os

        import mysql.connector
        import pandas as pd
        from dotenv import load_dotenv

        load_dotenv(f"{PROJECT_ROOT}/.env")

        expected_rows = len(
            pd.read_csv(f"{PROJECT_ROOT}/data/car_sales_cleaned.csv")
        )

        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
        )

        try:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM car_sales")
                actual_rows = cursor.fetchone()[0]
            finally:
                cursor.close()
        finally:
            conn.close()

        print(f"Expected rows: {expected_rows}")
        print(f"MySQL rows: {actual_rows}")

        if actual_rows != expected_rows:
            raise ValueError(
                f"MySQL row count mismatch: expected {expected_rows}, "
                f"found {actual_rows}"
            )
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
    validate_mysql = validate_mysql_data()
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
        >> validate_mysql
        >> sql
        >> advanced_sql
        >> visual
        >> advanced_visual
        >> comparison
        >> train
        >> explain
    )