from datetime import datetime

from airflow.sdk import DAG, task


with DAG(
    dag_id="car_sales_test_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["car-sales", "test"],
) as dag:

    @task
    def test_pipeline():
        print("Car Sales Airflow pipeline test successful!")

    test_pipeline()