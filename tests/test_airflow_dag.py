from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = PROJECT_ROOT / "airflow" / "dags" / "car_sales_pipeline.py"


EXPECTED_TASKS = [
    "validate_input_data",
    "clean_data",
    "data_quality",
    "load_mysql",
    "sql_analysis",
    "advanced_sql_analysis",
    "visualization",
    "advanced_visualization",
    "model_comparison",
    "train_best_model",
    "model_explainability",
]


def test_airflow_dag_file_exists():
    assert DAG_PATH.exists(), "Airflow DAG file is missing"


def test_dag_id_is_correct():
    content = DAG_PATH.read_text(encoding="utf-8")

    assert 'dag_id="car_sales_pipeline"' in content


def test_dag_limits_active_runs():
    content = DAG_PATH.read_text(encoding="utf-8")

    assert "max_active_runs=1" in content


def test_dag_has_expected_tasks():
    content = DAG_PATH.read_text(encoding="utf-8")

    for task_id in EXPECTED_TASKS:
        assert f"def {task_id}(" in content, f"Missing Airflow task: {task_id}"


def test_dag_has_expected_task_count():
    content = DAG_PATH.read_text(encoding="utf-8")

    task_count = sum(
        f"def {task_id}(" in content
        for task_id in EXPECTED_TASKS
    )

    assert task_count == len(EXPECTED_TASKS)


def test_dag_has_linear_dependencies():
    content = DAG_PATH.read_text(encoding="utf-8")

    expected_dependencies = [
        ("validate", "clean"),
        ("clean", "quality"),
        ("quality", "mysql"),
        ("mysql", "sql"),
        ("sql", "advanced_sql"),
        ("advanced_sql", "visual"),
        ("visual", "advanced_visual"),
        ("advanced_visual", "comparison"),
        ("comparison", "train"),
        ("train", "explain"),
    ]

    for upstream, downstream in expected_dependencies:
        assert (
            f"{upstream}\n        >> {downstream}"
            in content
            or f"{upstream}\n >> {downstream}"
            in content
            or f"{upstream} >> {downstream}"
            in content
        ), f"Missing dependency: {upstream} >> {downstream}"