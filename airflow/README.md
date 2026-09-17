@'
# Airflow Data Pipeline

This directory contains the Apache Airflow orchestration layer for the Car Sales Analytics and Prediction project.

## Architecture

The Airflow pipeline orchestrates the existing project scripts instead of duplicating their business logic.

```text
Airflow Scheduler
       |
       v
CeleryExecutor
       |
       v
Airflow Worker
       |
       v
Validate Input Data
       |
       v
Clean Data
       |
       v
Data Quality Report
       |
       v
Load Data into MySQL
       |
       v
SQL Analysis
       |
       v
Advanced SQL Analysis
       |
       v
Visualizations
       |
       v
Advanced Visualizations
       |
       v
Model Comparison
       |
       v
Train Best Model
       |
       v
Model Explainability