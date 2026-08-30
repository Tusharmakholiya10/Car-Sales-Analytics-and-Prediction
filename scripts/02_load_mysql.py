import pandas as pd
import mysql.connector
import os
# Read cleaned dataset
df = pd.read_csv("data/car_sales_cleaned.csv")

# Connect to MySQL

from dotenv import load_dotenv  
load_dotenv()
conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)

cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS car_sales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    sales_in_thousands FLOAT,
    __year_resale_value FLOAT,
    vehicle_type VARCHAR(50),
    price_in_thousands FLOAT,
    engine_size FLOAT,
    horsepower FLOAT,
    wheelbase FLOAT,
    width FLOAT,
    length FLOAT,
    curb_weight FLOAT,
    fuel_capacity FLOAT,
    fuel_efficiency FLOAT,
    latest_launch VARCHAR(50),
    power_perf_factor FLOAT
)
""")

# Remove old data so records are not duplicated
cursor.execute("DELETE FROM car_sales")

# Insert query
insert_query = """
INSERT INTO car_sales (
    manufacturer,
    model,
    sales_in_thousands,
    __year_resale_value,
    vehicle_type,
    price_in_thousands,
    engine_size,
    horsepower,
    wheelbase,
    width,
    length,
    curb_weight,
    fuel_capacity,
    fuel_efficiency,
    latest_launch,
    power_perf_factor
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Insert all rows
for _, row in df.iterrows():
    cursor.execute(insert_query, tuple(row))

# Save changes
conn.commit()

print(f"{len(df)} rows inserted successfully!")

# Close connection
cursor.close()
conn.close()