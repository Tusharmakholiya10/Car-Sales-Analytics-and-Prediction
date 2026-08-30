import mysql.connector
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)

queries = {
    "Top 5 Manufacturers by Sales": """
        SELECT manufacturer,
               ROUND(SUM(sales_in_thousands), 2) AS total_sales
        FROM car_sales
        GROUP BY manufacturer
        ORDER BY total_sales DESC
        LIMIT 5;
    """,

    "Average Price by Manufacturer": """
        SELECT manufacturer,
               ROUND(AVG(price_in_thousands), 2) AS avg_price
        FROM car_sales
        GROUP BY manufacturer
        ORDER BY avg_price DESC
        LIMIT 5;
    """
}

for title, query in queries.items():
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    result = pd.read_sql(query, conn)
    print(result)

conn.close()