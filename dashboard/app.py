from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Car Sales Dashboard",
    page_icon="🚗",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car_sales_cleaned.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()


# ==========================================
# DASHBOARD TITLE
# ==========================================

st.title("🚗 Car Sales Analytics Dashboard")

st.markdown(
    "Interactive analysis of car sales, manufacturers, pricing, "
    "performance, and fuel efficiency."
)


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("🔍 Filters")

# Manufacturer Filter
manufacturers = sorted(df["manufacturer"].unique())

selected_manufacturers = st.sidebar.multiselect(
    "Select Manufacturer",
    options=manufacturers,
    default=manufacturers
)


# Vehicle Type Filter
vehicle_types = sorted(df["vehicle_type"].unique())

selected_vehicle_types = st.sidebar.multiselect(
    "Select Vehicle Type",
    options=vehicle_types,
    default=vehicle_types
)


# Price Filter
min_price = float(df["price_in_thousands"].min())
max_price = float(df["price_in_thousands"].max())

selected_price_range = st.sidebar.slider(
    "Price Range (Thousands)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)


# ==========================================
# APPLY FILTERS
# ==========================================

filtered_df = df[
    (df["manufacturer"].isin(selected_manufacturers))
    & (df["vehicle_type"].isin(selected_vehicle_types))
    & (
        df["price_in_thousands"].between(
            selected_price_range[0],
            selected_price_range[1]
        )
    )
]


# ==========================================
# KPI METRICS
# ==========================================

st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df["sales_in_thousands"].sum()
average_price = filtered_df["price_in_thousands"].mean()
average_fuel_efficiency = filtered_df["fuel_efficiency"].mean()
total_models = filtered_df.shape[0]

col1.metric(
    "Total Sales",
    f"{total_sales:,.2f}K"
)

col2.metric(
    "Average Price",
    f"${average_price:,.2f}K"
)

col3.metric(
    "Avg Fuel Efficiency",
    f"{average_fuel_efficiency:,.2f}"
)

col4.metric(
    "Total Models",
    f"{total_models}"
)


# ==========================================
# TOP SELLING CARS
# ==========================================

st.subheader("🏆 Top Selling Cars")

top_cars = (
    filtered_df
    .sort_values("sales_in_thousands", ascending=False)
    .head(10)
)

fig_top_cars = px.bar(
    top_cars,
    x="sales_in_thousands",
    y="model",
    color="manufacturer",
    orientation="h",
    title="Top 10 Best-Selling Cars",
    labels={
        "sales_in_thousands": "Sales (Thousands)",
        "model": "Car Model"
    }
)

fig_top_cars.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig_top_cars,
    width="stretch"
)


# ==========================================
# SALES BY MANUFACTURER
# ==========================================

st.subheader("🏭 Sales by Manufacturer")

manufacturer_sales = (
    filtered_df
    .groupby("manufacturer", as_index=False)
    ["sales_in_thousands"]
    .sum()
    .sort_values("sales_in_thousands", ascending=False)
)

fig_manufacturer = px.bar(
    manufacturer_sales,
    x="manufacturer",
    y="sales_in_thousands",
    title="Total Sales by Manufacturer",
    labels={
        "sales_in_thousands": "Total Sales (Thousands)",
        "manufacturer": "Manufacturer"
    }
)

st.plotly_chart(
    fig_manufacturer,
    width="stretch"
)


# ==========================================
# PRICE VS SALES
# ==========================================

st.subheader("💰 Price vs Sales")

fig_price_sales = px.scatter(
    filtered_df,
    x="price_in_thousands",
    y="sales_in_thousands",
    color="manufacturer",
    size="horsepower",
    hover_data=[
        "model",
        "fuel_efficiency",
        "vehicle_type"
    ],
    title="Price vs Sales Performance",
    labels={
        "price_in_thousands": "Price (Thousands)",
        "sales_in_thousands": "Sales (Thousands)"
    }
)

st.plotly_chart(
    fig_price_sales,
    width="stretch"
)


# ==========================================
# FUEL EFFICIENCY ANALYSIS
# ==========================================

st.subheader("⛽ Fuel Efficiency by Manufacturer")

fuel_efficiency = (
    filtered_df
    .groupby("manufacturer", as_index=False)
    ["fuel_efficiency"]
    .mean()
    .sort_values("fuel_efficiency", ascending=False)
)

fig_fuel = px.bar(
    fuel_efficiency,
    x="manufacturer",
    y="fuel_efficiency",
    title="Average Fuel Efficiency by Manufacturer",
    labels={
        "fuel_efficiency": "Average Fuel Efficiency",
        "manufacturer": "Manufacturer"
    }
)

st.plotly_chart(
    fig_fuel,
    width="stretch"
)


# ==========================================
# DATA TABLE
# ==========================================

st.subheader("📄 Filtered Dataset")

st.dataframe(
    filtered_df,
    width="stretch"
)