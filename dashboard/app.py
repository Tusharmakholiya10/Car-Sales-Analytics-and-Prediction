from pathlib import Path
import joblib
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
MODEL_PATH = BASE_DIR / "models" / "best_sales_prediction_model.joblib" 

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


df = load_data()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

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
# ==========================================
# SALES PREDICTION
# ==========================================

st.divider()

st.header("🤖 Car Sales Prediction")

st.markdown(
    "Enter vehicle characteristics below to predict estimated sales."
)


# Create two columns for inputs
col1, col2 = st.columns(2)


with col1:

    prediction_manufacturer = st.selectbox(
        "Manufacturer",
        options=sorted(df["manufacturer"].unique())
    )

    prediction_vehicle_type = st.selectbox(
        "Vehicle Type",
        options=sorted(df["vehicle_type"].unique())
    )

    prediction_price = st.number_input(
        "Price (Thousands)",
        min_value=float(df["price_in_thousands"].min()),
        max_value=float(df["price_in_thousands"].max()),
        value=float(df["price_in_thousands"].median())
    )

    prediction_engine_size = st.number_input(
        "Engine Size",
        min_value=float(df["engine_size"].min()),
        max_value=float(df["engine_size"].max()),
        value=float(df["engine_size"].median())
    )

    prediction_horsepower = st.number_input(
        "Horsepower",
        min_value=float(df["horsepower"].min()),
        max_value=float(df["horsepower"].max()),
        value=float(df["horsepower"].median())
    )


with col2:

    prediction_wheelbase = st.number_input(
        "Wheelbase",
        min_value=float(df["wheelbase"].min()),
        max_value=float(df["wheelbase"].max()),
        value=float(df["wheelbase"].median())
    )

    prediction_width = st.number_input(
        "Width",
        min_value=float(df["width"].min()),
        max_value=float(df["width"].max()),
        value=float(df["width"].median())
    )

    prediction_length = st.number_input(
        "Length",
        min_value=float(df["length"].min()),
        max_value=float(df["length"].max()),
        value=float(df["length"].median())
    )

    prediction_curb_weight = st.number_input(
        "Curb Weight",
        min_value=float(df["curb_weight"].min()),
        max_value=float(df["curb_weight"].max()),
        value=float(df["curb_weight"].median())
    )

    prediction_fuel_capacity = st.number_input(
        "Fuel Capacity",
        min_value=float(df["fuel_capacity"].min()),
        max_value=float(df["fuel_capacity"].max()),
        value=float(df["fuel_capacity"].median())
    )

    prediction_fuel_efficiency = st.number_input(
        "Fuel Efficiency",
        min_value=float(df["fuel_efficiency"].min()),
        max_value=float(df["fuel_efficiency"].max()),
        value=float(df["fuel_efficiency"].median())
    )

    prediction_power_perf_factor = st.number_input(
        "Power Performance Factor",
        min_value=float(df["power_perf_factor"].min()),
        max_value=float(df["power_perf_factor"].max()),
        value=float(df["power_perf_factor"].median())
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🚀 Predict Sales", width="stretch"):

    prediction_data = pd.DataFrame(
        {
            "manufacturer": [prediction_manufacturer],
            "vehicle_type": [prediction_vehicle_type],
            "price_in_thousands": [prediction_price],
            "engine_size": [prediction_engine_size],
            "horsepower": [prediction_horsepower],
            "wheelbase": [prediction_wheelbase],
            "width": [prediction_width],
            "length": [prediction_length],
            "curb_weight": [prediction_curb_weight],
            "fuel_capacity": [prediction_fuel_capacity],
            "fuel_efficiency": [prediction_fuel_efficiency],
            "power_perf_factor": [
                prediction_power_perf_factor
            ]
        }
    )

    predicted_sales = model.predict(prediction_data)[0]

    st.success(
        f"### Predicted Sales: {predicted_sales:.2f} Thousand Units"
    )

    st.info(
        "⚠️ This prediction is based on a small historical dataset "
        "and should be treated as an analytical estimate, not a "
        "production forecast."
    )