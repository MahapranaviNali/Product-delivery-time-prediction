import streamlit as st
import pandas as pd
import joblib
import altair as alt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Delivery Dashboard", layout="wide")
st.title("📦 Delivery Time Analytics & Prediction Dashboard")

# === Upload CSV ===
uploaded_file = st.file_uploader("📁 Upload your delivery data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # === Preprocess: Add Delivery Time Column ===
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Delivery Date'] = pd.to_datetime(df['Delivery Date'])
    df['Delivery Time'] = (df['Delivery Date'] - df['Order Date']).dt.total_seconds() / 3600

    st.subheader("📊 Data Overview")
    st.dataframe(df.head())

    # === KPI Cards ===
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total Deliveries", len(df))
    col2.metric("⏱️ Avg Delivery Time (hrs)", round(df['Delivery Time'].mean(), 2))
    col3.metric("🚚 Most Used Vehicle", df['Vehicle Type'].mode()[0])

    # === Charts ===
    st.subheader("📈 Visual Insights")
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("**⏳ Delivery Time Distribution**")
        hist = alt.Chart(df).mark_bar().encode(
            alt.X("Delivery Time:Q", bin=True),
            y='count()'
        )
        st.altair_chart(hist, use_container_width=True)

    with chart2:
        st.markdown("**🚗 Avg Time by Vehicle Type**")
        avg_vehicle = df.groupby('Vehicle Type')['Delivery Time'].mean().reset_index()
        bar1 = alt.Chart(avg_vehicle).mark_bar().encode(
            x='Vehicle Type',
            y='Delivery Time'
        )
        st.altair_chart(bar1, use_container_width=True)

    # === More Charts ===
    chart3, chart4 = st.columns(2)

    with chart3:
        st.markdown("**🌦️ Weather vs Delivery Time**")
        box_weather = alt.Chart(df).mark_boxplot().encode(
            x='Weather:N',
            y='Delivery Time:Q'
        )
        st.altair_chart(box_weather, use_container_width=True)

    with chart4:
        st.markdown("**🚦 Traffic Level vs Delivery Time**")
        avg_traffic = df.groupby('Traffic Level')['Delivery Time'].mean().reset_index()
        bar2 = alt.Chart(avg_traffic).mark_bar().encode(
            x='Traffic Level',
            y='Delivery Time'
        )
        st.altair_chart(bar2, use_container_width=True)

    # === Prediction Form ===
    st.sidebar.header("🔍 Predict Delivery Time")
    distance = st.sidebar.number_input("Distance (km)", min_value=1, value=10)
    traffic = st.sidebar.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
    weather = st.sidebar.selectbox("Weather", ['Clear', 'Rainy', 'Stormy'])
    vehicle = st.sidebar.selectbox("Vehicle Type", ['Bike', 'Car', 'Truck'])

    # Prepare data for model
    df_model = df.copy()
    X = df_model[['Distance', 'Traffic Level', 'Weather', 'Vehicle Type']]
    y = df_model['Delivery Time']
    df_model = pd.get_dummies(X)
    input_df = pd.DataFrame({
        'Distance': [distance],
        'Traffic Level': [traffic],
        'Weather': [weather],
        'Vehicle Type': [vehicle]
    })
    input_encoded = pd.get_dummies(input_df)

    for col in df_model.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[df_model.columns]

    model = RandomForestRegressor()
    model.fit(df_model, y)
    prediction = model.predict(input_encoded)[0]

    st.sidebar.success(f"⏱️ Predicted Delivery Time: {prediction:.2f} hours")
else:
    st.info("Please upload your delivery dataset (CSV).")
