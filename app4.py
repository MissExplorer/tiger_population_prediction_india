import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Page config
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")
st.title("🐅 Tiger Population Forecast (Up to 2030)")
st.markdown("This tool forecasts India's tiger population using **Prophet** and **Linear Regression** models.")

# Upload CSV
data_file = st.file_uploader("📂 Upload your cleaned tiger population CSV", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)

    # Preprocess
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]

    # Model selection
    model_choice = st.radio("Select model for forecasting:", ["Prophet", "Linear Regression"])

    # Forecast horizon
    future_years = list(range(yearly["Year"].max() + 1, 2031))
    future_df = pd.DataFrame({"Year": future_years})
    future_df["ds"] = pd.to_datetime(future_df["Year"], format="%Y")

    forecast_df = None

    if model_choice == "Prophet":
        # Prophet Model
        model = Prophet()
        model.fit(yearly[["ds", "y"]])
        full_future = pd.concat([yearly[["ds"]], future_df[["ds"]]], ignore_index=True)
        forecast = model.predict(full_future)

        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_df["Year"] = forecast_df["ds"].dt.year

        # Merge for comparison
        merged = pd.merge(yearly, forecast_df, on="Year", how="left")
        merged = merged[merged["Year"] <= yearly["Year"].max()]

        # Metrics (on known years)
        mae = mean_absolute_error(merged["y"], merged["yhat"])
        rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
        r2 = r2_score(merged["y"], merged["yhat"])

    else:
        # Linear Regression
        lr = LinearRegression()
        lr.fit(yearly[["Year"]], yearly["y"])

        pred_years = yearly["Year"].tolist() + future_years
        pred_vals = lr.predict(np.array(pred_years).reshape(-1, 1))

        forecast_df = pd.DataFrame({
            "Year": pred_years,
            "yhat": pred_vals
        })

        forecast_df["ds"] = pd.to_datetime(forecast_df["Year"], format="%Y")

        # Metrics (on known years only)
        known_preds = forecast_df[forecast_df["Year"] <= yearly["Year"].max()]
        mae = mean_absolute_error(yearly["y"], known_preds["yhat"])
        rmse = np.sqrt(mean_squared_error(yearly["y"], known_preds["yhat"]))
        r2 = r2_score(yearly["y"], known_preds["yhat"])

    # --- Plotting ---
    st.subheader("📈 Forecast vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["y"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat"], mode="lines+markers", name="Forecast"))

    if model_choice == "Prophet":
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_upper"], name="Upper Bound", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df["Year"], y=forecast_df["yhat_lower"], name="Lower Bound", fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=False))

    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- Metrics ---
    st.subheader("📊 Model Fit on Historical Data")
    st.markdown(f"**MAE:** {mae:.2f}")
    st.markdown(f"**RMSE:** {rmse:.2f}")
    st.markdown(f"**R² Score:** {r2:.2f}")

    # --- Download Option ---
    st.subheader("📥 Download Forecast")
    future_forecast = forecast_df[forecast_df["Year"].isin(future_years)][["Year", "yhat"]]
    future_forecast = future_forecast.rename(columns={"yhat": "Predicted Population"})
    st.dataframe(future_forecast)
    csv = future_forecast.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="tiger_population_forecast_2030.csv", mime="text/csv")

else:
    st.info("Upload a cleaned CSV file to start.")
