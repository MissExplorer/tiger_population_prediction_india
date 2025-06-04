import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from datetime import datetime

# --- Page Setup ---
st.set_page_config(page_title="🐅 Tiger Trend: Train-Test Evaluation", layout="centered")
st.title("🐅 Tiger Population Trend Analysis with True Evaluation")
st.markdown("Compare **Linear Regression** and **Prophet** models using a proper train/test split.")

# --- Upload File ---
uploaded_file = st.file_uploader("📂 Upload cleaned CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Preprocess ---
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]

    # --- Train-Test Split ---
    split_year = yearly["Year"].quantile(0.8).astype(int)  # 80% cutoff
    train_df = yearly[yearly["Year"] <= split_year]
    test_df = yearly[yearly["Year"] > split_year]

    st.markdown(f"🧪 **Train set:** {train_df['Year'].min()} to {train_df['Year'].max()}  \n📊 **Test set:** {test_df['Year'].min()} to {test_df['Year'].max()}")

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(train_df[["Year"]], train_df["y"])
    test_df["LR_Prediction"] = lr.predict(test_df[["Year"]])

    # --- Prophet ---
    prophet = Prophet()
    prophet.fit(train_df[["ds", "y"]])
    future = test_df[["ds"]].copy()
    forecast = prophet.predict(future)
    test_df["Prophet_Prediction"] = forecast["yhat"].values

    # --- Metrics ---
    lr_mae = mean_absolute_error(test_df["y"], test_df["LR_Prediction"])
    lr_rmse = np.sqrt(mean_squared_error(test_df["y"], test_df["LR_Prediction"]))
    lr_r2 = r2_score(test_df["y"], test_df["LR_Prediction"])

    prophet_mae = mean_absolute_error(test_df["y"], test_df["Prophet_Prediction"])
    prophet_rmse = np.sqrt(mean_squared_error(test_df["y"], test_df["Prophet_Prediction"]))
    prophet_r2 = r2_score(test_df["y"], test_df["Prophet_Prediction"])

    st.subheader("📊 Test Set Metrics")
    st.markdown("**Linear Regression:**")
    st.markdown(f"- MAE: {lr_mae:.2f}")
    st.markdown(f"- RMSE: {lr_rmse:.2f}")
    st.markdown(f"- R² Score: {lr_r2:.2f}")
    st.markdown("**Prophet:**")
    st.markdown(f"- MAE: {prophet_mae:.2f}")
    st.markdown(f"- RMSE: {prophet_rmse:.2f}")
    st.markdown(f"- R² Score: {prophet_r2:.2f}")

    # --- Plotting ---
    st.subheader("📈 Actual vs Predicted (Test Set Only)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df["Year"], y=test_df["y"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=test_df["Year"], y=test_df["LR_Prediction"], mode="lines+markers", name="Linear Regression", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=test_df["Year"], y=test_df["Prophet_Prediction"], mode="lines+markers", name="Prophet", line=dict(dash='dash')))
    fig.update_layout(xaxis_title="Year", yaxis_title="Tiger Population", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- Download Option ---
    download_df = test_df[["Year", "y", "LR_Prediction", "Prophet_Prediction"]].rename(columns={
        "y": "Actual"
    })
    csv = download_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Test Predictions", data=csv, file_name="tiger_test_predictions.csv", mime="text/csv")

else:
    st.info("📂 Please upload a cleaned tiger population CSV file to begin.")
