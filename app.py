import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="🐅 Tiger Forecast", layout="centered")

st.title("🐅 National Tiger Population Forecast")
st.write("This app uses the **Prophet model** to forecast total tiger population in India using year-wise data from all reserves.")

# --- File uploader ---

uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Group by year ---
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]
    prophet_df = yearly[["ds", "y"]]

    # --- Split into train/test ---
    train = prophet_df.iloc[:-5]
    test = prophet_df.iloc[-5:]
    test_years = test["ds"]

    # --- Train Prophet ---
    model = Prophet()
    model.fit(train)

    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    # --- Evaluate ---
    y_true = test["y"].values
    y_pred = forecast.iloc[-5:]["yhat"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.subheader("📊 Evaluation Metrics")
    st.markdown(f"- **MAE**: {mae:.2f}")
    st.markdown(f"- **RMSE**: {rmse:.2f}")
    st.markdown(f"- **R² Score**: {r2:.2f}")

    # --- Plot forecast ---
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # --- Save and offer download ---
    forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_result.columns = ["Year", "Predicted Population", "Lower Bound", "Upper Bound"]
    forecast_result["Year"] = forecast_result["Year"].dt.year
    future_forecast = forecast_result[forecast_result["Year"] > train["ds"].dt.year.max()]

    csv = future_forecast.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Forecast CSV",
        data=csv,
        file_name="tiger_population_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Please upload the `Cleaned_dataset.csv` file to get started.")
