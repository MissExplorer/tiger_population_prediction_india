# import streamlit as st
# import pandas as pd
# import plotly.graph_objs as go

# # --- Streamlit page setup ---
# st.set_page_config(page_title="🐅 Tiger Population Trends", layout="centered")
# st.title("🐅 National Tiger Population Trends")
# st.markdown("Visualizing historical trends of India's total tiger population from your uploaded dataset.")

# # --- Upload file ---
# uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # --- Group by Year (national total) ---
#     yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()

#     # --- Interactive Plotly Line Chart ---
#     st.subheader("📈 Interactive Trend of Tiger Population")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=yearly["Year"], 
#         y=yearly["Tiger Population"],
#         mode="lines+markers",
#         name="Total Population",
#         line=dict(width=3)
#     ))

#     fig.update_layout(
#         xaxis_title="Year",
#         yaxis_title="Total Tiger Population",
#         title="National Tiger Population Over Time",
#         hovermode="x unified"
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     # --- Download historical trend as CSV ---
#     csv = yearly.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "📥 Download Trend Data as CSV",
#         data=csv,
#         file_name="tiger_population_trends.csv",
#         mime="text/csv"
#     )

# else:
#     st.info("📂 Please upload your `Cleaned_dataset.csv` file.")








import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

# --- Page Setup ---
st.set_page_config(page_title="🐅 Tiger Population Trends & Forecast", layout="centered")
st.title("🐅 Tiger Population Analysis: Linear Regression vs Prophet")
st.markdown("Upload a cleaned tiger population dataset to compare trend analysis using **Linear Regression** and **Prophet** forecasting.")

# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload cleaned CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Prepare Data ---
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
    yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
    yearly["y"] = yearly["Tiger Population"]

    # --- Linear Regression ---
    X = yearly[["Year"]]
    y = yearly["Tiger Population"]
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    y_pred_lr = lr_model.predict(X)

    # --- Prophet Model ---
    prophet_model = Prophet()
    prophet_model.fit(yearly[["ds", "y"]])
    future = prophet_model.make_future_dataframe(periods=3, freq="Y")
    forecast = prophet_model.predict(future)
    forecast_years = forecast[["ds", "yhat"]].copy()
    forecast_years["Year"] = forecast_years["ds"].dt.year

    # --- Merge Forecast with Actuals ---
    merged = pd.merge(yearly, forecast_years, on="Year", how="left")

    # --- Metrics for Prophet (only for known years) ---
    prophet_y_true = yearly["y"]
    prophet_y_pred = forecast.loc[forecast["ds"].isin(yearly["ds"])]["yhat"]
    mae_prophet = mean_absolute_error(prophet_y_true, prophet_y_pred)
    rmse_prophet = np.sqrt(mean_squared_error(prophet_y_true, prophet_y_pred))
    r2_prophet = r2_score(prophet_y_true, prophet_y_pred)

    # --- Metrics for Linear Regression ---
    mae_lr = mean_absolute_error(y, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y, y_pred_lr))
    r2_lr = r2_score(y, y_pred_lr)

    # --- Metrics Display ---
    st.subheader("📊 Model Comparison Metrics")
    st.markdown("**Linear Regression:**")
    st.markdown(f"- MAE: {mae_lr:.2f}")
    st.markdown(f"- RMSE: {rmse_lr:.2f}")
    st.markdown(f"- R² Score: {r2_lr:.2f}")
    st.markdown("**Prophet:**")
    st.markdown(f"- MAE: {mae_prophet:.2f}")
    st.markdown(f"- RMSE: {rmse_prophet:.2f}")
    st.markdown(f"- R² Score: {r2_prophet:.2f}")

    # --- Interactive Plot ---
    st.subheader("📈 Interactive Trend + Forecast Plot")
    fig = go.Figure()

    # Actual Data
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Tiger Population"],
                             mode="lines+markers", name="Actual",
                             line=dict(color="black", width=3)))

    # Linear Regression Line
    fig.add_trace(go.Scatter(x=yearly["Year"], y=y_pred_lr,
                             mode="lines", name="Linear Regression",
                             line=dict(dash="dot", color="orange")))

    # Prophet Forecast (including future)
    fig.add_trace(go.Scatter(x=forecast["ds"].dt.year, y=forecast["yhat"],
                             mode="lines", name="Prophet Forecast",
                             line=dict(color="green")))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Tiger Population",
        hovermode="x unified",
        title="Tiger Population Trends & Forecasts"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Download Combined Data ---
    export_df = yearly.copy()
    export_df["Linear Regression"] = y_pred_lr
    export_df = pd.merge(export_df, forecast_years[["Year", "yhat"]], on="Year", how="left")
    export_df.rename(columns={"yhat": "Prophet Prediction"}, inplace=True)

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Comparison Data",
        data=csv,
        file_name="tiger_trend_comparison.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Please upload your cleaned CSV file to continue.")

