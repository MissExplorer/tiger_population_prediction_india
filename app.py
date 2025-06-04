# import streamlit as st
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# st.set_page_config(page_title="🐅 Tiger Forecast", layout="centered")

# st.title("🐅 National Tiger Population Forecast")
# st.write("This app uses the **Prophet model** to forecast total tiger population in India using year-wise data from all reserves.")

# # --- File uploader ---

# uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])


# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # --- Group by year ---
#     yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
#     yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
#     yearly["y"] = yearly["Tiger Population"]
#     prophet_df = yearly[["ds", "y"]]

#     # --- Split into train/test ---
#     train = prophet_df.iloc[:-5]
#     test = prophet_df.iloc[-5:]
#     test_years = test["ds"]

#     # --- Train Prophet ---
#     model = Prophet()
#     model.fit(train)

#     future = model.make_future_dataframe(periods=5, freq='Y')
#     forecast = model.predict(future)

#     # --- Evaluate ---
#     y_true = test["y"].values
#     y_pred = forecast.iloc[-5:]["yhat"].values
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)

#     st.subheader("📊 Evaluation Metrics")
#     st.markdown(f"- **MAE**: {mae:.2f}")
#     st.markdown(f"- **RMSE**: {rmse:.2f}")
#     st.markdown(f"- **R² Score**: {r2:.2f}")

#     # --- Plot forecast ---
#     fig1 = model.plot(forecast)
#     st.pyplot(fig1)

#     # --- Save and offer download ---
#     forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
#     forecast_result.columns = ["Year", "Predicted Population", "Lower Bound", "Upper Bound"]
#     forecast_result["Year"] = forecast_result["Year"].dt.year
#     future_forecast = forecast_result[forecast_result["Year"] > train["ds"].dt.year.max()]

#     csv = future_forecast.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "📥 Download Forecast CSV",
#         data=csv,
#         file_name="tiger_population_forecast.csv",
#         mime="text/csv"
#     )

# else:
#     st.info("📂 Please upload the `Cleaned_dataset.csv` file to get started.")





# import streamlit as st
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# from plotly import graph_objs as go

# # --- Page setup ---
# st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")

# st.title("🐅 National Tiger Population Forecast")
# st.write("This app uses the **Prophet model** to forecast total tiger population in India using historical data from all reserves.")

# # --- File uploader ---
# uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # --- Group by year ---
#     yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()
#     yearly["ds"] = pd.to_datetime(yearly["Year"], format="%Y")
#     yearly["y"] = yearly["Tiger Population"]
#     prophet_df = yearly[["ds", "y"]]

#     # --- Split into train/test ---
#     train = prophet_df.iloc[:-5]
#     test = prophet_df.iloc[-5:]

#     # --- Train Prophet model ---
#     model = Prophet()
#     model.fit(train)

#     # --- Forecast future ---
#     future = model.make_future_dataframe(periods=5, freq='Y')
#     forecast = model.predict(future)

#     # --- Evaluation ---
#     y_true = test["y"].values
#     y_pred = forecast.iloc[-5:]["yhat"].values
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     r2 = r2_score(y_true, y_pred)

#     st.subheader("📊 Evaluation Metrics")
#     st.markdown(f"- **MAE**: {mae:.2f}")
#     st.markdown(f"- **RMSE**: {rmse:.2f}")
#     st.markdown(f"- **R² Score**: {r2:.2f}")

#     # --- Prepare forecast table ---
#     forecast_trimmed = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
#     forecast_trimmed.columns = ["Year", "Predicted Population", "Lower Bound", "Upper Bound"]
#     forecast_trimmed["Year"] = forecast_trimmed["Year"].dt.year

#     last_year = train["ds"].dt.year.max()
#     first_future_year = last_year + 1
#     last_5_past = forecast_trimmed[forecast_trimmed["Year"].between(last_year - 4, last_year)]
#     next_5_future = forecast_trimmed[forecast_trimmed["Year"] >= first_future_year]
#     display_forecast = pd.concat([last_5_past, next_5_future])

#     # --- Display forecast table ---
#     st.subheader("📊 Forecast Summary (Last 5 + Next 5 Years)")
#     st.dataframe(display_forecast)

#     # --- Static Prophet plot ---
#     st.subheader("📈 Prophet Forecast Chart")
#     fig1 = model.plot(forecast)
#     st.pyplot(fig1)

#     # --- Interactive Plotly forecast ---
#     st.subheader("📊 Interactive Forecast Plot")

#     fig2 = go.Figure()

#     fig2.add_trace(go.Scatter(
#         x=forecast["ds"], y=forecast["yhat"],
#         name="Predicted",
#         mode="lines+markers"
#     ))

#     fig2.add_trace(go.Scatter(
#         x=forecast["ds"], y=forecast["yhat_upper"],
#         name="Upper Bound",
#         mode="lines",
#         line=dict(dash="dot"),
#         opacity=0.5
#     ))

#     fig2.add_trace(go.Scatter(
#         x=forecast["ds"], y=forecast["yhat_lower"],
#         name="Lower Bound",
#         mode="lines",
#         line=dict(dash="dot"),
#         opacity=0.5,
#         fill='tonexty',
#         fillcolor='rgba(0,100,80,0.2)'
#     ))

#     fig2.update_layout(
#         xaxis_title="Year",
#         yaxis_title="Predicted Tiger Population",
#         title="Tiger Population Forecast (Interactive)",
#         hovermode="x unified"
#     )

#     st.plotly_chart(fig2, use_container_width=True)

#     # --- Download CSV ---
#     csv = display_forecast.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "📥 Download Forecast Summary as CSV",
#         data=csv,
#         file_name="tiger_population_forecast.csv",
#         mime="text/csv"
#     )

# else:
#     st.info("📂 Please upload the `Cleaned_dataset.csv` file to get started.")




import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from plotly import graph_objs as go

# --- Page setup ---
st.set_page_config(page_title="🐅 Tiger Population Forecast", layout="centered")

st.title("🐅 National Tiger Population Forecast")
st.write("This app uses the **Prophet model** to forecast total tiger population in India using historical data from all reserves.")

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

    # --- Train Prophet model ---
    model = Prophet()
    model.fit(train)

    # --- Forecast up to 2040 ---
    last_year = yearly["Year"].max()
    periods = 2040 - last_year
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)

    # --- Evaluation ---
    y_true = test["y"].values
    y_pred = forecast.iloc[-periods - 5:-periods]["yhat"].values  # only last 5 known years
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.subheader("📊 Evaluation Metrics")
    st.markdown(f"- **MAE**: {mae:.2f}")
    st.markdown(f"- **RMSE**: {rmse:.2f}")
    st.markdown(f"- **R² Score**: {r2:.2f}")

    # --- Static Prophet plot ---
    st.subheader("📈 Prophet Forecast Chart")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # --- Interactive Plotly forecast ---
    st.subheader("📊 Interactive Forecast Plot")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        name="Predicted",
        mode="lines+markers"
    ))

    fig2.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"],
        name="Upper Bound",
        mode="lines",
        line=dict(dash="dot"),
        opacity=0.5
    ))

    fig2.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"],
        name="Lower Bound",
        mode="lines",
        line=dict(dash="dot"),
        opacity=0.5,
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)'
    ))

    fig2.update_layout(
        xaxis_title="Year",
        yaxis_title="Predicted Tiger Population",
        title="Tiger Population Forecast (Up to 2040)",
        hovermode="x unified"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # --- Download full forecast as CSV ---
    forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_out.columns = ["Year", "Predicted Population", "Lower Bound", "Upper Bound"]
    forecast_out["Year"] = forecast_out["Year"].dt.year

    csv = forecast_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full Forecast (Including Future Years)",
        data=csv,
        file_name="tiger_population_forecast.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Please upload the `Cleaned_dataset.csv` file to get started.")
