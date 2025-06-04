import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# --- Streamlit page setup ---
st.set_page_config(page_title="🐅 Tiger Population Trends", layout="centered")
st.title("🐅 National Tiger Population Trends")
st.markdown("Visualizing historical trends of India's total tiger population from your uploaded dataset.")

# --- Upload file ---
uploaded_file = st.file_uploader("Upload your cleaned CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Group by Year (national total) ---
    yearly = df.groupby("Year", as_index=False)["Tiger Population"].sum()

    # --- Interactive Plotly Line Chart ---
    st.subheader("📈 Interactive Trend of Tiger Population")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["Year"], 
        y=yearly["Tiger Population"],
        mode="lines+markers",
        name="Total Population",
        line=dict(width=3)
    ))

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Tiger Population",
        title="National Tiger Population Over Time",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Download historical trend as CSV ---
    csv = yearly.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Trend Data as CSV",
        data=csv,
        file_name="tiger_population_trends.csv",
        mime="text/csv"
    )

else:
    st.info("📂 Please upload your `Cleaned_dataset.csv` file.")
