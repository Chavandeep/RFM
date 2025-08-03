import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="📊 RFM Customer Segmentation", layout="wide")
st.title("📊 RFM Customer Segmentation Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your sales data (CSV or Excel)", type=["csv", "xls", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

    st.subheader("Step 1: Map Column Names")
    st.write("Please select the corresponding columns from your file for the required fields.")

    # Select boxes to map necessary columns
    col1, col2, col3 = st.columns(3)
    with col1:
        invoice_col = st.selectbox("Select Invoice Number column", df.columns)
    with col2:
        date_col = st.selectbox("Select Invoice Date column", df.columns)
    with col3:
        customer_col = st.selectbox("Select Customer ID column", df.columns)

    col4, col5 = st.columns(2)
    with col4:
        quantity_col = st.selectbox("Select Quantity column", df.columns)
    with col5:
        price_col = st.selectbox("Select Unit Price column", df.columns)

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df["TotalPrice"] = df[quantity_col] * df[price_col]

        ref_date = df[date_col].max() + pd.Timedelta(days=1)

        rfm = df.groupby(customer_col).agg({
            date_col: lambda x: (ref_date - x.max()).days,
            invoice_col: 'nunique',
            "TotalPrice": 'sum'
        }).reset_index()

        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        st.subheader("Step 2: RFM Table")
        st.dataframe(rfm.head(10), use_container_width=True)

        st.subheader("Step 3: KMeans Clustering")
        k = st.slider("Select number of clusters", min_value=2, max_value=8, value=4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(rfm[["Recency", "Frequency", "Monetary"]])

        summary = rfm.groupby("Cluster").agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "CustomerID": "count"
        }).rename(columns={"CustomerID": "CustomerCount"}).reset_index()

        st.subheader("📂 Segment Breakdown")
        fig = px.bar(
            summary.sort_values("CustomerCount", ascending=True),
            x="CustomerCount",
            y="Cluster",
            orientation='h',
            color="Cluster",
            title="Customers per Cluster",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⬇️ Download Results")
        st.download_button(
            label="Download RFM with Clusters (CSV)",
            data=rfm.to_csv(index=False),
            file_name="rfm_clusters.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing data: {e}")

else:
    st.info("Please upload a CSV or Excel file with transaction data.")
