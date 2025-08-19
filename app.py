import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="RFM Analysis", layout="wide")

st.title("📊 RFM Analysis Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # Let user select columns
    st.subheader("Select Columns for RFM")
    col1, col2, col3 = st.columns(3)

    with col1:
        col_customer = st.selectbox("Customer ID Column", df.columns)
    with col2:
        col_date = st.selectbox("Transaction Date Column", df.columns)
    with col3:
        col_amount = st.selectbox("Amount Column", df.columns)

    # Button to trigger processing
    if st.button("Run RFM Analysis"):
        try:
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
            df = df.dropna(subset=[col_date, col_amount])

            # Reference date
            ref_date = df[col_date].max() + pd.Timedelta(days=1)

            # RFM Calculation
           rfm = df.groupby(col_customer).agg({
           col_date: lambda x: (ref_date - x.max()).days,   # Recency
           col_amount: ['count', 'sum']                     # Frequency & Monetary
           }).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            # K-Means Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            rfm[['R_Score', 'F_Score', 'M_Score']] = rfm[['Recency', 'Frequency', 'Monetary']]
            rfm['Cluster'] = kmeans.fit_predict(rfm[['Recency', 'Frequency', 'Monetary']])

            st.subheader("RFM Table with Clusters")
            st.dataframe(rfm)

            # Plotly 3D Scatter
            fig = px.scatter_3d(
                rfm, x='Recency', y='Frequency', z='Monetary',
                color='Cluster', hover_data=['CustomerID'],
                title="RFM Clusters (3D)"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during processing: {e}")
