import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="RFM Segmentation", layout="wide")

st.title("📊 RFM Customer Segmentation Tool")
st.markdown("Upload your customer dataset and choose which columns to use for Recency, Frequency, and Monetary calculations.")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Step 2: Column selection UI
    st.subheader("🔍 Select Columns for RFM Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        date_column = st.selectbox("📅 Date Column (for Recency)", df.columns)

    with col2:
        freq_column = st.selectbox("🔄 Frequency Column", df.columns)

    with col3:
        monetary_column = st.selectbox("💰 Monetary Column", df.columns)

    # Step 3: Convert data types safely
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    
    # Clean and convert Frequency column
    df[freq_column] = (
        df[freq_column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Clean and convert Monetary column
    df[monetary_column] = (
        df[monetary_column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Step 4: RFM calculation
    st.subheader("⚙️ RFM Calculation in Progress...")
    snapshot_date = df[date_column].max() + pd.Timedelta(days=1)

    rfm = df.groupby(df.index).agg({
        date_column: lambda x: (snapshot_date - x.max()).days,
        freq_column: 'sum',
        monetary_column: 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Step 5: RFM Scoring
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])

    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

    # Step 6: Display results
    st.success("✅ RFM Calculation Complete!")
    st.dataframe(rfm.head(), use_container_width=True)

    # Step 7: Download results
    csv = rfm.to_csv(index=True)
    st.download_button(
        "📥 Download RFM Results as CSV",
        data=csv,
        file_name="rfm_segmentation.csv",
        mime="text/csv"
    )
