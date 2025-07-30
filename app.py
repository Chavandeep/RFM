import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🧠 Customer Segmentation with RFM + KMeans")

uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        # File reading
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")

        # Preview
        with st.expander("🔍 Preview Raw Data"):
            st.dataframe(df.head(10))

        expected_cols = {"InvoiceDate", "InvoiceNo", "CustomerID", "Quantity", "UnitPrice"}
        if not expected_cols.issubset(set(df.columns)):
            st.error(f"❌ Your file must contain: {expected_cols}")
        else:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            df.dropna(subset=["CustomerID"], inplace=True)
            df["CustomerID"] = df["CustomerID"].astype(str)
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
            rfm = df.groupby("CustomerID").agg({
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
                "InvoiceNo": "nunique",
                "TotalPrice": "sum"
            })
            rfm.columns = ["Recency", "Frequency", "Monetary"]

            # Sidebar: Cluster selector
            k = st.sidebar.slider("🔢 Choose Number of Segments", 2, 10, 4)

            # Normalize
            rfm_norm = rfm.copy()
            for col in rfm.columns:
                rfm_norm[col] = (rfm[col] - rfm[col].mean()) / rfm[col].std()

            # KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            rfm["Segment"] = kmeans.fit_predict(rfm_norm)

            # Bar Graph
            st.subheader("📊 Segment Breakdown")
            segment_counts = rfm["Segment"].value_counts().sort_index()
            fig = px.bar(
                segment_counts,
                orientation="h",
                labels={"index": "Segment", "value": "Number of Customers"},
                title="Customer Count per Segment"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table output
            with st.expander("📋 View RFM Table"):
                st.dataframe(rfm.reset_index())

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("👈 Upload a file to begin")
