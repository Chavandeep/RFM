import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="RFM Customer Segmentation", layout="wide")
st.title("📊 RFM Customer Segmentation Tool")
st.markdown("Upload your e-commerce transaction data and segment customers using **Recency, Frequency, Monetary (RFM)** analysis with K-Means clustering.")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

# --------------------------
# Functions
# --------------------------
def preprocess_data_dynamic(df, customer_col, date_col, invoice_col, qty_col=None, price_col=None, monetary_col=None):
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Calculate or use existing Monetary
    if monetary_col and monetary_col in df.columns:
        df["TotalAmount"] = df[monetary_col]
    else:
        df["TotalAmount"] = df[qty_col] * df[price_col]
    
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,  # Recency
        invoice_col: "nunique",  # Frequency
        "TotalAmount": "sum"  # Monetary
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm

def perform_clustering(rfm, k):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    return rfm

def label_cluster(row, cluster_summary):
    c = row["Cluster"]
    avg_r = cluster_summary.loc[c, "Recency"]
    avg_f = cluster_summary.loc[c, "Frequency"]
    avg_m = cluster_summary.loc[c, "Monetary"]
    
    if avg_r < 40 and avg_f > 10 and avg_m > 5000:
        return "Loyal High-Spenders"
    elif avg_r > 90 and avg_f < 3:
        return "At-Risk / Inactive"
    elif avg_f < 3 and avg_m < 1000:
        return "Low-Value / One-Time"
    else:
        return "Potential / Average"

# --------------------------
# Main Logic
# --------------------------
if uploaded_file:
    try:
        file_extension = uploaded_file.name.split(".")[-1]
        df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)

        st.success(f"✅ File uploaded successfully! {df.shape[0]} rows × {df.shape[1]} columns detected.")
        
        # --------------------------
        # Column Mapping UI
        # --------------------------
        st.subheader("🛠 Step 1: Map Your Columns")
        st.markdown("Select which columns in your dataset correspond to **Customer ID**, **Date**, **Invoice Number**, and Monetary fields.")
        
        col1, col2 = st.columns(2)
        with col1:
            customer_col = st.selectbox("Customer ID Column", df.columns)
            date_col = st.selectbox("Invoice Date Column", df.columns)
            invoice_col = st.selectbox("Invoice Number Column", df.columns)
        with col2:
            use_existing_monetary = st.radio("How to get Monetary Value?", ["Calculate from Quantity × Price", "Use existing column"])
            if use_existing_monetary == "Calculate from Quantity × Price":
                qty_col = st.selectbox("Quantity Column", df.columns)
                price_col = st.selectbox("Unit Price Column", df.columns)
                monetary_col = None
            else:
                monetary_col = st.selectbox("Monetary Column", df.columns)
                qty_col = None
                price_col = None

        st.markdown("---")

        # --------------------------
        # Step 2: Process Data
        # --------------------------
        rfm = preprocess_data_dynamic(df, customer_col, date_col, invoice_col, qty_col, price_col, monetary_col)

        # --------------------------
        # Step 3: Clustering
        # --------------------------
        st.sidebar.header("🔢 Clustering Settings")
        num_clusters = st.sidebar.slider("Number of clusters", 2, 6, 4)
        
        rfm_clustered = perform_clustering(rfm.copy(), num_clusters)
        cluster_counts = rfm_clustered["Cluster"].value_counts().sort_index()

        # --------------------------
        # Step 4: Visualization
        # --------------------------
        st.subheader("📊 Segment Breakdown")
        fig = px.bar(
            cluster_counts,
            orientation='h',
            labels={'index': 'Cluster', 'value': 'Number of Customers'},
            text_auto=True,
            color=cluster_counts.index
        )
        st.plotly_chart(fig, use_container_width=True)

        # --------------------------
        # Step 5: Segment Summary
        # --------------------------
        cluster_summary = rfm_clustered.groupby("Cluster").mean().round(2)
        rfm_clustered["Segment Label"] = rfm_clustered.apply(lambda row: label_cluster(row, cluster_summary), axis=1)
        cluster_summary["Segment Label"] = rfm_clustered.groupby("Cluster")["Segment Label"].agg(lambda x: x.mode().iloc[0])

        st.subheader("📌 Segment Summary with Labels")
        st.dataframe(cluster_summary.reset_index(), use_container_width=True)

        # --------------------------
        # Step 6: Download Results
        # --------------------------
        st.download_button(
            "📥 Download Segmented RFM Data",
            rfm_clustered.reset_index().to_csv(index=False).encode(),
            file_name="rfm_segmented.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("📎 Please upload a CSV or XLSX file to begin.")
