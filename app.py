import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="RFM Customer Segmentation", layout="wide")

st.title("📊 RFM Customer Segmentation App")
st.markdown(
    "Upload your e-commerce transaction file (CSV or XLSX) and select the columns for "
    "**Recency, Frequency, and Monetary** to perform segmentation."
)

uploaded_file = st.file_uploader("📂 Upload your file", type=["csv", "xlsx"])


def convert_numeric(series):
    """Convert string numbers with commas to float."""
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")


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


if uploaded_file:
    try:
        file_extension = uploaded_file.name.split(".")[-1]
        df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)

        # Column selection UI
        st.sidebar.subheader("🛠 Select Columns for RFM")
        col_recency = st.sidebar.selectbox("Recency Column", df.columns)
        col_frequency = st.sidebar.selectbox("Frequency Column", df.columns)
        col_monetary = st.sidebar.selectbox("Monetary Column", df.columns)

        # Convert to numeric if needed
        df[col_recency] = convert_numeric(df[col_recency])
        df[col_frequency] = convert_numeric(df[col_frequency])
        df[col_monetary] = convert_numeric(df[col_monetary])

        # Build RFM dataframe
        rfm = df[[col_recency, col_frequency, col_monetary]].copy()
        rfm.columns = ["Recency", "Frequency", "Monetary"]

        # Select clusters
        st.sidebar.subheader("🔢 Select Number of Segments")
        num_clusters = st.sidebar.slider("Number of clusters", 2, 6, 4)

        # Clustering
        rfm_clustered = perform_clustering(rfm.copy(), num_clusters)
        cluster_counts = rfm_clustered["Cluster"].value_counts().sort_index()

        # Segment breakdown
        st.subheader("📊 Segment Breakdown")
        fig = px.bar(
            cluster_counts,
            orientation='h',
            labels={'index': 'Cluster', 'value': 'Number of Customers'},
            text_auto=True,
            color=cluster_counts.index
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cluster summary & labeling
        cluster_summary = rfm_clustered.groupby("Cluster").mean().round(2)
        rfm_clustered["Segment Label"] = rfm_clustered.apply(lambda row: label_cluster(row, cluster_summary), axis=1)
        cluster_summary["Segment Label"] = (
            rfm_clustered.groupby("Cluster")["Segment Label"].agg(lambda x: x.mode().iloc[0])
        )

        # Summary table
        st.subheader("📌 Segment Summary with Labels")
        st.dataframe(cluster_summary.reset_index(), use_container_width=True)

        # Download button
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
