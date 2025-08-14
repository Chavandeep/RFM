import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import streamlit as st
from io import StringIO

# Streamlit app title
st.title("📊 KMeans Clustering Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read file
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Column selection
    st.subheader("Select Columns for Clustering")
    selected_columns = st.multiselect(
        "Pick numeric columns to use for clustering:",
        options=df.columns,
        default=df.select_dtypes(include=[np.number]).columns.tolist()
    )

    if selected_columns:
        # Keep only selected columns and drop NaNs
        data_selected = df[selected_columns].dropna()

        if data_selected.empty:
            st.warning("No rows left after removing NaN values.")
        else:
            # Choose number of clusters
            k = st.slider("Number of clusters (k)", 2, 10, 3)

            # Run KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data_selected)

            # Add cluster labels to data
            data_selected["Cluster"] = labels
            st.subheader("Clustered Data")
            st.dataframe(data_selected)

            # Scatter plot (only if at least 2 columns are selected)
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    data_selected[selected_columns[0]],
                    data_selected[selected_columns[1]],
                    c=labels,
                    cmap="viridis"
                )
                ax.set_xlabel(selected_columns[0])
                ax.set_ylabel(selected_columns[1])
                ax.set_title("KMeans Clustering")
                plt.colorbar(scatter)
                st.pyplot(fig)
            else:
                st.info("Select at least 2 columns to see a scatter plot.")
    else:
        st.info("Please select at least one column to start clustering.")
