import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# -----------------------------
# Load Data (Processed)
# -----------------------------
df = pd.read_csv("final_data_Used_In_DBSCAN.csv")  

# -----------------------------
# Load Models (Pipelines)
# -----------------------------
pipeline_spatial = joblib.load("pipeline_spatial.pkl")
pipeline_temporal = joblib.load("pipeline_temporal.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🚖 Uber Clustering Dashboard")
page = st.sidebar.radio("Navigation", [
    "📊 Unsupervised Analysis",
    "🎯 Cluster Membership"
])

# =========================================================
# 🟢 PAGE 1 → Unsupervised Analysis
# =========================================================
if page == "📊 Unsupervised Analysis":

    st.title("📊 DBSCAN Clustering Analysis")

    st.markdown("### 🔹 Spatial Clustering (Lat + Lon)")

    fig1 = px.scatter(
        df,
        x="Lon",
        y="Lat",
        color=df["dbscan_cluster"].astype(str),
        title="DBSCAN Spatial Clusters",
        opacity=0.6
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### 🔹 Spatio-temporal Clustering (Lat + Lon + Hour)")

    fig2 = px.scatter(
        df,
        x="Lon",
        y="Lat",
        color=df["dbscan2_cluster"].astype(str),
        title="DBSCAN Spatio-temporal Clusters",
        opacity=0.6
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💡 Insight")

    st.info("""
    - Spatial clustering groups trips based only on geographic location.
    - Adding time (hour) creates more refined clusters.
    - Some large clusters are split into smaller, time-dependent patterns.
    """)

# =========================================================
# 🔴 PAGE 2 → Cluster Membership (Prediction)
# =========================================================
else:

    st.title("🎯 Cluster Membership Prediction")

    st.markdown("### 📥 Input")

    # -----------------------------
    # Inputs
    # -----------------------------
    lat = st.number_input("Latitude", value=40.7)
    lon = st.number_input("Longitude", value=-73.9)

    date = st.date_input("Date 📅")
    time = st.time_input("Time ⏰")

    mode = st.radio("Mode", [
        "Spatial (Location Only)",
        "Spatio-temporal (Location + Time)"
    ])

    # -----------------------------
    # Predict Button
    # -----------------------------
    if st.button("🚀  Cluster Membership"):

        # Extract hour
        hour = datetime.combine(date, time).hour

        # -----------------------------
        # Prediction
        # -----------------------------
        if mode == "Spatial (Location Only)":
            input_df = pd.DataFrame([[lat, lon]], columns=["Lat", "Lon"])
            cluster = pipeline_spatial.predict(input_df)[0]
            cluster_col = "dbscan_cluster"

        else:
            input_df = pd.DataFrame([[lat, lon, hour]], columns=["Lat", "Lon", "hour"])
            cluster = pipeline_temporal.predict(input_df)[0]
            cluster_col = "dbscan2_cluster"

        # -----------------------------
        # Result
        # -----------------------------
        st.success(f"✅ Cluster: {cluster}")

        # -----------------------------
        # Visualization
        # -----------------------------
        st.markdown("### 📊 Visualization")

        df["color"] = "Other"
        df.loc[df[cluster_col] == cluster, "color"] = "Selected Cluster"

        fig = px.scatter(
            df,
            x="Lon",
            y="Lat",
            color="color",
            color_discrete_map={
                "Other": "lightgray",
                "Selected Cluster": "blue"
            },
            opacity=0.5,
            title="Cluster Highlight"
        )

        # 🔴 User Point
        fig.add_scatter(
            x=[lon],
            y=[lat],
            mode="markers",
            marker=dict(color="red", size=12),
            name="Your Point 🔴"
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Insight
        # -----------------------------
        st.markdown("### 💡 Insight")

        if mode == "Spatial (Location Only)":
            st.info("""
            This point belongs to a geographically dense cluster.
            The prediction is based on spatial proximity to historical pickup locations.
            """)
        else:
            st.info(f"""
            This point belongs to a spatio-temporal cluster.
            The hour ({hour}) influenced the cluster assignment, capturing time-based patterns.
            """)
