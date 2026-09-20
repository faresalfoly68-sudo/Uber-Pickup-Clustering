from fastapi import FastAPI
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI(title="Uber Clustering API 🚖")

# Load Data (اختياري لو هتستخدمه)
df = pd.read_csv("final_data_Used_In_DBSCAN.csv")

# Load Models
pipeline_spatial = joblib.load("pipeline_spatial.pkl")
pipeline_temporal = joblib.load("pipeline_temporal.pkl")


# 🟢 Root
@app.get("/")
def home():
    return {"message": "Uber Clustering API is running 🚀"}


# 🔵 1- Spatial Prediction
@app.post("/predict/spatial")
def predict_spatial(lat: float, lon: float):

    input_df = pd.DataFrame([[lat, lon]], columns=["Lat", "Lon"])
    cluster = pipeline_spatial.predict(input_df)[0]

    return {
        "mode": "spatial",
        "lat": lat,
        "lon": lon,
        "cluster": int(cluster)
    }


# 🔴 2- Spatio-temporal Prediction
@app.post("/predict/temporal")
def predict_temporal(lat: float, lon: float, date: str, time: str):

    # تحويل date + time → hour
    dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
    hour = dt.hour

    input_df = pd.DataFrame([[lat, lon, hour]], columns=["Lat", "Lon", "hour"])
    cluster = pipeline_temporal.predict(input_df)[0]

    return {
        "mode": "spatio-temporal",
        "lat": lat,
        "lon": lon,
        "hour": hour,
        "cluster": int(cluster)
    }
