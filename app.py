import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="SPARX - Prototype", layout="centered")
st.title("SPARX — Sports Performance & Injury Risk Prototype")
st.markdown("Upload training logs (CSV) or enter athlete metrics to get an injury risk score and simple recommendations.")
MODEL_PATH = "model.joblib"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
model = load_model()
def predict(df):
    features = ["training_duration","rpe","sleep_hours","prev_injuries","sprint_distance","avg_hr"]
    X = df[features]
    proba = model.predict_proba(X)[:,1]
    df_result = df.copy()
    df_result["injury_risk"] = np.round(proba,3)
    df_result["risk_level"] = df_result["injury_risk"].apply(lambda x: "High" if x>0.6 else ("Medium" if x>0.35 else "Low"))
    return df_result
st.sidebar.header("Manual Input")
training_duration = st.sidebar.number_input("Training duration (minutes)", min_value=10, max_value=300, value=60)
rpe = st.sidebar.slider("RPE (1-10)", 1, 10, 6)
sleep_hours = st.sidebar.number_input("Sleep hours (last night)", min_value=2.0, max_value=12.0, value=7.0, step=0.5)
prev_injuries = st.sidebar.selectbox("Previous injury (0/1)", [0,1], index=0)
sprint_distance = st.sidebar.number_input("Sprint distance (meters per session)", min_value=0, max_value=5000, value=400)
avg_hr = st.sidebar.number_input("Avg heart rate during training", min_value=60, max_value=220, value=140)
if st.sidebar.button("Predict (manual)"):
    df_manual = pd.DataFrame([{
        "training_duration": training_duration,
        "rpe": rpe,
        "sleep_hours": sleep_hours,
        "prev_injuries": prev_injuries,
        "sprint_distance": sprint_distance,
        "avg_hr": avg_hr
    }])
    res = predict(df_manual)
    st.write(res)
    risk = res.loc[0,"injury_risk"]
    if risk > 0.6:
        st.warning("High risk of injury — recommend rest or low-impact recovery session.")
    elif risk > 0.35:
        st.info("Medium risk — consider reducing intensity and monitor recovery.")
    else:
        st.success("Low risk — continue planned training, monitor fatigue.")
st.header("Upload CSV (sample_data.csv format)")
uploaded = st.file_uploader("Upload CSV with columns: training_duration,rpe,sleep_hours,prev_injuries,sprint_distance,avg_hr", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    if set(["training_duration","rpe","sleep_hours","prev_injuries","sprint_distance","avg_hr"]).issubset(df.columns):
        res = predict(df)
        st.dataframe(res)
        st.markdown("### Summary")
        st.write(res["risk_level"].value_counts())
    else:
        st.error("CSV missing required columns. Download sample file from repo and try again.")
else:
    st.info("Or try the sample data by running `python train_model.py` to generate sample_data.csv and model.")
st.markdown("---")
st.markdown("Prototype by Athar Sharma — SPARX")
