import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
import os
import re

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="🌍 Air Quality Anomaly Detection",
    layout="wide",
    page_icon="🌱"
)
st.title("🌍 Air Quality Anomaly Detection")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1") 
)

def extract_sop_steps(raw_text):
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    lines = cleaned.splitlines()
    steps = [line.strip() for line in lines if line.strip() and
             (re.match(r'^\d+\.', line.strip()) or len(line.strip()) > 5)]
    return "\n".join(steps) if steps else cleaned

@st.cache_data
def load_default_data():
    df = pd.read_csv("AirQuality.csv") 
    # Try datetime
    datetime_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
    if datetime_cols:
        try:
            df['datetime'] = pd.to_datetime(df[datetime_cols[0]])
        except:
            df['datetime'] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
    else:
        df['datetime'] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
    return df

st.sidebar.header("📂 Upload Your Dataset (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "📂 User Uploaded File"
else:
    df = load_default_data()
    source = "✅ Default AirQuality.csv"

st.success(f"Using dataset: {source}")

st.write("### Preview of Dataset")
st.dataframe(df.head())

air_quality_keywords = ["CO", "NO2", "NOx", "O3", "SO2", "PM2.5", "PM10"]
matched = [col for col in df.columns if any(kw.lower() in col.lower() for kw in air_quality_keywords)]

if not matched:
    st.error("❌ This dataset does not appear to be Air Quality data. Please upload a valid Air Quality CSV.")
    st.stop()

contamination = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.2, 0.05, 0.01)

pollutant = st.sidebar.selectbox("Choose Pollutant for Analysis", matched)

scaler = StandardScaler()
X = scaler.fit_transform(df[[pollutant]])
clf = IsolationForest(contamination=contamination, random_state=42)
df['anomaly'] = clf.fit_predict(X)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df['datetime'], df[pollutant], label=f"{pollutant} Value", color="steelblue")
ax.scatter(df[df['anomaly']==1]['datetime'], df[df['anomaly']==1][pollutant],
           color="red", label="Anomaly", s=50)
ax.set_title(f"{pollutant} Levels Over Time", fontsize=14)
ax.set_ylabel(f"{pollutant}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

col1, col2 = st.columns(2)
col1.metric("📊 Data Points", len(df))
col2.metric("⚠️ Anomalies Detected", int(df['anomaly'].sum()))

if st.button("🤖 Run Diagnosis"):
    anomalies = df[df['anomaly'] == 1]
    if anomalies.empty:
        st.info("✅ No anomalies detected.")
    else:
        summary = f"{pollutant} spiked from {anomalies[pollutant].min():.2f} to {anomalies[pollutant].max():.2f}."
        st.subheader("📋 Anomaly Summary")
        st.write(summary)

        # Step 1: Diagnosis
        prompt = f"Air quality anomaly report: {summary}\nExplain possible causes and risks."
        response = client.chat.completions.create(
            model="qwen-3-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        diagnosis = extract_sop_steps(response.choices[0].message.content)
        st.subheader("🤖 Diagnosis")
        st.markdown(diagnosis)

        # Step 2: SOP
        sop_prompt = f"Based on this diagnosis:\n{diagnosis}\nWrite a clear 5-step SOP for safety & mitigation."
        response2 = client.chat.completions.create(
            model="qwen-3-8b",
            messages=[{"role": "user", "content": sop_prompt}],
            temperature=0.7,
        )
        sop = extract_sop_steps(response2.choices[0].message.content)
        st.subheader("🛠️ Suggested SOP")
        st.markdown(sop)
