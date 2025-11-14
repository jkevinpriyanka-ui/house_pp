import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predict", layout="wide", page_icon="🔮")
st.title("🔮 Predict House Price")

csv_path = os.path.join(os.path.dirname(__file__), "..", "house_data_with_predictions.csv")
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "best_pipeline.joblib")
csv_path = os.path.abspath(csv_path)
pipeline_path = os.path.abspath(pipeline_path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

df = load_data(csv_path)
best_enet = load_pipeline(pipeline_path)

# --- User Inputs ---
st.sidebar.header("Set Feature Values")
features = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt']
input_data = {}
for f in features:
    min_val = int(df[f].min())
    max_val = int(df[f].max())
    default_val = int(df[f].median())
    input_data[f] = st.sidebar.slider(f, min_val, max_val, default_val)

input_df = pd.DataFrame([input_data])

# Add missing columns
for col in df.drop(['SalePrice','PredictedPrice','DiffPercent'],axis=1).columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Predict
try:
    pred_log = best_enet.predict(input_df)[0]  # log scale prediction
    pred_price = np.expm1(pred_log)
    st.subheader("Predicted House Price")
    st.success(f"${pred_price:,.0f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")
