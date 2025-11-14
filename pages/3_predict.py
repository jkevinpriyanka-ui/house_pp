import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predict", layout="wide")
st.title("🔮 Predict House Price")

# Load pipeline & data
best_enet = joblib.load("best_pipeline.joblib")
df = pd.read_csv("house_data_with_predictions.csv")

# User Inputs
st.sidebar.header("Set Feature Values")
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
input_data = {}

for f in features:
    min_val = int(df[f].min())
    max_val = int(df[f].max())
    default_val = int(df[f].median())
    input_data[f] = st.sidebar.slider(f, min_val, max_val, default_val)

input_df = pd.DataFrame([input_data])
for col in df.drop(['SalePrice', 'PredictedPrice', 'DiffPercent'], axis=1).columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Predict
pred_log = best_enet.predict(input_df)[0]
pred_price = np.expm1(pred_log)
st.subheader("Predicted House Price")
st.success(f"${pred_price:,.0f}")
