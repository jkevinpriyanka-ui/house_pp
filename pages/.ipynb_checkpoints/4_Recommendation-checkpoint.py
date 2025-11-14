import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="House Recommendations", page_icon="🏠", layout="wide")
st.title("🏠 House Recommendations")

# --- Safe paths ---
csv_path = os.path.join(os.path.dirname(__file__), "..", "house_data_with_predictions.csv")
pipeline_path = os.path.join(os.path.dirname(__file__), "..", "best_pipeline.joblib")
csv_path = os.path.abspath(csv_path)
pipeline_path = os.path.abspath(pipeline_path)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['LogSalePrice'] = np.log1p(df['SalePrice'])
    return df

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

df = load_data(csv_path)
best_enet = load_pipeline(pipeline_path)

# Sidebar filters
st.sidebar.header("Filter Houses")
max_price = st.sidebar.slider("Maximum Sale Price", int(df['SalePrice'].min()), int(df['SalePrice'].max()), 300000, step=5000)
min_quality = st.sidebar.slider("Minimum Overall Quality", 1, 10, 5)
neighborhood_options = ['All'] + sorted(df['Neighborhood'].unique())
neighborhood = st.sidebar.selectbox("Select Neighborhood", neighborhood_options)

# Filter data
filtered_df = df.copy()
if neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['Neighborhood']==neighborhood]
filtered_df = filtered_df[(filtered_df['SalePrice']<=max_price)&(filtered_df['OverallQual']>=min_quality)]

# Predictions
X_cols = [c for c in df.columns if c not in ['SalePrice','PredictedPrice','DiffPercent','LogSalePrice']]
try:
    filtered_df['PredictedPrice_New'] = np.expm1(best_enet.predict(filtered_df[X_cols]))
    filtered_df['DiffPercent_New'] = ((filtered_df['PredictedPrice_New']-filtered_df['SalePrice'])/filtered_df['SalePrice'])*100
except Exception as e:
    st.error(f"Prediction failed: {e}")
    filtered_df['PredictedPrice_New'] = np.nan
    filtered_df['DiffPercent_New'] = np.nan

# Top 5 deals
top_deals = filtered_df.sort_values('DiffPercent_New',ascending=False).head(5)
st.subheader("Top 5 Recommended Houses")
st.dataframe(top_deals[['Neighborhood','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','YearBuilt','SalePrice','PredictedPrice_New','DiffPercent_New']])

# Scatter plot (log-transformed)
st.subheader("Predicted vs Sale Price - Filtered Houses")
fig = px.scatter(
    filtered_df,
    x=np.log1p(filtered_df['SalePrice']),
    y=np.log1p(filtered_df['PredictedPrice_New']),
    color='Neighborhood',
    size='OverallQual',
    hover_data=['GrLivArea','GarageCars','TotalBsmtSF','YearBuilt'],
    labels={'x':'Log(Actual Price)','y':'Log(Predicted Price)'},
    title="Filtered Actual vs Predicted Prices (Log Scale)"
)
fig.add_shape(
    type="line",
    x0=np.log1p(filtered_df['SalePrice'].min()), x1=np.log1p(filtered_df['SalePrice'].max()),
    y0=np.log1p(filtered_df['SalePrice'].min()), y1=np.log1p(filtered_df['SalePrice'].max()),
    line=dict(color='red', dash='dash')
)
fig.update_layout(template="plotly_white", height=600, title_x=0.5)
st.plotly_chart(fig, use_container_width=True)
