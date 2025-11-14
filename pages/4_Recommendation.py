import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.title(" House Recommendations")
df = pd.read_csv("house_data_with_predictions.csv")
best_enet = joblib.load("best_pipeline.joblib")

# Sidebar filters
st.sidebar.header("Filter Houses")
max_price = st.sidebar.slider("Maximum Sale Price", int(df['SalePrice'].min()), int(df['SalePrice'].max()), 300000, step=5000)
min_quality = st.sidebar.slider("Minimum Overall Quality", 1, 10, 5)
neighborhood_options = ['All'] + sorted(df['Neighborhood'].unique())
neighborhood = st.sidebar.selectbox("Select Neighborhood", neighborhood_options)

# Filter data
filtered_df = df.copy()
if neighborhood != 'All':
    filtered_df = filtered_df[filtered_df['Neighborhood'] == neighborhood]
filtered_df = filtered_df[(filtered_df['SalePrice'] <= max_price) & (filtered_df['OverallQual'] >= min_quality)]

# Recalculate predictions safely
X_cols = [c for c in df.columns if c not in ['SalePrice', 'PredictedPrice', 'DiffPercent']]
filtered_df['PredictedPrice_New'] = np.expm1(best_enet.predict(filtered_df[X_cols]))
filtered_df['DiffPercent_New'] = ((filtered_df['PredictedPrice_New'] - filtered_df['SalePrice']) / filtered_df['SalePrice']) * 100

# Top 5 deals
top_deals = filtered_df.sort_values('DiffPercent_New', ascending=False).head(5)
st.subheader("Top 5 Recommended Houses")
st.dataframe(top_deals[['Neighborhood', 'OverallQual', 'GrLivArea', 'GarageCars', 
                        'TotalBsmtSF', 'YearBuilt', 'SalePrice', 'PredictedPrice_New', 'DiffPercent_New']])

# Interactive Plot: Filtered Actual vs Predicted Prices
st.subheader(" Predicted vs Sale Price - Filtered Houses")
fig = px.scatter(
    filtered_df,
    x='SalePrice',
    y='PredictedPrice_New',
    color='Neighborhood',
    size='OverallQual',
    hover_data=['GrLivArea','GarageCars','TotalBsmtSF','YearBuilt'],
    labels={'SalePrice': 'Actual Price', 'PredictedPrice_New': 'Predicted Price'},
    title="Filtered Actual vs Predicted Prices"
)
fig.add_shape(
    type="line",
    x0=filtered_df['SalePrice'].min(), x1=filtered_df['SalePrice'].max(),
    y0=filtered_df['SalePrice'].min(), y1=filtered_df['SalePrice'].max(),
    line=dict(color='red', dash='dash')
)
fig.update_layout(template="plotly_white", height=600, title_x=0.5)
st.plotly_chart(fig, use_container_width=True)
