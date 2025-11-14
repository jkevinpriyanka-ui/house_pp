import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="EDA", page_icon="📊")

csv_path = os.path.join(os.path.dirname(__file__), "..", "house_data_with_predictions.csv")
csv_path = os.path.abspath(csv_path)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['LogSalePrice'] = np.log1p(df['SalePrice'])  # log-transform SalePrice for skew
    return df

df = load_data(csv_path)

st.title("🏠 Exploratory Data Analysis")

# --- SalePrice distribution ---
st.subheader("SalePrice Distribution")
bins = st.slider("Number of bins for histogram", 10, 100, 50, 5)
scale_option = st.radio("Plot scale:", ["Original", "Log-transformed"], index=1)

if scale_option == "Original":
    fig = px.histogram(df, x='SalePrice', nbins=bins, title="SalePrice Distribution")
else:
    fig = px.histogram(df, x='LogSalePrice', nbins=bins, title="Log(SalePrice) Distribution")
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- Top 10 numeric correlations ---
st.subheader("Top 10 Features Correlated with SalePrice")
numeric_df = df.select_dtypes(include=['float64', 'int64'])
target = 'LogSalePrice' if scale_option == "Log-transformed" else 'SalePrice'

if target in numeric_df.columns:
    corr = numeric_df.corr()[target].drop(target).sort_values(ascending=False).head(10)
    corr_df = corr.reset_index().rename(columns={'index':'Feature', target:'Correlation'})
    fig2 = px.bar(corr_df, x='Feature', y='Correlation', color='Correlation', title=f"Top 10 Features Correlated with {target}")
    fig2.update_layout(margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.write("No numeric columns available.")

# --- Categorical feature analysis ---
st.subheader("Categorical Feature Analysis")
cat_cols = sorted(df.select_dtypes(include=['object']).columns.tolist())
if cat_cols:
    selected_cat = st.selectbox("Select a categorical feature:", cat_cols)
    y_col = 'LogSalePrice' if scale_option == "Log-transformed" else 'SalePrice'
    fig3 = px.box(df, x=selected_cat, y=y_col, title=f"{y_col} vs {selected_cat}")
    fig3.update_layout(margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.write("No categorical columns available.")
