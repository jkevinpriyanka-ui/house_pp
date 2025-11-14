import streamlit as st
import pandas as pd
import plotly.express as px

st.title(" Exploratory Data Analysis")

df = pd.read_csv("house_data_with_predictions.csv")

st.subheader("SalePrice Distribution")
fig = px.histogram(df, x='SalePrice', nbins=50, title="SalePrice Distribution")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 10 Features Correlated with SalePrice")
corr = df.corr()['SalePrice'].sort_values(ascending=False)[1:11]
corr_df = corr.reset_index().rename(columns={'index':'Feature', 'SalePrice':'Correlation'})
fig2 = px.bar(corr_df, x='Feature', y='Correlation', title="Top 10 Correlated Features", color='Correlation')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Categorical Feature Analysis")
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
selected_cat = st.selectbox("Select Category", cat_cols)
fig3 = px.box(df, x=selected_cat, y='SalePrice', title=f"SalePrice vs {selected_cat}")
st.plotly_chart(fig3, use_container_width=True)
