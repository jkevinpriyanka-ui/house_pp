import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="Model Insights", page_icon="📊")

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

st.title("📊 Model Insights")

# --- Actual vs Predicted ---
st.subheader("Actual vs Predicted Prices")
fig = px.scatter(df, x='SalePrice', y='PredictedPrice', hover_data=['Neighborhood','GrLivArea'], title="Actual vs Predicted Prices")
fig.add_shape(type="line", x0=df['SalePrice'].min(), x1=df['SalePrice'].max(),
              y0=df['SalePrice'].min(), y1=df['SalePrice'].max(), line=dict(dash="dash", color="red"))
st.plotly_chart(fig, use_container_width=True)

# --- Top 10 Features ---
st.subheader("Top 10 Influential Features")
preprocessor = best_enet.named_steps['preprocessor']
model = best_enet.named_steps['model']

# Feature names
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name != 'remainder':
            if hasattr(trans,'get_feature_names_out'):
                feature_names.extend(trans.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)
feature_names = [f.split("__")[-1] if "__" in f else f for f in feature_names]

# Coefficients
if hasattr(model,'coef_'):
    coefs = model.coef_
else:
    st.warning("Model has no coefficients.")
    coefs = np.zeros(len(feature_names))

feat_imp = pd.DataFrame({'Feature': feature_names,'Coefficient': coefs,'AbsCoeff':np.abs(coefs)})
top10 = feat_imp.sort_values('AbsCoeff',ascending=False).head(10)

fig2 = px.bar(top10, x='AbsCoeff', y='Feature', orientation='h', title="Top 10 Features - ElasticNet", color='AbsCoeff')
fig2.update_yaxes(categoryorder='total ascending')
st.plotly_chart(fig2, use_container_width=True)
