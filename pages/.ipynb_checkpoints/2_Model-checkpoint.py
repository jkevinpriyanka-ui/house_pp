import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

st.title("📊 Model Insights")

df = pd.read_csv("house_data_with_predictions.csv")
best_enet = joblib.load("best_pipeline.joblib")

st.subheader("Actual vs Predicted Prices")
fig = px.scatter(df, x='SalePrice', y='PredictedPrice', hover_data=['Neighborhood', 'GrLivArea'],
                 title="Actual vs Predicted Prices")
fig.add_shape(
    type="line", line=dict(dash="dash", color="red"),
    x0=df['SalePrice'].min(), x1=df['SalePrice'].max(),
    y0=df['SalePrice'].min(), y1=df['SalePrice'].max()
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 10 Influential Features")
# Extract coefficients
preprocessor = best_enet.named_steps['preprocessor']
model = best_enet.named_steps['model']

try:
    feature_names = preprocessor.get_feature_names_out()
    #if old sklearn
except
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name != 'remainder':
            if hasattr(trans, 'get_feature_names_out'):
                feature_names.extend(trans.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)

coefs = model.coef_
feat_imp = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs, 'AbsCoeff': np.abs(coefs)})
top10 = feat_imp.sort_values('AbsCoeff', ascending=False).head(10)
fig2 = px.bar(top10, x='AbsCoeff', y='Feature', orientation='h', title="Top 10 Features - ElasticNet", color='AbsCoeff')
st.plotly_chart(fig2, use_container_width=True)
