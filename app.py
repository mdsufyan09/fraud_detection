import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model and column info
model = joblib.load("fraud_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Fraud Transaction Detection", page_icon="💳")
st.title("💳 Fraud Transaction Detection System")

st.markdown("""
This app predicts whether a transaction is **fraudulent or legitimate** based on its details.  
Enter the transaction details below and click **Predict Fraud**.
""")

# ---- Input Fields ----
amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, step=0.01)

transaction_time = st.text_input(
    "⏰ Transaction Date & Time (YYYY-MM-DD HH:MM:SS)",
    value=str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
)

# Convert datetime input to day/hour
try:
    dt = pd.to_datetime(transaction_time)
    tx_day = dt.day
    tx_hour = dt.hour
except Exception:
    st.error("⚠️ Invalid datetime format! Use YYYY-MM-DD HH:MM:SS")
    st.stop()

# Prepare input dataframe
input_data = pd.DataFrame([[amount, tx_day, tx_hour]], columns=model_columns)

# ---- Prediction ----
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Legitimate Transaction. (Risk: {probability*100:.2f}%)")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit & RandomForestClassifier")
