Fraud Transaction Detection System

A Streamlit-based machine learning app that detects fraudulent transactions
using a simulated transactions dataset.

## 🎯 Objective
To build a system that classifies if a transaction is **fraudulent or legitimate**.

## 🧠 Dataset
The dataset contains daily transaction `.pkl` files (April–September 2018)
with the following key columns:
- `TX_AMOUNT` – Transaction amount  
- `TX_DATETIME` – Date & time of transaction  
- `TX_FRAUD` – 1 for fraud, 0 for legitimate  

📌 **Note:**  
Frauds were simulated with simple rules.  
One major rule is:
> Any transaction with an amount **greater than 220** is marked as fraud.  
Hence, the model learns that high-value transactions are likely fraudulent —  
this is expected behavior given the dataset design.

## ⚙️ How to Run
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
Install dependencies:
pip install pandas scikit-learn streamlit joblib
Train the model:
python train_model.py
Launch the app:
streamlit run app.py
🚀 Features
Machine learning model (RandomForestClassifier)
Real-time fraud prediction
Simple and interactive Streamlit interface
📊 Output
fraud_model.pkl – trained model
model_columns.pkl – feature list
Streamlit app predicts fraud with risk percentage