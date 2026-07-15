
import streamlit as st
import pandas as pd
import joblib

st.write("App Started")

st.title(" Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to churn using Machine Learning.")

st.sidebar.title("About")

st.sidebar.info(
    """
    Customer Churn Prediction Project

    Developed by:
    PIYUSH KUMAR YADAV

    Algorithm:
    Random Forest Classifier
    """
)

try:
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    st.success("Models Loaded Successfully")
except Exception as e:
    st.error(e)
    st.stop()

st.write("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=100)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

if st.button("Predict"):

    # Create input DataFrame
    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    # One-Hot Encoding
    data = pd.get_dummies(data)

    # Match training columns
    data = data.reindex(columns=feature_names, fill_value=0)

    # Keep same column order
    data = data[feature_names]

    # Scale input
    data = scaler.transform(data)

    # Prediction probability
    probability = model.predict_proba(data)

    # Prediction
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error(" Customer is likely to Churn")
        st.write(f"Confidence: {probability[0][1] * 100:.2f}%")
    else:
        st.success("Customer is NOT likely to Churn")
        st.write(f"Confidence: {probability[0][0] * 100:.2f}%")

    st.subheader("Customer Details")

    st.write({
        "Gender": gender,
        "Senior Citizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "Tenure": tenure,
        "Monthly Charges": MonthlyCharges,
        "Total Charges": TotalCharges
    })
    
    if prediction[0] == 1:
       confidence = probability[0][1] * 100
    else:
       confidence = probability[0][0] * 100

    st.progress(int(confidence))
    st.write(f"Confidence: {confidence:.2f}%")