import streamlit as st
import joblib
import pandas as pd

# Load the model and label encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit app
st.title('Churn Prediction')

# Create input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', [0, 1])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure (months)', min_value=0, value=1)
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, format="%.2f")
total_charges = st.number_input('Total Charges', min_value=0.0, format="%.2f")

# Prepare input data
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Apply label encoders to the input data
for column in ['gender', 'Partner', 'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']:
    if column in label_encoders:
        encoder = label_encoders[column]
        input_data[column] = encoder.transform(input_data[column])
    else:
        st.error(f"Encoder for column '{column}' not found.")
        st.stop()

# Ensure that the input data has the same feature names as during model training
try:
    prediction = model.predict(input_data)
    result = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Prediction: {result}')
except Exception as e:
    st.error(f"Prediction failed: {e}")
