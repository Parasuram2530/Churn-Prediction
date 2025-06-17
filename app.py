import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the training model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Streamlit

st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")
if prediction_proba > 0.5:
    st.write('The Customer is Likely to Churn.')
else :
    st.write('The Customer is not likely to churn.')