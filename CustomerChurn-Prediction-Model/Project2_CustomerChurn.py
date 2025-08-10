import streamlit as st 
import numpy as np
import pandas as pd
import sklearn
import pickle

from sklearn.preprocessing import LabelEncoder , StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# LOADING MODEL 

model = pickle.load(open('Project2_CustomerChur.pkl'  , 'rb'))


#making app
st.title("Customer Churn Prediction APP")


gender = st.selectbox("Select Gender" , options = ["Female" , "Male"])
SeniorCitizen = st.selectbox("Are you senior citizen? " , options = ["Yes" , "No"])
Partner = st.selectbox("Do you have a partner?" , options = ["Yes" , "No"])
Dependents = st.selectbox("Are you dependent on others?" , options=["Yes" , "No"])
tenure = st.text_input("Enter your tenure")
PhoneService = st.selectbox("Do you have any Phone Service?" , options = ["Yes" , "No"])
MultipleLines = st.selectbox("Do you have multiple services?" ,  options = ["Yes" , "No" , "no phone service"])
Contract = st.selectbox("Your Contract?" , options = ["One year" , "Two year" , "Month-to_month"])
TotalCharges = st.text_input("Enter your total charges?")


#Model 

def predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges):
    data = {
        'gender' : [gender],
        'SeniorCitizen' : [SeniorCitizen],
        'Partner' : [Partner],
        'Dependents' : [Dependents],
        'tenure' : [tenure],
        'PhoneService' : [PhoneService],
        'MultipleLines' : [MultipleLines],
        'Contract' : [Contract],
        'TotalCharges' : [TotalCharges],
        
    }

    df1 = pd.DataFrame(data)

    #Encoding the categorical columns
    categorical_columns = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','Contract','TotalCharges']
    for cols in categorical_columns:
        df1[cols] = label_encoder.fit_transform(df1[cols])

    df1= scaler.fit_transform(df1)
    result = model.predict(df1).reshape(1,-1)
    return result[0]



if st.button("Predict"):

    result = predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges)

    if result == 0:

        st.write("Not Churn")
    else:
        st.write("Churn")