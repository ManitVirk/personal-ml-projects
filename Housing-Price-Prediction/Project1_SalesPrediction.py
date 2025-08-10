import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn

model = pickle.load(open('Project1_SalesPrediction.pkl' , 'rb'))


st.title("Project1_Sales_Prediction")

tv = st.text_input("Enter the TV sales: ")
radio = st.text_input("Enter the radio sales: ")
news = st.text_input("Enter the newspaper sales")

if st.button("Predict"):
    features = np.array([[tv,radio,news]] , dtype=np.float64)
    result = model.predict(features)
    st.write("PREDICTED SALE" , result)
