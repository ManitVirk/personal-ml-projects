import pandas as pd 
import numpy as np
import streamlit as st 
import pickle


#LOADING THE MODEL 

kmeans = pickle.load(open('kmeans1.pkl' , 'rb'))


def clustering(age , avg_spend , visits, promotion):
    new_customer = np.array([[age , avg_spend , visits, promotion]])
    predicted_cluster = kmeans.predict(new_customer)

    if predicted_cluster == 0:
        return "Daily"
    elif predicted_cluster == 1:

        return "Weekly"
    else:
        return "Promotion"



st.title("Customer Segmentation")
st.write("Enter Customer Info: ")

#row1
col1 , col2 = st.columns(2)

with col1: 
    st.subheader("Customer Age")
    age = st.number_input('Age' , min_value=0,max_value=99)


with col2: 
    st.subheader("Customer Spent Time")
    avg_spend = st.number_input('Average Spent' , min_value=0,max_value=1000)


#row2
col1 , col2 = st.columns(2)

with col1: 
    st.subheader("Visits Per Week")
    vists_per_week =  st.number_input('Visits Per Week' , min_value=0,max_value=99)


with col2: 
    st.subheader("Promotion Interest: ")
    promotion_interest = st.number_input('Promotion Interest' , min_value=0,max_value=1000)



if st.button("Predict Cluster"):
    cluster_label = clustering(age,avg_spend,vists_per_week,promotion_interest)
    st.success(f'The customer belongs to the "{cluster_label}" cluster.')
