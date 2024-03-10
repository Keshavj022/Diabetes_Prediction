# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 09:32:20 2024

@author: KIIT
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/KIIT/OneDrive/Desktop/MachineLearning/SavedMachineLearningModels/trained_model.sav', 'rb'))

#definig a function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return "Non-Diabetic"
    else:
      return "Diabetic"
  
#defining a main function  
def main():
    
    #giving the title of the webApp
    st.title("Diabtes Prediction")
    
    #getting input data for the model
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("SkinThickness Value")
    Insulin = st.text_input("Insulin Value")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    Age = st.text_input("Age of the person")
    
    #code for prediction
    diagnosis = ""
    
    #creating a button for prediction
    if st.button("Diabetes test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
   
    
#for terminal command
if __name__ == '__main__':
    main()

    