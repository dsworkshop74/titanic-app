import streamlit as st
import joblib
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = np.max(classifier.predict_proba(scaled_data)[0])
    return pred,prob

st.title("Titanic Survival Prediction Using KNN")

pclass = st.selectbox(
    "Pclass",
    [1, 2, 3],
    index=0
)

gender = st.selectbox(
    "Gender",
    ["male","female"],
    index=0
)

gender_map = {'male':0,'female':1}
gender = gender_map[gender]

age = st.number_input("Age",min_value=1.0, max_value=100.0, value=30.0)

sibsp = st.number_input("Sibsp",min_value=0, max_value=8, value=2)

parch = st.number_input("Parch",min_value=0, max_value=6, value=2)

fare = st.number_input("Fare",min_value=0.0, max_value=500.0, value=50.0)

embarked = st.selectbox(
    "Embarked",
    ["Cherbourg","Queenstown","Southampton"],
    index=0
)

embarked_map = {'Southampton':0,'Cherbourg':1,'Queenstown':2}
embarked = embarked_map[embarked]

if st.button("Predict survival"):
    d = {'Pclass': pclass,
        'gender': gender,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked}
    
    pred,prob = predict_survival(d)
    
    prob = round(prob,2)
    
    st.write(f"Survival prediction : {pred} and prob: {prob}")