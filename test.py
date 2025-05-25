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

d = {'Pclass': 1,
 'gender': 0,
 'Age': 20.0,
 'SibSp': 3,
 'Parch': 0,
 'Fare': 7.25,
 'Embarked': 0}

pred,prob = predict_survival(d)
print(pred,prob)