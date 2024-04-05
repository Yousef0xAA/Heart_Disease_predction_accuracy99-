import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
# from graduatedproject import BinaryClassification
from sklearn.preprocessing import StandardScaler

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(13, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def predict(data):
    PATH = "model.pth"
    SCALER_PATH = "scaler.pkl"

    new_model = BinaryClassification()
    new_model.load_state_dict(torch.load(PATH))
    new_model.eval()
    data = data.reshape(-1, 1)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    data_2d = scaler.transform(data.reshape(1, -1))
    
    # Convert the input data to a PyTorch tensor
    x1 = torch.tensor(data_2d, dtype=torch.float32)
    
    # Make a prediction using the trained model
    with torch.no_grad():
        output = new_model(x1)
        predicted_class = torch.round(torch.sigmoid(output))
    
    print("Predicted class:", predicted_class.item())
    
    return predicted_class

st.title("Heart Disease Classifier")
st.write("This app predicts whether someone is likely to have heart disease based on several health factors.")

age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
sex = st.selectbox("Sex", ["Male", "Female"])
if sex == "Male":
    sex = 1
else:
    sex = 0
cp = st.selectbox("Chest pain type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
if cp == "Typical angina":
    cp = 0
elif cp == "Atypical angina":
    cp = 1
elif cp == "Non-anginal pain":
    cp = 2
else:
    cp = 3
rest_bp = st.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=300, value=120, step=1)
chol = st.number_input("Serum cholesterol (mg/dl)", min_value=0, max_value=1000, value=200, step=1)
fbs = st.selectbox("Fasting blood sugar (> 120 mg/dl)", ["False", "True"])
if fbs == "False":
    fbs = 0
else:
    fbs = 1
rest_ecg = st.selectbox("Resting electrocardiographic results", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
if rest_ecg == "Normal":
    rest_ecg = 0
elif rest_ecg == "ST-T wave abnormality":
    rest_ecg = 1
else:
    rest_ecg = 2
max_hr = st.number_input("Maximum heart rate achieved", min_value=0, max_value=300, value=150, step=1)
ex_angina = st.selectbox("Exercise-induced angina", ["No", "Yes"])
if ex_angina == "No":
    ex_angina = 0
else:
    ex_angina = 1
st_depress = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment", ["Upsloping", "Flat", "Downsloping"])
if slope == "Upsloping":
    slope = 0
elif slope == "Flat":
    slope = 1
else:
    slope = 2
num_vessels = st.selectbox("Number of major vessels colored by fluoroscopy", ["0", "1", "2", "3"])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
if thal == "Normal":
    thal = 1
elif thal == "Fixed defect":
    thal = 2
else:
    thal = 3
data = np.array([age, sex, cp, rest_bp, chol, fbs, rest_ecg, max_hr, ex_angina, st_depress, slope, num_vessels, thal])
predicted_class = predict(data)
print(predicted_class)
if predicted_class == 1:
    st.write("The model predicts that you are at risk of heart disease.")
else:
    st.write("The model predicts that you are not at risk of heart disease.")