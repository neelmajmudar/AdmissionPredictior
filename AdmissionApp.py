import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = load_model('model.h5')

scaler = MinMaxScaler()
training_data = pd.read_csv('admissions_data.csv')
training_features = training_data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
scaler.fit(training_features)


def predict_admission_chance(data):
    data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0][0]
    return prediction * 100  # Convert to percentage

# Streamlit App

st.title("Admission Chance Predictor")

st.sidebar.header("Input Your Data")
    
# Slidebars for each input
gre_score = st.sidebar.slider("GRE Score", 260, 340, 300)
toefl_score = st.sidebar.slider("TOEFL Score", 0, 120, 110)
university_rating = st.sidebar.slider("University Rating", 1, 5, 4)
sop = st.sidebar.slider("Statement of Purpose (SOP)", 0.0, 5.0, 3.5, step=0.1)
lor = st.sidebar.slider("Letter of Recommendation (LOR)", 0.0, 5.0, 3.5, step=0.1)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 8.0, step=0.1)
research = st.sidebar.radio("Research Experience", ["Yes", "No"])
research = 1 if research == "Yes" else 0
    
# User Input Data
user_data = [gre_score, toefl_score, university_rating, sop, lor, cgpa, research]

# Display Prediction
if st.sidebar.button("Get Admission Chance"):
    admission_chance = predict_admission_chance(user_data)
    st.subheader("Predicted Admission Chance")
    st.write(f"Your chance of admission is: {admission_chance:.2f}%")
    

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.subheader("Your Input Data")
input_df = pd.DataFrame({'Feature': ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'],
                             'Value': user_data})
st.write(input_df)
st.markdown("</div>", unsafe_allow_html=True)
    
st.subheader("Model's Stats")
#Display each plot
plot_image_paths = ['Feature Importances.png', 'Actual vs Residual.png', 'Training&ValidationLoss.png', 'Training&ValidationMAE.png']
for image_path in plot_image_paths:
    st.image(image_path, caption=image_path, use_column_width=True)
