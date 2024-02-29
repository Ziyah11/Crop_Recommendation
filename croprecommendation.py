import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings ('ignore')
import streamlit as st 
import joblib
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('Crop_recommendation (2).csv')

st.markdown("<h1 style = 'color: #416D19; text-align: center; font-family: helvetica '>CROP RECOMMENDATION MODEL</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Oluyemi Isaiah(Ziyah)</h4>", unsafe_allow_html = True)

st.image('pngwing.com (2).png', width = 350, use_column_width = True )

st.markdown("<p>A crop recommendation machine learning model is a sophisticated solution designed to assist farmers in making informed decisions about the most suitable crops to plant based on various environmental and soil factors. This model leverages predictive analytics and machine learning algorithms to analyze key parameters such as nitrogen levels, potassium content, pH, rainfall patterns, and more.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html = True)
data.rename(columns = {'N' : 'Nitrogen', 'P' : 'Phosphorus', 'K' : 'Potassium'}, inplace = True)
st.dataframe(data, use_container_width = True )

st.sidebar.image('pngwing.com (4).png', caption= 'Welcome User')

st.sidebar.write('Feature Input')
Nitrogen= st.sidebar.number_input('Nitrogen', data['Nitrogen'].min(), data['Nitrogen'].max()+10000)
Phosphorus = st.sidebar.number_input('Phosphorus', data['Phosphorus'].min(), data['Phosphorus'].max()+ 10000)
Potassium = st.sidebar.number_input('Potassium', data['Potassium'].min(), data['Potassium'].max()+ 10000)
Temperature = st.sidebar.number_input('Temperature', int(data['temperature'].min()), int(data['temperature'].max()) + 10000)
Humidity = st.sidebar.number_input('Humidity', data['humidity'].min(), data['humidity'].max()+ 10000)
PH = st.sidebar.number_input('PH', data['ph'].min(), data['ph'].max()+ 10000)
Rainfall = st.sidebar.number_input('Rainfall', data['rainfall'].min(), data['rainfall'].max()+ 10000)

st.markdown("<br>", unsafe_allow_html= True)
st.write('Input Variables')


input_var = pd.DataFrame({'Nitrogen': [Nitrogen], 'Phosphorus': [Phosphorus],'Potassium' : [Potassium],'temperature': [Temperature],'humidity' :[Humidity], 'ph' : [PH], 'rainfall' :[Rainfall] })
st.dataframe(input_var)

model= joblib.load('croprecommendationRF.pkl')

predicter =st.button('The recommended Crop')
if predicter:
    prediction = model.predict(input_var)
    st.success(f"The recommended Crop {prediction} ")
    st.balloons()

