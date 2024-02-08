import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression

data = pd.read_csv('USA_Housing.csv')

model=joblib.load('housePredictor(1).pkl')



st.markdown("<h1 style = 'color: #CAA6A6; text-align: center; font-family: helvetica '>HOUSE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #E8C872; text-align: center; font-family: cursive '>Built By AJAYI MARVELOUS </h4>", unsafe_allow_html = True)


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com.png')


st.markdown("<p style = 'font-family: cursive' >A house prediction project involves using data analysis and machine learning to predict the prices or other factors related to houses. It uses historical data about houses, like their location, size, and features, to make these predictions. It can be helpful for real estate professionals and individuals looking to buy or sell a house. </p>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html= True)


st.sidebar.image('pngwing.com3.png', caption = 'Welcome Matini')

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width= True)

input_choice = st.sidebar.radio('choose your input type', ['slider input','number input'])

if input_choice =='Slider Input':
    area_income = st.sidebar.slider('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.slider('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num = st.sidebar.slider('Average Number Of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.slider('Average Number Of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.slider('Area Population', data['Area Population'].min(), data['Area Population'].max()) 

else:
    area_incoem = st.sidebar.number_input('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age = st.sidebar.number_input('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num = st.sidebar.number_input('Average Number Of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.number_input('Average Number Of Bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.number_input('Area Population', data['Area Population'].min(), data['Area Population'].max()) 

    # (['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
    #    'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']

input_vars = pd.DataFrame({'Avg. Area Income': [area_incoem],
                           'Avg. Area House Age': [house_age],
                           'Avg. Area Number of Rooms': [room_num],
                           'Avg. Area Number of Bedrooms': [bedrooms],
                           'Area Population': [population]
                           })


st.markdown("<br> ",unsafe_allow_html=True)
st.markdown("<h5 style = 'margin:-30px; color: olive; font-family helvetica '>User Input Variable</5>", unsafe_allow_html = True)
st.dataframe(input_vars)

st.markdown("<br> ",unsafe_allow_html=True)

# predicted = model.predict(input_vars)
predicted = model.predict(input_vars)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')