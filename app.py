import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2013-01-01'
end = '2023-12-31'
stock = 'AAPL'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker')
data= yf.download(user_input,stock,start,end)

# Describing Data
st.subheader('Data From 2013-2023')
st.write(data.describe())

#visualization
st.subheader('Closing Price vs Time Chart')
fig = pit.figure(figsize = (12,6))