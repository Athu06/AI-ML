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
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = pit.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Spliting data into Training and Testing
data_training= pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing= pd.DataFrame(data['Close'][int(len(data)*0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

x_train=[]
y_train=[]

for i in range(100, data_training.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')


past_100_days= data_training.tail(100)
final_data= past_100_days.append(data_testing, ignore_index=True)
input_data= scaler.fit_transform(final_data)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test , y_test =np.array(x_test) , np.array(y_test)
y_predicted= model.predict(x_test)
scaler = scaler.scale_

scale_factor=1/scaler[0]
y_predicted= y_predicted*scale_factor
y_test= y_test * scale_factor


st.subheader('Predictions vs Original')
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' , label='Original Price')
plt.plot(y_predicted, 'r' , label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pylot(fig2)




