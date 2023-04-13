import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime as dt
import streamlit as st

st.title('LSTM Forecast')

# Get user input for stock ticker
ticker = st.text_input('Enter a stock symbol (e.g. AAPL)', 'RELIANCE.NS', key='symbol-input5')
today = dt.datetime.today()
start_date = st.date_input('Start date:',
                                   today - dt.timedelta(days=365*1),  # The default time frame is 1 year.
                                   min_value=today - dt.timedelta(days=365*4),
                                   max_value=today - dt.timedelta(days=31*2))
end_date = st.date_input('End date:',
                                 min_value=start_date +
                                 dt.timedelta(days=31*2),
                                 max_value=today)

#Get Stock data
df3 = yf.download(ticker, start_date, end_date)

d1 = df3.copy()
d1.reset_index(inplace=True)
d1['Date'] = pd.to_datetime(d1['Date']).dt.date
st.dataframe(d1)

y = df3['Close'].fillna(method='ffill')
y = y.values.reshape(-1, 1)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# generate the input and output sequences
n_lookback = 50  # length of input sequences (lookback period)
n_forecast = 5  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# fit the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=50))
model.add(Dense(n_forecast))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

# generate the forecasts
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

#rmse = (mean_squared_error(X_, Y_))

# organize the results in a data frame
df3_past = df3[['Close']].reset_index()
df3_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
df3_past['Date'] = pd.to_datetime(df3_past['Date'])
df3_past['Forecast'] = np.nan
df3_past['Forecast'].iloc[-1] = df3_past['Actual'].iloc[-1]

df3_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df3_future['Date'] = pd.date_range(start=df3_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df3_future['Forecast'] = Y_.flatten()
df3_future['Actual'] = np.nan

results = df3_past.append(df3_future).set_index('Date')
d13 = results.copy()
d13.reset_index(inplace=True)
d13['Date'] = pd.to_datetime(d13['Date']).dt.date
st.write(d13.tail(6))

# plot the results
st.line_chart(results, use_container_width=True)

import plotly.express as px

# display the RMSE
#st.write('RMSE:', round(sqrt(rmse), 2))
