import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import streamlit as st
from matplotlib import dates as mdates
import datetime as dt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#%matplotlib inline
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
import warnings
warnings.filterwarnings('ignore')

st.title('Profit & Loss')

tickers = st.sidebar.text_input('Ticker:', 'RELIANCE.NS')

today = dt.datetime.today()
start_date = st.sidebar.date_input('Start date:',
                                   today - dt.timedelta(days=365*1),  # The default time frame is 1 year.
                                   min_value=today - dt.timedelta(days=365*4),
                                   max_value=today - dt.timedelta(days=31*2))
end_date = st.sidebar.date_input('End date:',
                                 min_value=start_date +
                                 dt.timedelta(days=31*2),
                                 max_value=today)
#Profit and Loss in trading
#1. P&L for daily
st.subheader('Daily Profit & Loss')

#Get Stock data
dataset1 = yf.download(tickers, start_date, end_date)

Start = st.sidebar.number_input('Please Input your Investment (in Rs.)', value=3000)

dataset1['Shares'] = 0
dataset1['PnL'] = 0
dataset1['End'] = Start
dataset1['Shares'] = dataset1['End'].shift(1) / dataset1['Adj Close'].shift(1)
dataset1['PnL'] = dataset1['Shares'] * (dataset1['Adj Close'] - dataset1['Adj Close'].shift(1))
dataset1['End'] = dataset1['End'].shift(1) + dataset1['PnL']
d2 = dataset1.copy()
d2.reset_index(inplace=True)
d2['Date'] = pd.to_datetime(d2['Date']).dt.date
st.write(d2)


d1 = dataset1.copy()
# Create a plot
import plotly.express as px

fig1 = px.line(d1, x=d1.index, y=d1["PnL"], title='PnL')
fig1.update_yaxes(title_text='Price') #Update Y-Axis title
fig1.update_traces(line_color='#FF0000') #Update Chart colo

# Display the plot using Streamlit
st.plotly_chart(fig1)

#2. Profit or Loss
st.subheader('Profit or Loss')

# How many shares can get with the current money?
Shares = round(int(float(Start) / dataset1['Adj Close'][0]),1)
Purchase_Price = dataset1['Adj Close'][0] # Invest in the Beginning Price
Current_Value = dataset1['Adj Close'][-1] # Value of stock of Ending Price
Purchase_Cost = Shares * Purchase_Price
Current_Value = Shares * Current_Value
Profit_or_Loss = Current_Value - Purchase_Cost
percentage_gain_or_loss = (Profit_or_Loss/Current_Value) * 100
percentage_returns = (Current_Value - Purchase_Cost)/ Purchase_Cost 
net_gains_or_losses = (dataset1['Adj Close'][-1] - dataset1['Adj Close'][0]) / dataset1['Adj Close'][0]
total_return = ((Current_Value/Purchase_Cost)-1) * 100

# Get P&L by symbol
#df_pnl_by_symbol = dataset1.groupby('user_input').sum()['PnL']

# Display total P&L
st.write('Total P&L: Rs. {:,.2f}'.format(Profit_or_Loss))
st.write('Percentage Gain or Loss: %s %%' % round(percentage_gain_or_loss,2))
st.write('Percentage of Returns: %s %%' % round(percentage_returns,2))
st.write('Net gains or losses: %s %%' % round(net_gains_or_losses,2))
st.write('Total Returns: %s %%' % round(total_return,2))