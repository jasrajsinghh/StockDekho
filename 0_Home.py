import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import db_functions as functions


# Add a title and a short description to the app.
st.markdown('''
# StockDekho
## Financial Dashboard''')

st.write("A web app to interactively visualise price information and technical indicators for an asset. "
         "Simply specify the asset’s ticker (e.g. AAPL, ADANIPORTS.NS OR RELIANCE.NS, etc.)," 
         "the time frame of interest, and whether to see a summary of the business/asset or not (optional).")


# Add a sidebar for user input.
st.sidebar.header('Stock Parameters')

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

show_summary = st.sidebar.checkbox('Show summary')

if tickers:
    df = functions.get_price(tickers, start_date, end_date)
    # Write an if/else statement to check whether the ticker exists.
    # If it does not exist, an error message will be displayed to the user.

    if df.shape[0] == 0:
        st.error('Ticker does not exist! Try again.')

    else:
        info_df = functions.get_info_df(tickers)

        st.header(info_df.loc['Name'][0])

        if show_summary:
            st.markdown('## Summary')
            test = info_df.astype(str)
            st.dataframe(test)
            df12 = yf.download(tickers, start_date, end_date, progress=False)
            df12.reset_index(inplace=True)
            df12['Date'] = pd.to_datetime(df['Date']).dt.date

            st.dataframe(df12)
            

        st.markdown('''
         Info: All prices are in Rs.
          ''')
        
        df = functions.get_MACD(df)
        df = functions.get_RSI(df)
        df = functions.get_trading_strategy(df)
        closed_dates_list = functions.get_closed_dates(df)

        
        option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'MACD', 'RSI', 'EMA', 'Volume'])

        if option == 'Close':
            st.write('Close Price')
            fig = make_subplots(rows=4,
                                cols=1,
                                shared_xaxes=False,
                                vertical_spacing=0.005,
                                row_width=[0.2, 0.3, 0.3, 0.8])

            fig = functions.plot_candlestick_chart(fig,
                                                df,
                                                row=1,
                                                plot_EMAs=False,
                                                plot_strategy=True)            
            # Update x-axis.
            fig.update_xaxes(rangebreaks=[dict(values=closed_dates_list)],
                            range=[df['Date'].iloc[0] - dt.timedelta(days=3), df['Date'].iloc[-1] + dt.timedelta(days=3)])

            # Update basic layout properties (width&height, background color, title, etc.).
            fig.update_layout(width=800,
                            height=800,
                            plot_bgcolor='#0E1117',
                            paper_bgcolor='#0E1117',
                            title={
                                'text': '{} - Dashboard'.format(tickers),
                                'y': 0.98
                                  },
                            hovermode='x unified',
                            legend=dict(orientation='h',
                                        xanchor='left',
                                        x=0.05,
                                        yanchor='bottom',
                                        y=1.003))

            # Customize axis parameters.
            axis_lw, axis_color = 2, 'white'
            fig.update_layout(xaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))  
            
            config={
                    'modeBarButtonsToAdd': ['drawline']
                }

            st.plotly_chart(fig, use_container_width=True, config=config)


   ####EK KHATAMMMM
            

        elif option == 'MACD':
            st.write('Moving Average Convergence Divergence')
            fig = make_subplots(rows=2,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.005,
                                row_width=[0.2, 0.3])

            fig = functions.plot_candlestick_chart(fig,
                                                df,
                                                row=1,
                                                plot_EMAs=False,
                                                plot_strategy=True)

            fig = functions.plot_MACD(fig, df, row=2)

            # Update x-axis.
            fig.update_xaxes(rangebreaks=[dict(values=closed_dates_list)],
                            range=[df['Date'].iloc[0] - dt.timedelta(days=3), df['Date'].iloc[-1] + dt.timedelta(days=3)])

            # Update basic layout properties (width&height, background color, title, etc.).
            fig.update_layout(width=800,
                            height=800,
                            plot_bgcolor='#0E1117',
                            paper_bgcolor='#0E1117',
                            title={
                                'text': '{} - Dashboard'.format(tickers),
                                'y': 0.98
                                    },
                            hovermode='x unified',
                            legend=dict(orientation='h',
                                        xanchor='left',
                                        x=0.05,
                                        yanchor='bottom',
                                        y=1.003))

            # Customize axis parameters.
            axis_lw, axis_color = 2, 'white'
            fig.update_layout(xaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))

            fig.update_layout(xaxis2=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis2=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))
            
            config={
                    'modeBarButtonsToAdd': ['drawline']
                   }

            st.plotly_chart(fig, use_container_width=True, config=config)
           
            


        elif option == 'RSI':
            st.write('Relative Strength Indicator')
            fig = make_subplots(rows=2,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.090,
                                row_width=[0.2, 0.3])

            fig = functions.plot_candlestick_chart(fig,
                                                df,
                                                row=1,
                                                plot_EMAs=False,
                                                plot_strategy=True)

            fig = functions.plot_RSI(fig, df, row=2)

            # Update x-axis.
            fig.update_xaxes(rangebreaks=[dict(values=closed_dates_list)],
                            range=[df['Date'].iloc[0] - dt.timedelta(days=3), df['Date'].iloc[-1] + dt.timedelta(days=3)])

            # Update basic layout properties (width&height, background color, title, etc.).
            fig.update_layout(width=800,
                            height=800,
                            plot_bgcolor='#0E1117',
                            paper_bgcolor='#0E1117',
                            title={
                                'text': '{} - Dashboard'.format(tickers),
                                'y': 0.98
                                    },
                            hovermode='x unified',
                            legend=dict(orientation='h',
                                        xanchor='left',
                                        x=0.05,
                                        yanchor='bottom',
                                        y=1.003))

            # Customize axis parameters.
            axis_lw, axis_color = 2, 'white'
            fig.update_layout(xaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))
            fig.update_layout(xaxis2=dict(linewidth=axis_lw,
                                      linecolor=axis_color,
                                      mirror=True,
                                      showgrid=False),
                          yaxis2=dict(linewidth=axis_lw,
                                      linecolor=axis_color,
                                      mirror=True,
                                      showgrid=False),
                          font=dict(color=axis_color))
            
            config={
                    'modeBarButtonsToAdd': ['drawline']
                }

            st.plotly_chart(fig, use_container_width=True, config=config)
                   
            

        elif option == 'EMA':
            st.write('Expoenetial Moving Average')
                    # Plot the four plots.
            fig = make_subplots(rows=4,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.005,
                                row_width=[0.2, 0.3, 0.3, 0.8])

            fig = functions.plot_candlestick_chart(fig,
                                                df,
                                                row=1,
                                                plot_EMAs=True,
                                                plot_strategy=True)
                        # Update x-axis.
            fig.update_xaxes(rangebreaks=[dict(values=closed_dates_list)],
                            range=[df['Date'].iloc[0] - dt.timedelta(days=3), df['Date'].iloc[-1] + dt.timedelta(days=3)])

            # Update basic layout properties (width&height, background color, title, etc.).
            fig.update_layout(width=800,
                            height=800,
                            plot_bgcolor='#0E1117',
                            paper_bgcolor='#0E1117',
                            title={
                                'text': '{} - Dashboard'.format(tickers),
                                'y': 0.98
                                    },
                            hovermode='x unified',
                            legend=dict(orientation='h',
                                        xanchor='left',
                                        x=0.05,
                                        yanchor='bottom',
                                        y=1.003))

            # Customize axis parameters.
            axis_lw, axis_color = 2, 'white'
            fig.update_layout(xaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))
            
            config={
                    'modeBarButtonsToAdd': ['drawline']
                }

            st.plotly_chart(fig, use_container_width=True, config=config)

        else:
            st.write('Volume')
            # Plot the four plots.
            fig = make_subplots(rows=2,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.002,
                                row_width=[0.2, 0.3])

            fig = functions.plot_candlestick_chart(fig,
                                                df,
                                                row=1,
                                                plot_EMAs=False,
                                                plot_strategy=True)

            fig = functions.plot_volume(fig, df, row=2)

            # Update x-axis.
            fig.update_xaxes(rangebreaks=[dict(values=closed_dates_list)],
                            range=[df['Date'].iloc[0] - dt.timedelta(days=3), df['Date'].iloc[-1] + dt.timedelta(days=3)])

            # Update basic layout properties (width&height, background color, title, etc.).
            fig.update_layout(width=800,
                            height=800,
                            plot_bgcolor='#0E1117',
                            paper_bgcolor='#0E1117',
                            title={
                                'text': '{} - Dashboard'.format(tickers),
                                'y': 0.98
                                    },
                            hovermode='x unified',
                            legend=dict(orientation='h',
                                        xanchor='left',
                                        x=0.05,
                                        yanchor='bottom',
                                        y=1.003))

            # Customize axis parameters.
            axis_lw, axis_color = 2, 'white'
            fig.update_layout(xaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis1=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))                   

            fig.update_layout(xaxis2=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            yaxis2=dict(linewidth=axis_lw,
                                        linecolor=axis_color,
                                        mirror=True,
                                        showgrid=False),
                            font=dict(color=axis_color))
            
            config={
                    'modeBarButtonsToAdd': ['drawline']
                }

            st.plotly_chart(fig, use_container_width=True, config=config)

     

              
