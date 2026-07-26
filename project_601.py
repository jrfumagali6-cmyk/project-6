import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yfinance as yf
import streamlit as st

# Page settings
st.set_page_config(
    page_title='Project 6', 
    page_icon='📈', 
    layout='wide'
)

# Project Description
st.header('Project 6')
st.markdown('''
    **Monte Carlo simulation for a financial asset using the Geometric Brownian Motion model.**
''')

# Side Bar settings
st.sidebar.header('Options Menu')

# Ticker selection
ticker = st.sidebar.text_input('Asset Ticker', value='ITUB4.SA')

# Number of days for downloading and modeling
modeling_days = st.sidebar.number_input('Days for modeling', min_value=5, value=200)

# Days for simulation
T = st.sidebar.number_input('Days for simulation', min_value=1, value=50)

# Number of simulations
n_simulations = st.sidebar.number_input('Number of simulations', min_value=50, value=500)

# Button to run the simulation
if st.sidebar.button('▶️ Simulate'):
    # Collects quotes from the last modeling_days days.
    end_date = datetime.today()
    start_date = end_date - timedelta(days=modeling_days)
    df = yf.Ticker(ticker).history(start=start_date , end=end_date)

    ### Models the GBM parameters
    close_prices = df['Close']
    returns = close_prices.pct_change()

    # Current price (Last closing price)
    S0 = close_prices.iloc[-1]

    # Expected rate of return (μ)
    mu = np.mean(returns)

    # Volatility (σ)
    sigma = np.std(returns)

    ### Runs the simulations
    simulations = np.zeros((n_simulations, T))
    simulations[:, 0] = S0
    for t in range(1, T):
        Z = np.random.standard_normal(n_simulations)
        simulations[:, t] = simulations[:, t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

    ### Displays the simulation statistics.
    st.markdown('''
        <h6 style="font-size: 19px; text-align: center; margin-top: 15px">
            Final simulation price relative to the last closing price
        </h6>'''
        , unsafe_allow_html=True
    )

    final_prices = simulations[:, -1]
    mean_final_price = np.mean(final_prices)
    min_final_price = np.min(final_prices)
    max_final_price = np.max(final_prices)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label='Final average price', 
            value=round(mean_final_price, 2), 
            delta=round(mean_final_price - close_prices[-1], 2)
        )
    
    with col2:
        st.metric(
            label='Final minimum price', 
            value=round(min_final_price, 2), 
            delta=round(min_final_price - close_prices[-1], 2)
        )

    with col3:
        st.metric(
            label='Maximum final price', 
            value=round(max_final_price, 2), 
            delta=round(max_final_price - close_prices[-1], 2)
        )

    ### Plots the prices from the simulations.
    colors = cm.rainbow(np.linspace(0, 1, n_simulations))
    plt.figure(figsize=(10, 6))
    plt.plot(close_prices.values, label='Previous quotes', color='blue', lw=1.5)

    # Plots each simulation
    for i in range(n_simulations):
        plt.plot(np.arange(len(close_prices), len(close_prices) + T), simulations[i, :], color=colors[i], alpha=0.7, lw=1)

    # Calculates and plots the average path of the simulations.
    mean_simulation = np.mean(simulations, axis=0)
    plt.plot(np.arange(len(close_prices), len(close_prices) + T), mean_simulation, color='black', label='Average Path', lw=1, linestyle='--')

    plt.title('Monte Carlo Simulations')
    plt.xlabel('Period')
    plt.ylabel('Closing price')
    plt.legend(loc='upper left')
    plt.grid(True)
    st.pyplot(plt)

    ### Plots the histogram of the simulations.
    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=50, color='blue', alpha=0.6, edgecolor='black')
    plt.title('Histogram of final prices from Monte Carlo simulations')
    plt.xlabel('Final price')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(np.mean(final_prices), color='red', linestyle='dashed', linewidth=2, label='Final average price')
    plt.legend()
    st.pyplot(plt)

st.sidebar.markdown('''
    <p style="margin-top: 30px; text-align: center">
       Python Projects for the Financial Market<br>
    </p>
''', unsafe_allow_html=True)