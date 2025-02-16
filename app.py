import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.markov_switching import MarkovRegression

st.title("SPY Index with Markov Switching Model")

# Sidebar controls
st.sidebar.header("Model Parameters")
k_states = st.sidebar.slider("Number of States", min_value=2, max_value=4, value=2, step=1, help="Number of hidden states in the Markov Switching model. 2 is generally uptrend/downtrend.")
switching_variance = st.sidebar.checkbox("Switching Variance", value=True, help="Allow variance to switch between states.")
use_log_returns = st.sidebar.checkbox("Use Log Returns", value=True, help="Analyze log returns instead of price levels. This is generally recommended for financial time series.")
# Date Range
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Data Acquisition
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

try:
    data = load_data("SPY", start_date, end_date)
    if data.empty:
        st.error("No data found for the specified date range. Please adjust the start and end dates.")
        st.stop() # stop execution if no data

    st.subheader("SPY Data")
    st.dataframe(data.head())

    # Prepare Data for Model
    if use_log_returns:
        returns = np.log(data['Adj Close']).diff().dropna()
    else:
        returns = data['Adj Close'].dropna() # Just use price if not log returns
    model = MarkovRegression(returns, k_regimes=k_states, switching_variance=switching_variance)
    try:
        results = model.fit()
    except Exception as e:
        st.error(f"Model fitting failed: {e}.  Try a different date range or different model parameters.")
        st.stop()

    st.subheader("Model Summary")
    st.write(results.summary())



    # Plotting
    st.subheader("SPY Price with Regime Highlighting")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Price Data
    ax.plot(data.index, data['Adj Close'], label='SPY Price', color='blue')

    # Get Regime Probabilities
    regime_probs = results.smoothed_marginal_probabilities
    regime_probs.index = data.index[1:] if use_log_returns else data.index #adjust index if using returns

    # Find Regime Changes and Highlight Regions
    regime_changes = regime_probs.idxmax(axis=1) #finds date of highest probability regime

    current_regime = regime_changes.iloc[0]
    start_date_regime = regime_changes.index[0]
    for i in range(1, len(regime_changes)):
        if regime_changes.iloc[i] != current_regime:
            end_date_regime = regime_changes.index[i]
            if current_regime == regime_probs.columns[0]: #assuming first regime is downtrend
                ax.axvspan(start_date_regime, end_date_regime, color='red', alpha=0.2, label='Downtrend' if i==1 else None) #only label first time to avoid dups
            else:
                ax.axvspan(start_date_regime, end_date_regime, color='green', alpha=0.2, label='Uptrend' if i==1 else None)
            current_regime = regime_changes.iloc[i]
            start_date_regime = end_date_regime #continue from the regime change
    #handle the last region
    end_date_regime = regime_changes.index[-1]
    if current_regime == regime_probs.columns[0]:  # assuming first regime is downtrend
        ax.axvspan(start_date_regime, end_date_regime, color='red', alpha=0.2,
                   label='Downtrend' if i == 1 else None)  # only label first time to avoid dups
    else:
        ax.axvspan(start_date_regime, end_date_regime, color='green', alpha=0.2,
                   label='Uptrend' if i == 1 else None)

    ax.set_xlabel("Date")
    ax.set_ylabel("SPY Price")
    ax.set_title("SPY Price with Markov Switching Regimes")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    #Regime Prob Plot
    st.subheader("Regime Probabilities")
    fig_regime, ax_regime = plt.subplots(figsize=(12, 6))
    for col in regime_probs.columns:
        ax_regime.plot(regime_probs.index, regime_probs[col], label=col)
    ax_regime.set_xlabel("Date")
    ax_regime.set_ylabel("Probability")
    ax_regime.set_title("Regime Probabilities Over Time")
    ax_regime.legend()
    ax_regime.grid(True)
    st.pyplot(fig_regime)



except Exception as e:
    st.error(f"An error occurred: {e}")
