import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

st.title("SPY Index with Hidden Markov Model")

# Sidebar controls
n_components = st.sidebar.slider("Number of Hidden States", min_value=2, max_value=4, value=2, step=1, help="Number of hidden states in the HMM.")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Data Acquisition
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

try:
    data = load_data("SPY", start_date, end_date)
    returns = np.log(data['Adj Close']).diff().dropna()

    # HMM Model
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=42)
    model.fit(returns.values.reshape(-1, 1))  # Reshape for single feature

    # Predict States
    hidden_states = model.predict(returns.values.reshape(-1, 1))

    # Plotting
    st.subheader("SPY Price with HMM Regime Highlighting")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index[1:], data['Adj Close'][1:], label='SPY Price', color='blue') # shifted index

    # Color Regions Based on Hidden State
    for i in range(n_components):
        state_indices = np.where(hidden_states == i)[0]
        if len(state_indices) > 0:
            start_date_regime = data.index[state_indices[0]+1]
            end_date_regime = data.index[state_indices[-1]+1]
            if i == 0:
                ax.axvspan(start_date_regime, end_date_regime, color='red', alpha=0.2, label='State 0')  # Downtrend approximation
            else:
                ax.axvspan(start_date_regime, end_date_regime, color='green', alpha=0.2, label=f'State {i}')  # Uptrend approximation

    ax.set_xlabel("Date")
    ax.set_ylabel("SPY Price")
    ax.set_title("SPY Price with HMM Regimes")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
