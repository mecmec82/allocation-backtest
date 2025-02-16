import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import fetch_data, calculate_returns, calculate_portfolio_returns, calculate_performance_metrics, calculate_correlation_matrix, identify_risk_regime

st.title("Investment Portfolio Dashboard")

# 1. User Inputs
st.header("Portfolio Configuration")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# ETF Selection
available_etfs = ["SPY", "TLT", "GLD", "XLU", "XLP"]  # Add more as needed
selected_etfs = st.multiselect("Select ETFs", available_etfs, default=["SPY", "TLT"])

# Regime Detection Parameters
st.sidebar.header("Regime Detection")
short_ma_length = st.sidebar.slider("Short Moving Average Length", 5, 50, 20)
long_ma_length = st.sidebar.slider("Long Moving Average Length", 50, 200, 100)
vix_threshold = st.sidebar.slider("VIX Threshold", 10.0, 50.0, 25.0)

# Risk-Off Allocation
st.sidebar.header("Risk-Off Allocation")
risk_off_allocation = {}
for etf in selected_etfs:
    risk_off_allocation[etf] = st.sidebar.slider(f"Risk-Off {etf} (%)", 0.0, 100.0, 0.0 if etf == "SPY" else 50.0 if etf == "TLT" else 0.0) / 100.0
# Normalize Risk-Off Allocations
total_risk_off_allocation = sum(risk_off_allocation.values())
for etf in risk_off_allocation:
    risk_off_allocation[etf] /= total_risk_off_allocation

# 2. Data Fetching and Calculation
try:
    spy_data = fetch_data(['SPY'], start_date, end_date)['SPY'] #Spy data for regime identification
    vix_data = fetch_data(['^VIX'], start_date, end_date)['^VIX']
    all_etf_data = fetch_data(selected_etfs, start_date, end_date)
    etf_returns = calculate_returns(all_etf_data)

    # Identify Risk Regimes
    spy_data_with_regime = identify_risk_regime(spy_data.copy(), short_ma_length, long_ma_length, vix_data, vix_threshold)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()  # Stop execution if there's an error

# 3. SPY Chart with Regime Highlighting
st.header("SPY Price with Risk Regimes")
fig_spy = go.Figure()
fig_spy.add_trace(go.Scatter(x=spy_data_with_regime.index, y=spy_data_with_regime['adjclose'], name='SPY Price'))

# Add shaded regions for risk-on/risk-off
risk_on_indices = spy_data_with_regime[spy_data_with_regime['Regime'] == 1].index
risk_off_indices = spy_data_with_regime[spy_data_with_regime['Regime'] == -1].index

for i in range(len(risk_on_indices)):
  if i == 0 or risk_on_indices[i] != risk_on_indices[i-1] + pd.Timedelta(days=1):
    start_date_risk_on = risk_on_indices[i]
  if i == len(risk_on_indices) -1 or risk_on_indices[i] != risk_on_indices[i+1] - pd.Timedelta(days=1):
    end_date_risk_on = risk_on_indices[i]
    fig_spy.add_vrect(x0=start_date_risk_on, x1=end_date_risk_on, fillcolor="green", opacity=0.2, layer="below", line_width=0) # Risk-On


for i in range(len(risk_off_indices)):
  if i == 0 or risk_off_indices[i] != risk_off_indices[i-1] + pd.Timedelta(days=1):
    start_date_risk_off = risk_off_indices[i]
  if i == len(risk_off_indices) -1 or risk_off_indices[i] != risk_off_indices[i+1] - pd.Timedelta(days=1):
    end_date_risk_off = risk_off_indices[i]
    fig_spy.add_vrect(x0=start_date_risk_off, x1=end_date_risk_off, fillcolor="red", opacity=0.2, layer="below", line_width=0) # Risk-Off


st.plotly_chart(fig_spy)

# 4. Backtesting and Returns Chart

#Initial Allocation to 100% SPY for backtesting
spy_only_allocation = {'SPY': 1.0}
for etf in selected_etfs:
  if etf != 'SPY':
    spy_only_allocation[etf] = 0.0

spy_only_returns = calculate_portfolio_returns(etf_returns, spy_only_allocation)

portfolio_returns = [] # Backtesting
spy_returns = [] # for comparison
cumulative_portfolio_value = 100.0
cumulative_spy_value = 100.0
portfolio_values = [cumulative_portfolio_value]
spy_values = [cumulative_spy_value]

#Backtesting
for i, date in enumerate(etf_returns.index):
  regime = spy_data_with_regime.loc[date, 'Regime']
  if regime == 1: #Risk-On - allocate to Spy
    daily_return = calculate_portfolio_returns(etf_returns.iloc[[i]], spy_only_allocation).values[0]
    cumulative_portfolio_value *= (1 + daily_return)
  elif regime == -1: # Risk-Off - Allocate to user-defined allocation
    daily_return = calculate_portfolio_returns(etf_returns.iloc[[i]], risk_off_allocation).values[0]
    cumulative_portfolio_value *= (1+daily_return)
  else: #Neutral
    daily_return = calculate_portfolio_returns(etf_returns.iloc[[i]], spy_only_allocation).values[0]
    cumulative_portfolio_value *= (1+daily_return)
  
  cumulative_spy_value *= (1 + spy_only_returns.iloc[i]) #Track SPY

  portfolio_values.append(cumulative_portfolio_value)
  spy_values.append(cumulative_spy_value)

portfolio_values = pd.Series(portfolio_values[:-1], index = etf_returns.index)
spy_values = pd.Series(spy_values[:-1], index = etf_returns.index)

st.header("Portfolio vs. SPY Returns")
fig_returns = go.Figure()
fig_returns.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values, name='Portfolio Returns'))
fig_returns.add_trace(go.Scatter(x=spy_values.index, y=spy_values, name='SPY Returns'))

st.plotly_chart(fig_returns)

# 5. Performance Metrics
st.header("Performance Metrics")

st.subheader("Portfolio Performance")
portfolio_perf = calculate_performance_metrics(calculate_returns(portfolio_values)) #Calculate performance of portfolio
st.write(portfolio_perf)

st.subheader("SPY Performance")
spy_perf = calculate_performance_metrics(calculate_returns(spy_values))
st.write(spy_perf)
