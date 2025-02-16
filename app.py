import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import fetch_data, calculate_returns, calculate_portfolio_returns, calculate_performance_metrics, calculate_correlation_matrix

st.title("Investment Portfolio Dashboard")

# 1. User Inputs
st.header("Portfolio Configuration")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# ETF Selection
available_etfs = ["SPY", "TLT", "GLD", "XLU", "XLP"]  # Add more as needed
selected_etfs = st.multiselect("Select ETFs", available_etfs, default=["SPY", "TLT"])

# Risk Regime (Manual)
risk_regime = st.selectbox("Risk Regime", ["Risk-On", "Risk-Off"])

# Allocation
allocation = {}
for etf in selected_etfs:
    allocation[etf] = st.slider(f"Allocation to {etf} (%)", 0.0, 100.0, 0.0 if etf != "SPY" else 70.0) / 100.0  # Default SPY 70%

# Normalize Allocations
total_allocation = sum(allocation.values())
for etf in allocation:
    allocation[etf] /= total_allocation

# Display Allocation
st.write("Current Allocation:")
st.write(allocation)


# 2. Data Fetching and Calculation
try:
    data = fetch_data(selected_etfs, start_date, end_date)
    returns = calculate_returns(data)
    portfolio_returns = calculate_portfolio_returns(returns, allocation)
    performance = calculate_performance_metrics(portfolio_returns)
    correlation_matrix = calculate_correlation_matrix(returns)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()  # Stop execution if there's an error

# 3. Risk-Off Adjustment (Illustrative)
if risk_regime == "Risk-Off":
    # Example: Shift allocation from SPY to TLT
    risk_off_allocation = {"SPY": 0.2, "TLT": 0.8} # Reduce SPY to 20%, increase TLT to 80%
    #Normalize Allocation based on ETFs that the user selected
    risk_off_allocation_adj = {}
    for etf in selected_etfs:
      if etf in risk_off_allocation:
        risk_off_allocation_adj[etf] = risk_off_allocation[etf]
      else:
        risk_off_allocation_adj[etf] = allocation[etf] #other ETFs stay the same

    total_allocation_adj = sum(risk_off_allocation_adj.values())
    for etf in risk_off_allocation_adj:
        risk_off_allocation_adj[etf] /= total_allocation_adj
    
    st.write("Risk-Off Allocation:")
    st.write(risk_off_allocation_adj)

    portfolio_returns = calculate_portfolio_returns(returns, risk_off_allocation_adj)  # Recalculate

    performance = calculate_performance_metrics(portfolio_returns)

# 4. Display Results
st.header("Performance Metrics")
st.write(performance)


# 5. Charts
st.header("Visualizations")
#Cumulative Return plot
cumulative_returns = (1 + portfolio_returns).cumprod()
fig_cumulative_returns = px.line(cumulative_returns, title="Cumulative Portfolio Returns")
st.plotly_chart(fig_cumulative_returns)
st.write("Correlation Matrix:")
st.dataframe(correlation_matrix)
