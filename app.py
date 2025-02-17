import streamlit as st
import pandas as pd
import yahoo_fin.stock_info as si
import plotly.graph_objects as go
import numpy as np
from datetime import date, timedelta

st.title("Customizable BTC/SPY Allocation Strategy Backtest")
st.write("Compares a customizable BTC/SPY allocation strategy against 100% SPY and 100% BTC benchmarks with transaction fees and hover labels.")

# --- User Inputs ---
st.sidebar.header("Backtest Settings")

date_option = st.sidebar.selectbox("Date Range Options", ["Specific Dates", "Relative to Today"])

if date_option == "Specific Dates":
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
else:
    relative_period = st.sidebar.selectbox("Relative Period",
                                          ["Last 30 Days", "Last 90 Days", "Year to Date", "Last Year", "Last 5 Years"])
    today = date.today()
    if relative_period == "Last 30 Days":
        start_date = today - timedelta(days=30)
        end_date = today
    elif relative_period == "Last 90 Days":
        start_date = today - timedelta(days=90)
        end_date = today
    elif relative_period == "Year to Date":
        start_date = date(today.year, 1, 1)
        end_date = today
    elif relative_period == "Last Year":
        start_date = date(today.year - 1, today.month, today.day)
        end_date = today
    elif relative_period == "Last 5 Years":
        start_date = today - timedelta(days=5*365) # approx, leap years not considered
        end_date = today
    start_date = pd.to_datetime(start_date) # Convert to datetime for yahoo_fin
    end_date = pd.to_datetime(end_date)     # Convert to datetime for yahoo_fin


initial_investment = st.sidebar.number_input("Initial Investment", value=10000)

st.sidebar.header("Strategy Parameters")
ma_period = st.sidebar.number_input("Moving Average Period", min_value=5, max_value=200, value=20, step=5)
min_spy_allocation_percent = st.sidebar.slider("Min SPY Allocation (%)", min_value=0, max_value=100, value=50, step=10)
max_spy_allocation_percent = st.sidebar.slider("Max SPY Allocation (%)", min_value=0, max_value=100, value=100, step=10)
max_btc_allocation_percent = st.sidebar.slider("Max BTC Allocation (%)", min_value=0, max_value=100, value=20, step=5)
transaction_fee_percent = st.sidebar.number_input("Transaction Fee (%) per Switch", min_value=0.0, max_value=5.0, value=0.5, step=0.1) / 100.0


min_spy_allocation = min_spy_allocation_percent / 100.0
max_spy_allocation = max_spy_allocation_percent / 100.0
max_btc_allocation = max_btc_allocation_percent / 100.0
transaction_fee = transaction_fee_percent # already in decimal form


# --- Data Fetching Function ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker, start_date, end_date):
    try:
        data = si.get_data(ticker, start_date=start_date, end_date=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# --- Performance Metrics Calculation Functions ---
def calculate_cagr(cumulative_returns, periods_per_year=252): # Assuming 252 trading days per year
    start_value = cumulative_returns.iloc[0]
    end_value = cumulative_returns.iloc[-1]
    years = len(cumulative_returns) / periods_per_year
    cagr = (end_value / start_value)**(1 / years) - 1
    return cagr

def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252): # Assuming daily returns, risk-free rate = 0
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe_ratio = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
    return sharpe_ratio

# --- Strategy Logic and Backtesting ---
spy_data = get_historical_data("SPY", start_date, end_date)
btc_data = get_historical_data("BTC-USD", start_date, end_date)

if spy_data is None or btc_data is None:
    st.stop()

spy_data = spy_data[['adjclose', 'close']].rename(columns={'adjclose': 'SPY', 'close': 'SPY_Close_For_MA'}) # Keep close for MA calculation
btc_data = btc_data[['adjclose']].rename(columns={'adjclose': 'BTC'})
data = pd.concat([spy_data, btc_data], axis=1).dropna()

if data.empty:
    st.error("No overlapping data for SPY and BTC-USD within the selected date range.")
    st.stop()

data[f'SPY_MA_{ma_period}'] = data['SPY_Close_For_MA'].rolling(window=ma_period).mean() # SPY MA
data['BTC_SPY_Ratio'] = data['BTC'] / data['SPY']
data[f'Ratio_MA_{ma_period}'] = data['BTC_SPY_Ratio'].rolling(window=ma_period).mean() # Ratio MA

strategy_returns = []
benchmark_spy_returns = []
benchmark_btc_returns = []
portfolio_value_strategy = initial_investment
portfolio_value_benchmark_spy = initial_investment
portfolio_value_benchmark_btc = initial_investment
cumulative_values_strategy = [portfolio_value_strategy]
cumulative_values_benchmark_spy = [cumulative_value
