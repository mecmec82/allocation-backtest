import streamlit as st
import pandas as pd
import yahoo_fin.stock_info as si
import plotly.graph_objects as go

st.title("Simple BTC/SPY Allocation Strategy Backtest")
st.write("Backtests a strategy of switching between 100% SPY and 50/50 SPY/BTC based on BTC/SPY relative ratio.")

# --- User Inputs ---
st.sidebar.header("Backtest Settings")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
initial_investment = st.sidebar.number_input("Initial Investment", value=10000)

# --- Data Fetching Function ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker, start_date, end_date):
    try:
        data = si.get_data(ticker, start_date=start_date, end_date=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# --- Strategy Logic and Backtesting ---
spy_data = get_historical_data("SPY", start_date, end_date)
btc_data = get_historical_data("BTC-USD", start_date, end_date)

if spy_data is None or btc_data is None:
    st.stop()

spy_data = spy_data[['adjclose']].rename(columns={'adjclose': 'SPY'})
btc_data = btc_data[['adjclose']].rename(columns={'adjclose': 'BTC'})
data = pd.concat([spy_data, btc_data], axis=1).dropna()

if data.empty:
    st.error("No overlapping data for SPY and BTC-USD within the selected date range.")
    st.stop()

data['BTC_SPY_Ratio'] = data['BTC'] / data['SPY']
data['Ratio_MA_20'] = data['BTC_SPY_Ratio'].rolling(window=20).mean()

strategy_returns = []
benchmark_returns = []
portfolio_value_strategy = initial_investment
portfolio_value_benchmark = initial_investment
cumulative_values_strategy = [portfolio_value_strategy]
cumulative_values_benchmark = [portfolio_value_benchmark]

for i in range(1, len(data)):
    today = data.iloc[i]
    yesterday = data.iloc[i-1]

    ratio_today = today['BTC_SPY_Ratio']
    ma_20_yesterday = yesterday['Ratio_MA_20'] # Use yesterday's MA to make decision at market open

    spy_return = today['SPY'] / yesterday['SPY'] - 1
    btc_return = today['BTC'] / yesterday['BTC'] - 1

    # Strategy Allocation Logic
    if ratio_today > ma_20_yesterday:
        # 50% SPY, 50% BTC
        strategy_allocation_spy = 0.5
        strategy_allocation_btc = 0.5
    else:
        # 100% SPY
        strategy_allocation_spy = 1.0
        strategy_allocation_btc = 0.0

    strategy_daily_return = (strategy_allocation_spy * spy_return) + (strategy_allocation_btc * btc_return)
    strategy_returns.append(strategy_daily_return)
    portfolio_value_strategy *= (1 + strategy_daily_return)
    cumulative_values_strategy.append(portfolio_value_strategy)

    # Benchmark (100% SPY)
    benchmark_returns.append(spy_return)
    portfolio_value_benchmark *= (1 + spy_return)
    cumulative_values_benchmark.append(portfolio_value_benchmark)


cumulative_returns_strategy = pd.Series(cumulative_values_strategy, index=data.index)
cumulative_returns_benchmark = pd.Series(cumulative_values_benchmark, index=data.index)

# --- Plotting ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_returns_strategy.index, y=cumulative_returns_strategy,
                         mode='lines', name='Strategy Returns (BTC/SPY MA Strategy)'))
fig.add_trace(go.Scatter(x=cumulative_returns_benchmark.index, y=cumulative_returns_benchmark,
                         mode='lines', name='Benchmark Returns (100% SPY)'))

fig.update_layout(title='Cumulative Returns: BTC/SPY MA Strategy vs 100% SPY',
                  xaxis_title='Date',
                  yaxis_title='Portfolio Value ($)',
                  xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# --- Performance Summary ---
st.subheader("Final Portfolio Value")
col1, col2 = st.columns(2)
with col1:
    st.metric("BTC/SPY MA Strategy", value=f"${portfolio_value_strategy:,.2f}")
with col2:
    st.metric("100% SPY Benchmark", value=f"${portfolio_value_benchmark:,.2f}")


st.sidebar.markdown("""
---
**Disclaimer:** This is for educational purposes only and not financial advice.
Investment decisions should be based on your own research and risk tolerance.
Past performance is not indicative of future results. Bitcoin and cryptocurrencies are highly volatile and risky assets.
""")
