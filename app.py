import streamlit as st
import pandas as pd
import yahoo_fin.stock_info as si
import plotly.graph_objects as go
import numpy as np

st.title("BTC/SPY Capped Allocation Strategy Backtest")
st.write("Compares a BTC/SPY capped allocation strategy (max 100%) against 100% SPY and 100% BTC benchmarks.")

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

data['SPY_MA_20'] = data['SPY_Close_For_MA'].rolling(window=20).mean() # SPY 20-day MA
data['BTC_SPY_Ratio'] = data['BTC'] / data['SPY']
data['Ratio_MA_20'] = data['BTC_SPY_Ratio'].rolling(window=20).mean()

strategy_returns = []
benchmark_spy_returns = []
benchmark_btc_returns = []
portfolio_value_strategy = initial_investment
portfolio_value_benchmark_spy = initial_investment
portfolio_value_benchmark_btc = initial_investment
cumulative_values_strategy = [portfolio_value_strategy]
cumulative_values_benchmark_spy = [portfolio_value_benchmark_spy]
cumulative_values_benchmark_btc = [portfolio_value_benchmark_btc]

trades_data = [] # List to store trade information
current_allocation = "100% SPY / 0% BTC" # Initial allocation
annotations = [] # List to store annotations for the chart


for i in range(1, len(data)):
    today = data.iloc[i]
    yesterday = data.iloc[i-1]

    ratio_today = today['BTC_SPY_Ratio']
    ma_20_yesterday_ratio = yesterday['Ratio_MA_20'] # Use yesterday's MA to make decision at market open
    spy_close_today = today['SPY_Close_For_MA']
    spy_ma_20_yesterday = yesterday['SPY_MA_20']

    spy_return = today['SPY'] / yesterday['SPY'] - 1
    btc_return = today['BTC'] / yesterday['BTC'] - 1

    # --- Independent Allocation Logic (Capped at 100%) ---
    # SPY Allocation (Base)
    if spy_close_today < spy_ma_20_yesterday:
        base_spy_allocation = 0.5  # 50% SPY (Risk-Off)
    else:
        base_spy_allocation = 1.0  # 100% SPY (Risk-On)

    # BTC Allocation
    if ratio_today > ma_20_yesterday_ratio:
        btc_allocation = 0.2  # 20% BTC (Risk-On)
    else:
        btc_allocation = 0.0  # 0% BTC (Risk-Off)

    # Combine and Cap Total Allocation
    if base_spy_allocation == 1.0 and btc_allocation == 0.2:
        strategy_allocation_spy = 0.8  # 80% SPY
        strategy_allocation_btc = 0.2  # 20% BTC
    else:
        strategy_allocation_spy = base_spy_allocation
        strategy_allocation_btc = btc_allocation

    new_allocation = f"{int(strategy_allocation_spy * 100)}% SPY / {int(strategy_allocation_btc * 100)}% BTC"


    if new_allocation != current_allocation: # Allocation switch detected
        trades_data.append({
            'Date': today.name.strftime('%Y-%m-%d'),
            'From Allocation': current_allocation,
            'To Allocation': new_allocation
        })
        annotations.append(dict(x=today.name, y=cumulative_values_strategy[-1], xref="x", yref="y",
                            text=f"Switch to<br>{new_allocation}", showarrow=True, arrowhead=1, arrowcolor="blue", bgcolor="white"))
        current_allocation = new_allocation # Update current allocation

    strategy_daily_return = (strategy_allocation_spy * spy_return) + (strategy_allocation_btc * btc_return)
    strategy_returns.append(strategy_daily_return)
    portfolio_value_strategy *= (1 + strategy_daily_return)
    cumulative_values_strategy.append(portfolio_value_strategy)

    # Benchmark (100% SPY)
    benchmark_spy_returns.append(spy_return)
    portfolio_value_benchmark_spy *= (1 + spy_return)
    cumulative_values_benchmark_spy.append(portfolio_value_benchmark_spy)

    # Benchmark (100% BTC)
    benchmark_btc_returns.append(btc_return)
    portfolio_value_benchmark_btc *= (1 + btc_return)
    cumulative_values_benchmark_btc.append(portfolio_value_benchmark_btc)


cumulative_returns_strategy = pd.Series(cumulative_values_strategy, index=data.index)
cumulative_returns_benchmark_spy = pd.Series(cumulative_values_benchmark_spy, index=data.index)
cumulative_returns_benchmark_btc = pd.Series(cumulative_values_benchmark_btc, index=data.index)

# --- Plotting ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=cumulative_returns_strategy.index, y=cumulative_returns_strategy,
                         mode='lines', name='Capped BTC/SPY MA Strategy'))
fig.add_trace(go.Scatter(x=cumulative_returns_benchmark_spy.index, y=cumulative_returns_benchmark_spy,
                         mode='lines', name='100% SPY Benchmark'))
fig.add_trace(go.Scatter(x=cumulative_returns_benchmark_btc.index, y=cumulative_returns_benchmark_btc,
                         mode='lines', name='100% BTC Benchmark'))

fig.update_layout(title='Cumulative Returns: Strategy vs Benchmarks with Allocation Switches',
                  xaxis_title='Date',
                  yaxis_title='Portfolio Value ($)',
                  xaxis_rangeslider_visible=False,
                  annotations=annotations) # Add annotations to the layout

st.plotly_chart(fig, use_container_width=True)

# --- Performance Summary Table ---
st.subheader("Performance Metrics")

strategy_daily_returns_series = pd.Series(strategy_returns, index=data.index[1:]) # Daily returns series for Sharpe Ratio
benchmark_spy_daily_returns_series = pd.Series(benchmark_spy_returns, index=data.index[1:])
benchmark_btc_daily_returns_series = pd.Series(benchmark_btc_returns, index=data.index[1:])


performance_data = {
    'Strategy': ['Capped BTC/SPY MA Strategy', '100% SPY Benchmark', '100% BTC Benchmark'],
    'CAGR': [
        calculate_cagr(cumulative_returns_strategy),
        calculate_cagr(cumulative_returns_benchmark_spy),
        calculate_cagr(cumulative_returns_benchmark_btc)
    ],
    'Max Drawdown': [
        calculate_max_drawdown(cumulative_returns_strategy),
        calculate_max_drawdown(cumulative_returns_benchmark_spy),
        calculate_max_drawdown(cumulative_returns_benchmark_btc)
    ],
    'Sharpe Ratio': [
        calculate_sharpe_ratio(strategy_daily_returns_series),
        calculate_sharpe_ratio(benchmark_spy_daily_returns_series),
        calculate_sharpe_ratio(benchmark_btc_daily_returns_series)
    ]
}

performance_df = pd.DataFrame(performance_data)
performance_df.set_index('Strategy', inplace=True) # Set Strategy as index for better display
st.dataframe(performance_df.style.format({ # Format as percentages and rounded Sharpe Ratio
    'CAGR': '{:.2%}',
    'Max Drawdown': '{:.2%}',
    'Sharpe Ratio': '{:.2f}'
}))

# --- Trade Table ---
st.subheader("Allocation Switches (Trades)")
if trades_data:
    trades_df = pd.DataFrame(trades_data)
    st.dataframe(trades_df)
else:
    st.info("No allocation switches occurred within the selected date range.")


# --- Final Portfolio Value Summary ---
st.subheader("Final Portfolio Value")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Capped BTC/SPY MA Strategy", value=f"${portfolio_value_strategy:,.2f}")
with col2:
    st.metric("100% SPY Benchmark", value=f"${portfolio_value_benchmark_spy:,.2f}")
with col3:
    st.metric("100% BTC Benchmark", value=f"${portfolio_value_benchmark_btc:,.2f}")


st.sidebar.markdown("""
---
**Disclaimer:** This is for educational purposes only and not financial advice.
Investment decisions should be based on your own research and risk tolerance.
Past performance is not indicative of future results. Bitcoin and cryptocurrencies are highly volatile and risky assets.
""")
