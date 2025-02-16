import streamlit as st
import pandas as pd
import yahoo_fin.stock_info as si
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Investment Portfolio Relative Rotation Dashboard")
st.write("Analyze asset/sector performance relative to SPY using technical indicators.")

# --- User Inputs ---
st.sidebar.header("Settings")
asset_choices = ["AGG", "GLD", "VNQ", "QQQ", "IWM", "EEM", "EFA"] # Example assets - Bonds, Gold, Real Estate, Tech, Small Cap, Emerging Mkts, Developed Mkts ex-US
selected_assets = st.sidebar.multiselect("Select Assets to Analyze (vs SPY)", asset_choices, default=["AGG", "GLD", "VNQ"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
ma_periods = st.sidebar.number_input("Moving Average Periods", min_value=5, max_value=200, value=20, step=5)
stoch_rsi_periods = st.sidebar.number_input("Stochastic RSI Periods", min_value=5, max_value=20, value=14, step=1)
stoch_rsi_k = st.sidebar.number_input("Stochastic RSI %K", min_value=1, max_value=10, value=3, step=1)
stoch_rsi_d = st.sidebar.number_input("Stochastic RSI %D", min_value=1, max_value=10, value=3, step=1)
macd_fast_period = st.sidebar.number_input("MACD Fast Period", min_value=5, max_value=30, value=12, step=1)
macd_slow_period = st.sidebar.number_input("MACD Slow Period", min_value=10, max_value=50, value=26, step=1)
macd_signal_period = st.sidebar.number_input("MACD Signal Period", min_value=3, max_value=15, value=9, step=1)

# --- Data Fetching Function ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_historical_data(ticker, start_date, end_date):
    try:
        data = si.get_data(ticker, start_date=start_date, end_date=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# --- Indicator Calculation Functions ---
def calculate_ma(df, periods, column_name='Close'): # added column_name parameter with default 'Close'
    df[f'MA_{periods}'] = df[column_name].rolling(window=periods).mean() # use column_name
    return df

def calculate_stoch_rsi(df, periods, k_periods, d_periods, column_name='Close'): # added column_name parameter with default 'Close'
    # RSI
    delta = df[column_name].diff() # use column_name
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=periods, min_periods=periods).mean()
    avg_loss = loss.rolling(window=periods, min_periods=periods).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Stochastic RSI
    rsi_min = rsi.rolling(window=periods, min_periods=periods).min()
    rsi_max = rsi.rolling(window=periods, min_periods=periods).max()
    stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_max)) * 100

    df['Stoch_RSI'] = stoch_rsi
    df[f'Stoch_RSI_K_{k_periods}'] = df['Stoch_RSI'].rolling(window=k_periods).mean()
    df[f'Stoch_RSI_D_{d_periods}'] = df[f'Stoch_RSI_K_{k_periods}'].rolling(window=d_periods).mean()
    return df

def calculate_macd(df, fast_period, slow_period, signal_period, column_name='Close'): # added column_name parameter with default 'Close'
    ema_fast = df[column_name].ewm(span=fast_period, adjust=False).mean() # use column_name
    ema_slow = df[column_name].ewm(span=slow_period, adjust=False).mean() # use column_name
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    return df

# --- Main App Logic ---
if selected_assets:
    spy_data = get_historical_data("SPY", start_date, end_date)
    if spy_data is None:
        st.stop()
    spy_data.rename(columns={'close': 'Close', 'adjclose': 'Adj Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True) # standardize column names
    spy_data = calculate_ma(spy_data.copy(), ma_periods) # Calculate MA for SPY as well for relative comparison

    for asset in selected_assets:
        asset_data = get_historical_data(asset, start_date, end_date)
        if asset_data is None:
            continue # Skip to next asset if data fetch fails

        asset_data.rename(columns={'close': 'Close', 'adjclose': 'Adj Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True) # standardize column names
        asset_data = calculate_ma(asset_data.copy(), ma_periods)
        asset_data = calculate_stoch_rsi(asset_data.copy(), stoch_rsi_periods, stoch_rsi_k, stoch_rsi_d)
        asset_data = calculate_macd(asset_data.copy(), macd_fast_period, macd_slow_period, macd_signal_period)

        # Calculate Relative Performance
        relative_df = pd.DataFrame(index=asset_data.index)
        relative_df['Relative_Ratio'] = asset_data['Close'] / spy_data['Close']
        relative_df = calculate_ma(relative_df.copy(), ma_periods, column_name='Relative_Ratio') # MA on Relative Ratio - specify column_name
        relative_df = calculate_stoch_rsi(relative_df.copy(), stoch_rsi_periods, stoch_rsi_k, stoch_rsi_d, column_name='Relative_Ratio') # Stoch RSI on Relative Ratio - specify column_name
        relative_df = calculate_macd(relative_df.copy(), macd_fast_period, macd_slow_period, macd_signal_period, column_name='Relative_Ratio') # MACD on Relative Ratio - specify column_name


        st.header(f"Analysis for {asset} vs SPY")

        # --- Plotting ---
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.4, 0.2, 0.2, 0.2],
                            subplot_titles=(f'{asset} Price vs SPY (for Reference)',
                                            f'Relative Ratio ({asset}/SPY)',
                                            f'Relative Stochastic RSI ({asset}/SPY)',
                                            f'Relative MACD ({asset}/SPY)'))

        # Price Chart
        fig.add_trace(go.Candlestick(x=asset_data.index,
                                     open=asset_data['Open'],
                                     high=asset_data['High'],
                                     low=asset_data['Low'],
                                     close=asset_data['Close'],
                                     name=asset), row=1, col=1)
        fig.add_trace(go.Scatter(x=asset_data.index, y=asset_data[f'MA_{ma_periods}'],
                                 line=dict(color='blue', width=1), name=f'{asset} MA ({ma_periods})'), row=1, col=1)
        fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data['Close'],
                                 line=dict(color='red', width=1), name='SPY (Close)', visible="legendonly"), row=1, col=1) # SPY for reference, hide by default
        fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data[f'MA_{ma_periods}'],
                                 line=dict(color='purple', width=1, dash='dash'), name=f'SPY MA ({ma_periods}) (Reference)', visible="legendonly"), row=1, col=1) # SPY MA for reference, hide by default


        # Relative Ratio Chart
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df['Relative_Ratio'],
                                 line=dict(color='black', width=1), name='Relative Ratio'), row=2, col=1)
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df[f'MA_{ma_periods}'],
                                 line=dict(color='blue', width=1), name=f'Rel. Ratio MA ({ma_periods})'), row=2, col=1)
        fig.add_hline(y=1, line=dict(color='grey', width=1, dash='dash'), row=2, col=1) # Line at 1 for reference (asset = SPY)

        # Relative Stochastic RSI Chart
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df['Stoch_RSI'],
                                 line=dict(color='purple', width=1), name='Rel. Stoch RSI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df[f'Stoch_RSI_K_{stoch_rsi_k}'],
                                 line=dict(color='blue', width=1), name=f'Rel. Stoch RSI %K ({stoch_rsi_k})'), row=3, col=1)
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df[f'Stoch_RSI_D_{stoch_rsi_d}'],
                                 line=dict(color='red', width=1), name=f'Rel. Stoch RSI %D ({stoch_rsi_d})'), row=3, col=1)
        fig.add_hline(y=20, line=dict(color='grey', width=1, dash='dash'), row=3, col=1) # Oversold line
        fig.add_hline(y=80, line=dict(color='grey', width=1, dash='dash'), row=3, col=1) # Overbought line

        # Relative MACD Chart
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df['MACD'],
                                 line=dict(color='green', width=1), name='Rel. MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=relative_df.index, y=relative_df['MACD_Signal'],
                                 line=dict(color='red', width=1), name='Rel. MACD Signal'), row=4, col=1)
        fig.add_trace(go.Bar(x=relative_df.index, y=relative_df['MACD_Histogram'],
                             marker_color=['green' if val >= 0 else 'red' for val in relative_df['MACD_Histogram']],
                             name='Rel. MACD Histogram'), row=4, col=1)
        fig.add_hline(y=0, line=dict(color='grey', width=1, dash='dash'), row=4, col=1) # Zero line for MACD

        fig.update_layout(title=f'{asset} Relative Performance vs SPY',
                          height=1000, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.write("---") # Separator between assets

else:
    st.info("Select assets from the sidebar to begin analysis.")

st.sidebar.markdown("""
---
**Disclaimer:** This is for educational purposes only and not financial advice.
Investment decisions should be based on your own research and risk tolerance.
Past performance is not indicative of future results.
""")
