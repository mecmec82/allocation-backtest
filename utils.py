import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
from scipy import stats

def fetch_data(tickers, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = si.get_data(ticker, start_date=start_date, end_date=end_date)
            data[ticker] = data[ticker][['adjclose', 'close']]  # Keep adjusted close and close
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue
    return pd.DataFrame(data)

def calculate_returns(data):
    """Calculates daily returns from price data."""
    return data.pct_change().dropna()

def calculate_portfolio_returns(returns, allocation):
    """Calculates portfolio returns based on allocation."""
    portfolio_returns = 0
    for ticker, weight in allocation.items():
        portfolio_returns += returns[ticker] * weight
    return portfolio_returns

def calculate_performance_metrics(returns, risk_free_rate=0.0):
    """Calculates performance metrics."""
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return)**(252/len(returns)) - 1  # Annualize
    sharpe_ratio = (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    annualized_volatility = returns.std() * np.sqrt(252)
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Volatility": annualized_volatility
    }

def calculate_correlation_matrix(returns):
    """Calculates the correlation matrix of returns."""
    return returns.corr()

def identify_risk_regime(spy_data, short_ma_length, long_ma_length, vix_data, vix_threshold):
    """Identifies risk-on/risk-off regimes based on moving averages and VIX."""
    spy_data['Short_MA'] = spy_data['adjclose'].rolling(window=short_ma_length).mean()
    spy_data['Long_MA'] = spy_data['adjclose'].rolling(window=long_ma_length).mean()
    spy_data['Regime'] = 0  # 0: Neutral
    spy_data.loc[spy_data['Short_MA'] > spy_data['Long_MA'], 'Regime'] = 1  # 1: Risk-On
    spy_data.loc[spy_data['Short_MA'] < spy_data['Long_MA'], 'Regime'] = -1 # -1: Risk-Off

    #VIX check
    spy_data['VIX'] = vix_data['close']
    spy_data.loc[spy_data['VIX'] > vix_threshold, 'Regime'] = -1

    spy_data.dropna(inplace=True) #Drop NaN values after MA calculation

    return spy_data
