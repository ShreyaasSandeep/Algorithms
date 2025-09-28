import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download("AAPL", start="2018-01-01", end="2025-01-01", auto_adjust=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

data["returns"] = data["Close"].pct_change()

data["momentum_long"] = data["Close"].pct_change(20)
data["momentum_short"] = data["Close"].pct_change(5)

data["signal_long"] = (data["momentum_long"] - data["momentum_long"].rolling(20).mean()) / data["momentum_long"].rolling(20).std()
data["signal_short"] = (data["momentum_short"] - data["momentum_short"].rolling(5).mean()) / data["momentum_short"].rolling(5).std()

rolling_window = 252

sharpe_short = (data["signal_short"].rolling(rolling_window).mean() /
                data["signal_short"].rolling(rolling_window).std()).fillna(0)
sharpe_long = (data["signal_long"].rolling(rolling_window).mean() /
               data["signal_long"].rolling(rolling_window).std()).fillna(0)

total_sharpe = abs(sharpe_short) + abs(sharpe_long)
weight_short = sharpe_short / total_sharpe
weight_long = sharpe_long / total_sharpe

data["combined_signal"] = weight_short * data["signal_short"] + weight_long * data["signal_long"]

data["position"] = data["combined_signal"].clip(-0.01, 0.01) / 0.01

momentum_threshold = 0
data["trading_allowed"] = (data["combined_signal"] > momentum_threshold).astype(int)
data["position_filtered"] = data["position"] * data["trading_allowed"]

data["SMA_200"] = data["Close"].rolling(window=200).mean()
data = data.dropna(subset=["SMA_200"])
data["sma_filter"] = np.where(data["Close"] > data["SMA_200"], 1, 0.5)
data["position_filtered_sma"] = data["position_filtered"] * data["sma_filter"]

cost_rate = 0.001

data["strategy"] = data["position"].shift(1) * data["returns"]
trades = data["position"].diff().abs()
trade_costs = trades * cost_rate
data["net_strategy"] = data["strategy"] - trade_costs

data["strategy_filtered_sma"] = data["position_filtered_sma"].shift(1) * data["returns"]
trades_filtered_sma = data["position_filtered_sma"].diff().abs()
trade_costs_filtered_sma = trades_filtered_sma * cost_rate
data["strategy_filtered_sma_net"] = data["strategy_filtered_sma"] - trade_costs_filtered_sma

data["AAPL_hold"] = (1 + data["returns"]).cumprod()
spy = yf.download("SPY", start="2018-01-01", end="2025-01-01", auto_adjust=True)
spy["returns"] = spy["Close"].pct_change()
spy_cum = (1 + spy["returns"]).cumprod()

combined = pd.DataFrame({
    "Strategy": (1 + data["strategy"]).cumprod(),
    "Strategy + Costs": (1 + data["net_strategy"]).cumprod(),
    "Strategy + Costs + Filters": (1 + data["strategy_filtered_sma_net"]).cumprod(),
    "AAPL Buy-and-Hold": data["AAPL_hold"],
    "SPY Buy-and-Hold": spy_cum
})

combined.plot(figsize=(12,6), logy=True)
plt.title("AAPL Momentum Strategy vs SPY vs Buy and Hold")
plt.ylabel("Cumulative Log Returns")
plt.legend()
plt.show()

def performance_summary(series):
    ret = series.pct_change().dropna()
    annual_return = (1 + ret.mean())**252 - 1
    annual_vol = ret.std() * np.sqrt(252)
    return pd.Series({
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol
    })

print("Performance Summary (Final Iteration):")
print(performance_summary(combined["Strategy + Costs + Filters"]))
