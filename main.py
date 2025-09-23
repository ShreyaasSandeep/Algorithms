import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

data = yf.download("AAPL", start="2018-01-01", end="2025-01-01", auto_adjust=True)
data["returns"] = data["Close"].pct_change()

data["momentum_long"] = data["Close"].pct_change(20)
data["momentum_short"] = data["Close"].pct_change(5)
vol = data["returns"].rolling(40).std()
data["signal_long"] = data["momentum_long"] / vol
data["signal_short"] = data["momentum_short"] / vol
data["combined_signal"] = 0.7 * data["signal_long"] + 0.3 * data["signal_short"]

data["position"] = data["combined_signal"].clip(-0.5, 0.5) / 0.5

data["strategy"] = data["position"].shift(1) * data["returns"]

cost_rate = 0.001
trades = data["position"].diff().abs()
trade_costs = trades * cost_rate
data["net_strategy"] = data["strategy"] - trade_costs

cumulative = (1 + data["net_strategy"]).cumprod()
rolling_max = cumulative.cummax()
drawdown = cumulative / rolling_max - 1
data["drawdown"] = drawdown

drawdown_limit = -0.2
momentum_threshold = 0
position_filtered = []
trading_allowed = True

for i in range(len(data)):
    if drawdown.iloc[i] < drawdown_limit:
        trading_allowed = False
    elif data["combined_signal"].iloc[i] > momentum_threshold:
        trading_allowed = True

    if trading_allowed:
        position_filtered.append(data["position"].iloc[i])
    else:
        position_filtered.append(0)

data["position_filtered"] = position_filtered

data["strategy_filtered"] = data["position_filtered"].shift(1) * data["returns"]
trades_filtered = data["position_filtered"].diff().abs()
trade_costs_filtered = trades_filtered * cost_rate
data["strategy_filtered_net"] = data["strategy_filtered"] - trade_costs_filtered

spy = yf.download("SPY", start="2018-01-01", end="2025-01-01", auto_adjust=True)
spy["returns"] = spy["Close"].pct_change()
spy_cum = (1 + spy["returns"]).cumprod()

data["AAPL_hold"] = (1 + data["returns"]).cumprod()

combined = (1 + data[["strategy", "net_strategy", "strategy_filtered_net"]]).cumprod()
combined["SPY"] = spy_cum
combined["AAPL_hold"] = data["AAPL_hold"]

combined.plot(figsize=(12,6))
plt.title("AAPL Momentum Strategy vs SPY & Buy-and-Hold")
plt.ylabel("Cumulative Returns")
plt.legend([
    "Strategy",
    "Strategy w/ Costs",
    "Strategy w/ Costs & Filter",
    "SPY",
    "Buy-and-Hold"
])
plt.show()
