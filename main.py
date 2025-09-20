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

data["combined_signal"] = 0.5 * data["signal_long"] + 0.5 * data["signal_short"]
data["position"] = data["combined_signal"].clip(-0.2, 0.2) / 0.2
data["strategy"] = data["position"].shift(1) * data["returns"]

(1 + data[["returns", "strategy"]]).cumprod().plot(figsize=(10,6))
plt.title("Multi-Horizon Momentum Strategy vs Buy & Hold (AAPL)")
plt.show()
