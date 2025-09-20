import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

data = yf.download("AAPL", start="2018-01-01", end="2025-01-01", auto_adjust=True)

data["returns"] = data["Close"].pct_change()
data["momentum"] = data["Close"].pct_change(20)
#data["position"] = np.where(data["momentum"] > -0.01, 1, 0)
data["position"] = data["momentum"].clip(-0.01, 0.01) / 0.01
data["strategy"] = data["position"].shift(1) * data["returns"]

(1 + data[["returns", "strategy"]]).cumprod().plot(figsize=(10,6))
plt.title("Momentum Strategy vs Buy & Hold (AAPL)")
plt.show()
