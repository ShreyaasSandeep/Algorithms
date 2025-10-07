import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def multi_ticker_momentum_dynamic(tickers, start="2018-01-01", end="2025-01-01",
                                  target_vol=0.3, cost_rate=0.001, slippage_rate=0.0005,
                                  vol_lookback=20, plot=True):
    def single_momentum(ticker):
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data["returns"] = data["Close"].pct_change()
        data["momentum_long"] = data["Close"].pct_change(20)
        data["momentum_short"] = data["Close"].pct_change(5)

        data["signal_long"] = (data["momentum_long"] - data["momentum_long"].mean()) / data["momentum_long"].std()
        data["signal_short"] = (data["momentum_short"] - data["momentum_short"].mean()) / data["momentum_short"].std()

        data["signal_long"] = data["signal_long"].ewm(span=60).mean()
        data["signal_short"] = data["signal_short"].ewm(span=20).mean()

        z_clip = 3.0
        data["signal_long"] = data["signal_long"].clip(-z_clip, z_clip)
        data["signal_short"] = data["signal_short"].clip(-z_clip, z_clip)

        window_rsi = 14
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(span=window_rsi, adjust=False).mean()
        roll_down = down.ewm(span=window_rsi, adjust=False).mean()
        RS = roll_up / roll_down
        data['RSI'] = 100 - (100 / (1 + RS))
        data['RSI_signal'] = (data['RSI'] - 50) / 50

        rolling_window = 252
        sharpe_long = (data["signal_long"].rolling(rolling_window).mean() /
                       data["signal_long"].rolling(rolling_window).std()).fillna(0)
        sharpe_short = (data["signal_short"].rolling(rolling_window).mean() /
                        data["signal_short"].rolling(rolling_window).std()).fillna(0)

        total_sharpe = abs(sharpe_long) + abs(sharpe_short)
        total_sharpe = total_sharpe.replace(0, np.nan)

        weight_long = sharpe_long / total_sharpe
        weight_short = sharpe_short / total_sharpe

        data["combined_signal"] = (
                weight_long * data["signal_long"] +
                weight_short * data["signal_short"] +
                0.25 * data["RSI_signal"]
        )

        upper_threshold = -0.25
        lower_threshold = -0.5
        position = 0
        positions_hysteresis = []
        for sig in data["combined_signal"]:
            if sig > upper_threshold:
                position = 1
            elif sig < lower_threshold:
                position = 0.5
            positions_hysteresis.append(position)

        data["position_hysteresis"] = np.array(positions_hysteresis)

        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
        data = data.dropna(subset=["SMA_200", "EMA_200"])

        sma_signal = np.where(data["Close"] > data["SMA_200"], 1, 0.5)
        ema_signal = np.where(data["Close"] > data["EMA_200"], 1, 0.5)

        trend_strength = data["combined_signal"].abs()
        trend_strength = (trend_strength - trend_strength.min()) / (trend_strength.max() - trend_strength.min())
        w_ema = trend_strength
        w_sma = 1 - w_ema

        data["weighted_filter"] = w_sma * sma_signal + w_ema * ema_signal
        data["position_filtered"] = data["position_hysteresis"] * data["weighted_filter"]

        realized_vol = data["returns"].rolling(20).std() * np.sqrt(252)
        scaling = (target_vol / realized_vol).clip(0, 3)
        data["position_scaled"] = data["position_filtered"] * scaling

        vol_threshold = 0.3
        data["vol_filter"] = np.where(realized_vol < vol_threshold, 1, 0.5)
        data["position_final"] = data["position_scaled"] * data["vol_filter"]

        data["strategy"] = data["position_final"].shift(1) * data["returns"]
        trades = data["position_final"].diff().abs()
        trade_costs = trades * cost_rate
        slippage_costs = trades * slippage_rate
        data["strategy_net"] = data["strategy"] - trade_costs - slippage_costs

        return data["strategy_net"]

    strategy_returns = pd.DataFrame({t: single_momentum(t) for t in tickers})

    rolling_vol = strategy_returns.rolling(vol_lookback).std() * np.sqrt(252)
    rolling_weights = 1 / rolling_vol
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    portfolio_returns = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)
    portfolio_cum = (1 + portfolio_returns).cumprod()

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True)
    spy["returns"] = spy["Close"].pct_change()
    spy_cum = (1 + spy["returns"]).cumprod()

    def performance_summary(series):
        ret = series.pct_change().dropna()
        cumulative = (1 + ret).cumprod()
        annual_return = cumulative.iloc[-1] ** (252 / len(ret)) - 1
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        max_dd = (1 - cumulative / cumulative.cummax()).max()
        win_rate = (ret > 0).mean()
        return pd.Series({
            "CAGR": annual_return,
            "Volatility": annual_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Win Rate": win_rate
        })

    summary = performance_summary(portfolio_cum)

    bh_data = {}
    for t in tickers:
        temp = yf.download(t, start=start, end=end, auto_adjust=True)
        temp["returns"] = temp["Close"].pct_change()
        bh_data[t] = temp["returns"]
    bh_returns = pd.DataFrame(bh_data).dropna()
    bh_portfolio_returns = bh_returns.mean(axis=1)
    bh_portfolio_cum = (1 + bh_portfolio_returns).cumprod()

    if plot:
        pd.DataFrame({
            "Momentum Portfolio": portfolio_cum,
            "Equal-Weight BH": bh_portfolio_cum,
            "SPY Buy-and-Hold": spy_cum
        }).plot(figsize=(12, 6), logy=True,
                title="Momentum Portfolio Strategy vs Equal-Weight Buy and Hold vs SPY")
        plt.show()

    summary_bh = performance_summary(bh_portfolio_cum)
    summary_spy = performance_summary(spy_cum)

    print("Momentum Portfolio Summary:\n", summary)
    print("\nEqual-Weight Buy-and-Hold Summary:\n", summary_bh)
    print("\nSPY Buy-and-Hold Summary:\n", summary_spy)

    return portfolio_cum, summary, rolling_weights


tickers = ["AAPL", "GOOG", "IBM", "MSFT", "AMZN"]
portfolio_cum, summary, rolling_weights = multi_ticker_momentum_dynamic(tickers)
print(summary)
