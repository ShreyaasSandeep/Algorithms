import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

API_KEY = "PKOYSXCCALC4OTPQ146W"
API_SECRET = "H8WojE6vXjVzwrW2CKSjXSomIJ5kyVBg5Gf9pS2o"
BASE_URL = "https://paper-api.alpaca.markets/v2"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "JPM", "JNJ", "XOM", "UNH",
    "PG", "NVDA", "HD", "V", "MA", "PFE", "BA", "KO", "PEP", "CAT", "T", "CVX"
]

sector_map = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Communication Services",
    "AMZN": "Consumer Discretionary", "JPM": "Financials", "JNJ": "Healthcare",
    "XOM": "Energy", "UNH": "Healthcare", "PG": "Consumer Staples",
    "NVDA": "Technology", "HD": "Consumer Discretionary", "V": "Financials",
    "MA": "Financials", "PFE": "Healthcare", "BA": "Industrials",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "CAT": "Industrials",
    "T": "Communication Services", "CVX": "Energy"
}

def fetch_alpaca_data_batch(tickers, start, end, timeframe='1D'):
    all_bars = []
    for ticker in tickers:
        barset = api.get_bars(
            symbol=ticker,
            timeframe=timeframe,
            start=pd.to_datetime(start).strftime('%Y-%m-%d'),
            end=pd.to_datetime(end).strftime('%Y-%m-%d'),
            adjustment='all'
        ).df
        if not barset.empty:
            barset['symbol'] = ticker
            all_bars.append(barset)
    barset_all = pd.concat(all_bars)
    barset_all = barset_all.reset_index()
    barset_all['timestamp'] = pd.to_datetime(barset_all['timestamp'])
    return barset_all

def multi_ticker_momentum_alpaca(
    tickers, start="2018-01-01", end="2025-10-11",
    target_vol=0.5, cost_rate=0.001, slippage_rate=0.0005,
    vol_lookback=20, plot=True,
    max_ticker_weight=0.25, max_sector_weight=0.40,
    max_leverage=2.0, drawdown_limit=0.10,
    sector_map=sector_map
):
    all_data = fetch_alpaca_data_batch(tickers, start, end)

    def single_momentum(ticker):
        data = all_data[all_data['symbol'] == ticker].copy()
        data = data[['timestamp','open','high','low','close','volume']]
        data.columns = ["Date","Open","High","Low","Close","Volume"]
        data.set_index("Date", inplace=True)

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
        sharpe_long = (data["signal_long"].rolling(rolling_window).mean() / data["signal_long"].rolling(rolling_window).std()).fillna(0)
        sharpe_short = (data["signal_short"].rolling(rolling_window).mean() / data["signal_short"].rolling(rolling_window).std()).fillna(0)
        total_sharpe = abs(sharpe_long) + abs(sharpe_short)
        total_sharpe = total_sharpe.replace(0, np.nan)
        weight_long = sharpe_long / total_sharpe
        weight_short = sharpe_short / total_sharpe
        data["combined_signal"] = weight_long*data["signal_long"] + weight_short*data["signal_short"] + 0.25*data["RSI_signal"]

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

        data["SMA_200"] = data["Close"].rolling(200).mean()
        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()
        data = data.dropna(subset=["SMA_200","EMA_200"])
        sma_signal = np.where(data["Close"] > data["SMA_200"], 1, 0.5)
        ema_signal = np.where(data["Close"] > data["EMA_200"], 1, 0.5)
        trend_strength = data["combined_signal"].abs()
        trend_strength = (trend_strength - trend_strength.min()) / (trend_strength.max() - trend_strength.min())
        w_ema = trend_strength
        w_sma = 1 - w_ema
        data["weighted_filter"] = w_sma*sma_signal + w_ema*ema_signal
        data["position_filtered"] = data["position_hysteresis"] * data["weighted_filter"]

        realized_vol = data["returns"].rolling(20).std() * np.sqrt(252)
        scaling = (target_vol / realized_vol).clip(0,3)
        data["position_scaled"] = data["position_filtered"] * scaling

        vol_threshold = 0.5
        data["vol_filter"] = np.where(realized_vol < vol_threshold, 1, 0.5)
        data["position_final"] = data["position_scaled"] * data["vol_filter"]

        data["strategy"] = data["position_final"].shift(1) * data["returns"]
        trades = data["position_final"].diff().abs()
        trade_costs = trades*cost_rate
        slippage_costs = trades*slippage_rate
        data["strategy_net"] = data["strategy"] - trade_costs - slippage_costs
        return data["strategy_net"]

    strategy_returns = pd.DataFrame({t: single_momentum(t) for t in tickers})

    rolling_vol = strategy_returns.rolling(vol_lookback).std() * np.sqrt(252)
    rolling_weights = 1 / rolling_vol
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    rolling_weights = rolling_weights.clip(upper=max_ticker_weight)
    for date in rolling_weights.index:
        weights_day = rolling_weights.loc[date]
        sector_sums = {}
        for ticker in weights_day.index:
            sector = sector_map.get(ticker,"Other")
            sector_sums[sector] = sector_sums.get(sector,0) + weights_day[ticker]
        for sector, total_weight in sector_sums.items():
            if total_weight > max_sector_weight:
                scale_factor = max_sector_weight / total_weight
                for ticker in [t for t,s in sector_map.items() if s==sector]:
                    if ticker in rolling_weights.columns:
                        rolling_weights.loc[date,ticker] *= scale_factor
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    portfolio_returns = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)

    portfolio_realized_vol = portfolio_returns.rolling(vol_lookback).std() * np.sqrt(252)
    leverage_factor = (target_vol / portfolio_realized_vol).clip(0, max_leverage)
    portfolio_returns_levered = portfolio_returns * leverage_factor.shift(1)

    cum_portfolio = (1 + portfolio_returns_levered).cumprod()
    cum_max = cum_portfolio.cummax()
    drawdown = 1 - cum_portfolio / cum_max
    drawdown_factor = np.where(drawdown < drawdown_limit, 1, 0)
    portfolio_returns_final = portfolio_returns_levered * drawdown_factor
    portfolio_cum_final = (1 + portfolio_returns_final).cumprod()

    spy_data = fetch_alpaca_data_batch(["SPY"], start, end)
    spy_data = spy_data[spy_data['symbol']=='SPY'][['timestamp','close']].copy()
    spy_data.columns = ["Date","Close"]
    spy_data.set_index("Date", inplace=True)
    spy_data["returns"] = spy_data["Close"].pct_change()
    spy_cum = (1 + spy_data["returns"]).cumprod()

    def performance_summary(series):
        ret = series.pct_change().dropna()
        cumulative = (1+ret).cumprod()
        annual_return = cumulative.iloc[-1]**(252/len(ret))-1
        annual_vol = ret.std()*np.sqrt(252)
        sharpe = annual_return/annual_vol
        max_dd = (1-cumulative/cumulative.cummax()).max()
        win_rate = (ret>0).mean()
        return pd.Series({
            "CAGR": annual_return,
            "Volatility": annual_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Win Rate": win_rate
        })

    summary = performance_summary(portfolio_cum_final)
    bh_data = pd.DataFrame({t: fetch_alpaca_data_batch([t], start, end).set_index('timestamp')['close'].pct_change() for t in tickers}).dropna()
    bh_portfolio_cum = (1+bh_data.mean(axis=1)).cumprod()
    summary_bh = performance_summary(bh_portfolio_cum)
    summary_spy = performance_summary(spy_cum)

    if plot:
        pd.DataFrame({
            "Momentum Portfolio (Levered + DD Stop)": portfolio_cum_final,
            "Equal-Weight BH": bh_portfolio_cum,
            "SPY Buy-and-Hold": spy_cum
        }).plot(figsize=(12,6), logy=True, title="Momentum Portfolio vs BH vs SPY")
        plt.show()

    print("Momentum Portfolio Summary:\n", summary)
    print("\nEqual-Weight Buy-and-Hold Summary:\n", summary_bh)
    print("\nSPY Buy-and-Hold Summary:\n", summary_spy)

    return portfolio_cum_final, summary, rolling_weights, leverage_factor


portfolio_cum, summary, rolling_weights = multi_ticker_momentum_alpaca(
    tickers=tickers,
    start="2018-01-01",
    end="2025-10-16",
    max_ticker_weight=0.25,
    max_sector_weight=0.4,
    sector_map=sector_map,
    plot=True
)
