import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, pickle


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

def fetch_one(symbol, start, end, timeframe='1D'):
    bars = api.get_bars(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment='all'
    ).df
    if not bars.empty:
        bars['symbol'] = symbol
    return bars



def fetch_alpaca_data_batch(tickers, start, end, timeframe='1D', max_workers=8, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    start = pd.to_datetime(start).strftime('%Y-%m-%d')
    end = pd.to_datetime(end).strftime('%Y-%m-%d')

    all_data = []
    for t in tickers:
        cache_file = f"{cache_dir}/{t}_{start}_{end}_{timeframe}.pkl"
        if os.path.exists(cache_file):
            bars = pickle.load(open(cache_file, "rb"))
        else:
            try:
                bars = fetch_one(t, start, end, timeframe)
                if not bars.empty:
                    pickle.dump(bars, open(cache_file, "wb"))
            except Exception as e:
                print(f"Error fetching {t}: {e}")
                bars = pd.DataFrame()

        if not bars.empty:
            all_data.append(bars)

    if not all_data:
        raise ValueError("No data fetched from Alpaca.")

    df = pd.concat(all_data).reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def compute_signals(all_data, target_vol=0.5, cost_rate=0.001, slippage_rate=0.0005):
    all_data = all_data.copy()
    all_data = all_data.sort_values(['symbol', 'timestamp'])

    all_data['returns'] = all_data.groupby('symbol')['close'].pct_change()
    all_data['momentum_long'] = all_data.groupby('symbol')['close'].pct_change(20)
    all_data['momentum_short'] = all_data.groupby('symbol')['close'].pct_change(5)

    all_data['signal_long'] = all_data.groupby('symbol')['momentum_long'].transform(
        lambda x: (x - x.mean()) / x.std()
    ).ewm(span=60).mean()

    all_data['signal_short'] = all_data.groupby('symbol')['momentum_short'].transform(
        lambda x: (x - x.mean()) / x.std()
    ).ewm(span=20).mean()

    z_clip = 3.0
    all_data['signal_long'] = all_data['signal_long'].clip(-z_clip, z_clip)
    all_data['signal_short'] = all_data['signal_short'].clip(-z_clip, z_clip)

    delta = all_data.groupby('symbol')['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    RS = roll_up / roll_down
    all_data['RSI_signal'] = (100 - (100 / (1 + RS)) - 50) / 50

    rolling_window = 252
    sharpe_long = all_data.groupby('symbol')['signal_long'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean() / x.rolling(rolling_window, min_periods=1).std()
    ).fillna(0)

    sharpe_short = all_data.groupby('symbol')['signal_short'].transform(
        lambda x: x.rolling(rolling_window, min_periods=1).mean() / x.rolling(rolling_window, min_periods=1).std()
    ).fillna(0)

    eps = 1e-8
    total_sharpe = (abs(sharpe_long) + abs(sharpe_short)).fillna(0)
    total_sharpe = total_sharpe.where(total_sharpe > eps, eps)
    weight_long = sharpe_long / total_sharpe
    weight_short = sharpe_short / total_sharpe
    weight_long = weight_long.replace([np.inf, -np.inf], 0).fillna(1)
    weight_short = weight_short.replace([np.inf, -np.inf], 0).fillna(1)

    all_data['combined_signal'] = weight_long * all_data['signal_long'] + \
                                  weight_short * all_data['signal_short'] + \
                                  0.25 * all_data['RSI_signal']

    def apply_hysteresis(signal, upper=-0.25, lower=-1):
        pos = np.zeros(len(signal))
        for i in range(len(signal)):
            if i == 0:
                pos[i] = 0
            else:
                if signal[i] > upper:
                    pos[i] = 1
                elif signal[i] < lower:
                    pos[i] = 0.5
                else:
                    pos[i] = pos[i - 1]
        return pos

    all_data['position_hysteresis'] = all_data.groupby('symbol')['combined_signal'].transform(
        lambda x: apply_hysteresis(x.values)
    )

    all_data['SMA_200'] = all_data.groupby('symbol')['close'].transform(lambda x: x.rolling(200, min_periods=1).mean())
    all_data['EMA_200'] = all_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    sma_signal = np.where(all_data['close'] > all_data['SMA_200'], 1, 0.5)
    ema_signal = np.where(all_data['close'] > all_data['EMA_200'], 1, 0.5)

    trend_strength = (all_data['combined_signal'].abs() - all_data['combined_signal'].abs().min()) / \
                     (all_data['combined_signal'].abs().max() - all_data['combined_signal'].abs().min())

    w_ema = trend_strength
    w_sma = 1 - w_ema

    all_data['weighted_filter'] = w_sma * sma_signal + w_ema * ema_signal
    all_data['position_filtered'] = all_data['position_hysteresis'] * all_data['weighted_filter']

    realized_vol = all_data.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(20, min_periods=1).std() * np.sqrt(252))
    scaling = (target_vol / realized_vol).clip(0, 3)
    all_data['position_final'] = all_data['position_filtered'] * scaling

    trades = all_data.groupby('symbol')['position_final'].diff().abs()
    all_data['next_open'] = all_data.groupby('symbol')['open'].shift(-1)
    all_data['strategy'] = all_data['position_final'].shift(1) * (all_data['next_open'] / all_data['close'] - 1)

    spread_factor = 0.0002
    vol_factor = 0.1 * all_data['returns'].rolling(5, min_periods=1).std()
    volume_penalty = np.random.normal(0, 0.0001, len(all_data))
    all_data['dynamic_cost'] = cost_rate + slippage_rate + spread_factor + vol_factor + volume_penalty

    all_data['strategy_net'] = all_data['strategy'] - trades * all_data['dynamic_cost'].clip(lower=0)

    return all_data


def construct_portfolio(all_data, tickers, sector_map,
                                  target_vol=0.5, vol_lookback=20,
                                  max_ticker_weight=0.25, max_sector_weight=0.4,
                                  max_leverage=2.0):
    strategy_returns = all_data.pivot(index='timestamp', columns='symbol', values='strategy_net')
    strategy_returns = strategy_returns.fillna(0)

    rolling_vol = strategy_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)

    rolling_weights = 1 / rolling_vol
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)
    rolling_weights = rolling_weights.clip(upper=max_ticker_weight)

    for date in rolling_weights.index:
        weights_day = rolling_weights.loc[date]
        sector_sums = {}
        for ticker in weights_day.index:
            sector = sector_map.get(ticker, "Other")
            sector_sums[sector] = sector_sums.get(sector, 0) + weights_day[ticker]
        for sector, total_weight in sector_sums.items():
            if total_weight > max_sector_weight:
                scale_factor = max_sector_weight / total_weight
                for ticker in [t for t, s in sector_map.items() if s == sector]:
                    if ticker in rolling_weights.columns:
                        rolling_weights.loc[date, ticker] *= scale_factor

    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    portfolio_returns = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)

    portfolio_realized_vol = portfolio_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)
    leverage_factor = (target_vol / portfolio_realized_vol).clip(0, max_leverage)
    portfolio_returns_levered = portfolio_returns * leverage_factor.shift(1)

    rolling_20d_ret = portfolio_returns.rolling(20).sum()
    reactive_leverage = np.where(rolling_20d_ret < -0.05, 0.5, 1.0)
    portfolio_returns_levered *= reactive_leverage

    portfolio_cum_final = (1 + portfolio_returns_levered).cumprod()

    return portfolio_cum_final, rolling_weights, leverage_factor


def multi_ticker_momentum_alpaca(tickers, start="2018-01-01", end="2025-10-11",
                                 sector_map=sector_map, plot=True,
                                 target_vol=0.5, cost_rate=0.001, slippage_rate=0.0005,
                                 vol_lookback=20, max_ticker_weight=0.25,
                                 max_sector_weight=0.4, max_leverage=2.0,
                                 ):
    all_data = fetch_alpaca_data_batch(tickers, start, end)

    all_data = compute_signals(all_data, target_vol=target_vol,
                               cost_rate=cost_rate, slippage_rate=slippage_rate)

    portfolio_cum, rolling_weights, leverage_factor = construct_portfolio(
        all_data, tickers, sector_map, target_vol=target_vol, vol_lookback=vol_lookback,
        max_ticker_weight=max_ticker_weight, max_sector_weight=max_sector_weight,
        max_leverage=max_leverage
    )

    spy_data = fetch_alpaca_data_batch(["SPY"], start, end)
    spy_data = spy_data[spy_data['symbol'] == 'SPY'][['timestamp', 'close']].copy()
    spy_data.columns = ["Date", "Close"]
    spy_data.set_index("Date", inplace=True)
    spy_data["returns"] = spy_data["Close"].pct_change()
    spy_cum = (1 + spy_data["returns"]).cumprod()

    bh_data = pd.DataFrame({t: fetch_alpaca_data_batch([t], start, end)
                           .set_index('timestamp')['close'].pct_change() for t in tickers}).dropna()
    bh_portfolio_cum = (1 + bh_data.mean(axis=1)).cumprod()

    def performance_summary(series, benchmark=None):
        ret = series.pct_change().dropna()
        cumulative = (1 + ret).cumprod()

        annual_return = cumulative.iloc[-1] ** (252 / len(ret)) - 1
        annual_vol = ret.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol

        downside = ret[ret < 0]
        downside_vol = downside.std() * np.sqrt(252)
        sortino = annual_return / downside_vol if downside_vol != 0 else np.nan

        max_dd = (1 - cumulative / cumulative.cummax()).max()
        calmar = annual_return / max_dd if max_dd != 0 else np.nan
        win_rate = (ret > 0).mean()

        beta = np.nan
        if benchmark is not None:
            bench_ret = benchmark.pct_change().dropna()
            aligned = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
            aligned.columns = ["portfolio", "benchmark"]
            if len(aligned) > 5:
                cov = np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1]
                var = np.var(aligned["benchmark"])
                beta = cov / var if var != 0 else np.nan

        return pd.Series({
            "CAGR": annual_return,
            "Volatility": annual_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "Beta vs SPY": beta,
            "Max Drawdown": max_dd,
            "Win Rate": win_rate
        })

    summary = performance_summary(portfolio_cum, spy_cum)
    summary_bh = performance_summary(bh_portfolio_cum)
    summary_spy = performance_summary(spy_cum)

    if plot:
        pd.DataFrame({
            "Momentum Portfolio": portfolio_cum,
            "Equal-Weight BH": bh_portfolio_cum,
            "SPY Buy-and-Hold": spy_cum
        }).plot(figsize=(12, 6), logy=True, title="Momentum Portfolio vs BH vs SPY")
        plt.show()

    print("Momentum Portfolio Summary:\n", summary)
    print("\nEqual-Weight Buy-and-Hold Summary:\n", summary_bh)
    print("\nSPY Buy-and-Hold Summary:\n", summary_spy)

    return portfolio_cum, summary, rolling_weights, leverage_factor

def robustness_report(portfolio_cum, title="Momentum Portfolio"):
    portfolio_ret = portfolio_cum.pct_change().dropna()

    rolling_window = 126
    rolling_sharpe = portfolio_ret.rolling(rolling_window).mean() / portfolio_ret.rolling(rolling_window).std() * np.sqrt(252)
    plt.figure(figsize=(12,4))
    rolling_sharpe.plot(color='blue')
    plt.title(f"{title} - Rolling 6-Month Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.grid(True)
    plt.show()

    monthly_ret = portfolio_ret.resample('ME').apply(lambda x: (1+x).prod()-1)
    monthly_ret.plot(kind='bar', figsize=(14,4), color='skyblue')
    plt.title(f"{title} - Monthly Returns")
    plt.xlabel("Month")
    plt.ylabel("Return")
    plt.show()

    cum_max = portfolio_cum.cummax()
    drawdown = 1 - portfolio_cum / cum_max
    plt.figure(figsize=(12,4))
    drawdown.plot(color='red')
    plt.title(f"{title} - Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()
    print(f"Max Drawdown: {drawdown.max():.2%}")

    VaR_5 = np.percentile(portfolio_ret, 5)
    VaR_1 = np.percentile(portfolio_ret, 1)
    CVaR_5 = portfolio_ret[portfolio_ret <= VaR_5].mean()
    CVaR_1 = portfolio_ret[portfolio_ret <= VaR_1].mean()
    print(f"5% VaR: {VaR_5:.2%}, 5% CVaR: {CVaR_5:.2%}")
    print(f"1% VaR: {VaR_1:.2%}, 1% CVaR: {CVaR_1:.2%}")

    wins = portfolio_ret[portfolio_ret > 0]
    losses = portfolio_ret[portfolio_ret < 0]
    print(f"Win rate: {len(wins)/len(portfolio_ret):.2%}")
    print(f"Average win: {wins.mean():.2%}, Average loss: {losses.mean():.2%}")

portfolio_cum, summary, rolling_weights, leverage_factor = multi_ticker_momentum_alpaca(
    tickers=tickers,
    start="2018-01-01",
    end="2025-10-24",
    max_ticker_weight=0.25,
    max_sector_weight=0.1,
    sector_map=sector_map,
    plot=True
)

robustness_report(portfolio_cum, title="Momentum Portfolio")

