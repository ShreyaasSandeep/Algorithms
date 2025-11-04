import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
import time
import threading
from sklearn.linear_model import SGDRegressor



API_KEY = "PKOYSXCCALC4OTPQ146W"
API_SECRET = "H8WojE6vXjVzwrW2CKSjXSomIJ5kyVBg5Gf9pS2o"
BASE_URL = "https://paper-api.alpaca.markets/v2"

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

#List of equities that the portfolio consists of
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "JPM", "JNJ", "XOM", "UNH",
    "PG", "NVDA", "HD", "V", "MA", "PFE", "BA", "KO", "PEP", "CAT", "T", "CVX"
]

#Dictionary mapping equities to the industry
sector_map = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Communication Services",
    "AMZN": "Consumer Discretionary", "JPM": "Financials", "JNJ": "Healthcare",
    "XOM": "Energy", "UNH": "Healthcare", "PG": "Consumer Staples",
    "NVDA": "Technology", "HD": "Consumer Discretionary", "V": "Financials",
    "MA": "Financials", "PFE": "Healthcare", "BA": "Industrials",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "CAT": "Industrials",
    "T": "Communication Services", "CVX": "Energy"
}

# Collects OHLC data for single ticker in timeframe
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

#Collects all data for all tickers and caches it
def fetch_alpaca_data_batch(tickers, start, end, timeframe='1D', max_workers=8, cache_dir="cache"):
    #Ensures cache directory is created
    os.makedirs(cache_dir, exist_ok=True)
    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')

    # Prevents simultaneous writes to the same cache
    cache_lock = threading.Lock()

    def task(ticker):
        #Creates unique cache file for each ticker
        cache_file = f"{cache_dir}/{ticker}_{start_str}_{end_str}_{timeframe}.pkl"
        #Path if cache file exists already for ticker
        if os.path.exists(cache_file):
            try:
                obj = pickle.load(open(cache_file, "rb"))
                if not obj.empty:
                    return obj
            except Exception as e:
                print(f"[{ticker}] failed to read cache, will refetch: {e}")

        # Path if cache file doesn't exist or there is an error
        bars = fetch_one(ticker, start_str, end_str, timeframe=timeframe)
        if bars is None or bars.empty:
            return pd.DataFrame()

        # Ensures symbol column in present in dataframe
        bars = bars.copy()
        if 'symbol' not in bars.columns:
            bars['symbol'] = ticker

        try:
            with cache_lock:
                if not os.path.exists(cache_file):
                    pickle.dump(bars, open(cache_file, "wb"))
        except Exception as e:
            print(f"[{ticker}] error writing cache: {e}")

        return bars

    all_data = []
    errors = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(task, t): t for t in tickers}
        for fut in as_completed(future_to_ticker):
            t = future_to_ticker[fut]
            try:
                res = fut.result()
                if isinstance(res, pd.DataFrame) and not res.empty:
                    all_data.append(res)
                else:
                    print(f"[{t}] no data returned (empty DataFrame).")
            except Exception as e:
                print(f"[{t}] failed after retries: {e}")
                errors[t] = str(e)

    if not all_data:
        raise ValueError(f"No data fetched from Alpaca. Errors: {errors}")

    df = pd.concat(all_data).reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


#Uses signals to direct trades
def compute_signals(all_data, target_vol=0.5, cost_rate=0.001, slippage_rate=0.0005):
    all_data = all_data.copy()
    all_data = all_data.sort_values(['symbol', 'timestamp'])

    # Returns and momentum
    all_data['returns'] = all_data.groupby('symbol')['close'].pct_change()
    all_data['momentum_long'] = all_data.groupby('symbol')['close'].pct_change(20)
    all_data['momentum_short'] = all_data.groupby('symbol')['close'].pct_change(5)

    # Volatility ratio
    vol_short = all_data.groupby('symbol')['returns'].transform(lambda x: x.rolling(5, min_periods=1).std())
    vol_long = all_data.groupby('symbol')['returns'].transform(lambda x: x.rolling(20, min_periods=1).std())
    all_data['volatility_ratio'] = (vol_short / (vol_long + 1e-8)).fillna(1)

    # Normalized momentum signals
    all_data['signal_long'] = all_data.groupby('symbol')['momentum_long'].transform(
        lambda x: ((x - x.mean()) / (x.std() + 1e-8)).ewm(span=60).mean()
    )
    all_data['signal_short'] = all_data.groupby('symbol')['momentum_short'].transform(
        lambda x: ((x - x.mean()) / (x.std() + 1e-8)).ewm(span=20).mean()
    )

    # RSI calculation
    delta = all_data.groupby('symbol')['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False).mean()
    roll_down = down.ewm(span=14, adjust=False).mean()
    RS = roll_up / (roll_down + 1e-8)
    all_data['RSI_signal'] = (100 - (100 / (1 + RS)) - 50) / 50

    # SMA and EMA
    all_data['SMA_200'] = all_data.groupby('symbol')['close'].transform(lambda x: x.rolling(200, min_periods=1).mean())
    all_data['EMA_200'] = all_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    sma_signal = np.where(all_data['close'] > all_data['SMA_200'], 1, 0.5)
    ema_signal = np.where(all_data['close'] > all_data['EMA_200'], 1, 0.5)

    trend_strength = (all_data['signal_long'].abs() - all_data['signal_long'].abs().min()) / \
                     (all_data['signal_long'].abs().max() - all_data['signal_long'].abs().min() + 1e-8)
    all_data['weighted_filter'] = trend_strength * ema_signal + (1 - trend_strength) * sma_signal

    # Bollinger Bands
    bb_lookback = 20
    bb_k = 2
    all_data['SMA_BB'] = all_data.groupby('symbol')['close'].transform(lambda x: x.rolling(bb_lookback).mean())
    all_data['STD_BB'] = all_data.groupby('symbol')['close'].transform(lambda x: x.rolling(bb_lookback).std())
    all_data['BB_upper'] = all_data['SMA_BB'] + bb_k * all_data['STD_BB']
    all_data['BB_lower'] = all_data['SMA_BB'] - bb_k * all_data['STD_BB']
    all_data['BB_zscore'] = (all_data['close'] - all_data['SMA_BB']) / (all_data['STD_BB'] + 1e-8)

    #Volume spike features
    eps = 1e-8
    vol_lookback_mean = 20
    vol_lookback_std = 20

    all_data['vol_mean'] = all_data.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(vol_lookback_mean, min_periods=1).mean()
    )
    all_data['vol_std'] = all_data.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(vol_lookback_std, min_periods=1).std().fillna(0)
    )
    all_data['volume_spike_ratio_raw'] = all_data['volume'] / (all_data['vol_mean'] + eps)
    all_data['volume_spike_log'] = np.log1p(all_data['volume_spike_ratio_raw'])
    all_data['volume_zscore'] = (all_data['volume'] - all_data['vol_mean']) / (all_data['vol_std'] + eps)

    clip_upper, clip_lower = 5.0, -5.0
    all_data['volume_spike_log'] = all_data['volume_spike_log'].clip(lower=0.0, upper=clip_upper)
    all_data['volume_zscore'] = all_data['volume_zscore'].clip(lower=clip_lower, upper=clip_upper)

    all_data['volume_spike_combined'] = 0.6 * (all_data['volume_spike_log'] / (all_data['volume_spike_log'].std() + eps)) + \
                                       0.4 * (all_data['volume_zscore'] / (all_data['volume_zscore'].std() + eps))
    all_data['volume_spike_rank'] = all_data.groupby('timestamp')['volume_spike_combined'].transform(
        lambda x: x.rank(pct=True)
    )

    all_data['next_open'] = all_data.groupby('symbol')['open'].shift(-1)
    all_data['next_open_return'] = all_data['next_open'] / all_data['close'] - 1

    features = [
        'signal_long', 'signal_short', 'RSI_signal',
        'weighted_filter', 'BB_zscore', 'volatility_ratio',
        'volume_spike_rank'
    ]
    all_data = all_data.dropna(subset=features + ['next_open_return']).copy()

    # Rolling SGD model
    def rolling_sgd_predictions(df, features, target='next_open_return',
                                window=252, alpha=0.01, retrain_interval=5):
        df = df.copy()
        df['combined_signal'] = np.nan

        for sym in df['symbol'].unique():
            sym_df = df[df['symbol'] == sym].copy()
            X = sym_df[features].values
            y = sym_df[target].values
            preds = []

            model = SGDRegressor(
                alpha=alpha,
                penalty='l2',
                max_iter=1,
                tol=None,
                warm_start=True,
                learning_rate='constant',
                eta0=0.01,
                random_state=42
            )

            for i in range(window, len(sym_df)):
                if (i - window) % retrain_interval == 0:
                    X_train = X[i - window:i]
                    y_train = y[i - window:i]
                    model.partial_fit(X_train, y_train)
                preds.append(model.predict(X[[i]])[0])

            df.loc[sym_df.index[window:], 'combined_signal'] = preds

        return df

    all_data = rolling_sgd_predictions(all_data, features)
    all_data['combined_signal_for_execution'] = all_data.groupby('symbol')['combined_signal'].shift(1)

    # Position hysteresis
    def apply_hysteresis(signal, upper=-0.25, lower=-1):
        pos = np.zeros(len(signal))
        for i in range(len(signal)):
            if i == 0:
                pos[i] = 0
            else:
                if signal[i] > upper:
                    pos[i] = 1
                elif signal[i] < lower:
                    pos[i] = -1
                else:
                    pos[i] = pos[i - 1]
        return pos

    all_data['position_hysteresis'] = all_data.groupby('symbol')['combined_signal_for_execution'] \
        .transform(lambda x: apply_hysteresis(x.values))
    all_data['position_filtered'] = all_data['position_hysteresis'] * all_data['weighted_filter']

    # Risk scaling and final position
    realised_vol = all_data.groupby('symbol')['returns'].transform(lambda x: x.rolling(20, min_periods=1).std() * np.sqrt(252))
    scaling = (target_vol / realised_vol).clip(0, 3)
    all_data['position_final'] = all_data['position_filtered'] * scaling

    # Strategy returns and costs
    all_data['strategy'] = all_data['position_final'].shift(1) * (all_data['next_open'] / all_data['close'] - 1)
    all_data['position_change'] = all_data.groupby('symbol')['position_final'].diff().abs()
    all_data['dynamic_cost'] = cost_rate + slippage_rate
    all_data['strategy_net'] = all_data['strategy'] - all_data['position_change'] * all_data['dynamic_cost']

    return all_data






def construct_portfolio(all_data, tickers, sector_map,
                        target_vol=0.5, vol_lookback=20,
                        max_ticker_weight=0.25, max_sector_weight=0.1,
                        max_leverage=2.0,
                        crisis_drawdown_threshold=-0.10,
                        crisis_leverage_multiplier=0.2):

    #Creates a DataFrame based on data given
    strategy_returns = all_data.pivot(index='timestamp', columns='symbol', values='strategy_net').fillna(0)

    #Calculates rolling volatility, adjusts weights inversely to volatility, and then normalises them
    rolling_vol = strategy_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)
    rolling_weights = 1 / rolling_vol
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0).clip(upper=max_ticker_weight)

    #Adjusts weights of tickers to fit with max sector constraints
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

    #Normalises weights
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    #Collects data for SPY Buy-and-Hold benchmark
    spy_data = fetch_alpaca_data_batch(["SPY"], all_data['timestamp'].min(), all_data['timestamp'].max())
    spy_data = spy_data[spy_data['symbol']=='SPY'].set_index('timestamp')['close']
    spy_cummax = spy_data.cummax()

    #Detects crisis periods and adjusts weights to minimise exposure, before renormalising
    crisis_series = (spy_data / spy_cummax - 1 < crisis_drawdown_threshold).astype(int).reindex(rolling_weights.index).fillna(0)
    crisis_mask = crisis_series == 1
    rolling_weights.loc[crisis_mask] *= crisis_leverage_multiplier
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    #Generates overall portfolio returns, measures volatility and scales to meet target, and applies leverage based on if it's a crisis or not
    portfolio_returns = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)
    portfolio_realized_vol = portfolio_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)
    leverage_factor = (target_vol / portfolio_realized_vol).clip(0, max_leverage)
    leverage_factor *= np.where(crisis_series==1, crisis_leverage_multiplier, 1.0)
    portfolio_returns_levered = portfolio_returns * leverage_factor.shift(1)

    #Calculates performance over last 20 days and if weak, exposure is reduced
    rolling_20d_ret = portfolio_returns.rolling(20).sum()
    reactive_leverage = np.where(rolling_20d_ret < -0.05, 0.5, 1.0)
    portfolio_returns_levered *= reactive_leverage

    portfolio_cum_final = (1 + portfolio_returns_levered).cumprod()

    return portfolio_cum_final, rolling_weights, leverage_factor

def multi_ticker_momentum_alpaca(tickers, start="2018-01-01", end="2025-11-01",
                                 sector_map=sector_map, plot=True,
                                 target_vol=0.5, cost_rate=0.001, slippage_rate=0.0005,
                                 vol_lookback=20, max_ticker_weight=0.25,
                                 max_sector_weight=0.1, max_leverage=2.0,
                                 crisis_drawdown_threshold=-0.10,
                                 crisis_leverage_multiplier=0.2):

    #Generate a DataFrame with historical data and all features of the strategy
    all_data = fetch_alpaca_data_batch(tickers, start, end)
    all_data = compute_signals(all_data, target_vol=target_vol, cost_rate=cost_rate, slippage_rate=slippage_rate)

    #Uses the DataFrame and constructs a portfolio
    portfolio_cum, rolling_weights, leverage_factor = construct_portfolio(
        all_data, tickers, sector_map, target_vol=target_vol, vol_lookback=vol_lookback,
        max_ticker_weight=max_ticker_weight, max_sector_weight=max_sector_weight,
        max_leverage=max_leverage, crisis_drawdown_threshold=crisis_drawdown_threshold,
        crisis_leverage_multiplier=crisis_leverage_multiplier
    )
    #Generated data for SPY benchmark
    spy_data = fetch_alpaca_data_batch(["SPY"], start, end)
    spy_data = spy_data[spy_data['symbol'] == 'SPY'][['timestamp', 'close']].copy()
    spy_data.columns = ["Date", "Close"]
    spy_data.set_index("Date", inplace=True)
    spy_data["returns"] = spy_data["Close"].pct_change()
    spy_cum = (1 + spy_data["returns"]).cumprod()

    #Generated data for Buy-and-Hold benchmark
    bh_data = pd.DataFrame({t: fetch_alpaca_data_batch([t], start, end)
                           .set_index('timestamp')['close'].pct_change() for t in tickers}).dropna()
    bh_portfolio_cum = (1 + bh_data.mean(axis=1)).cumprod()

    def performance_summary(series, benchmark=None):
        #Calculates the metrics of performance referred to in the summary
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

        #Displays summary
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
    #Creates summary for all 3 to compare
    summary = performance_summary(portfolio_cum, spy_cum)
    summary_bh = performance_summary(bh_portfolio_cum)
    summary_spy = performance_summary(spy_cum)

    #Plots returns for all 3
    if plot:
        df_plot = pd.DataFrame({
            "Momentum Portfolio": portfolio_cum,
            "Equal-Weight BH": bh_portfolio_cum,
            "SPY Buy-and-Hold": spy_cum
        })
        ax = df_plot.plot(figsize=(12, 6), logy=True, title="Momentum Portfolio vs BH vs SPY")

        plt.show()

    print("Momentum Portfolio Summary:\n", summary)
    print("\nEqual-Weight Buy-and-Hold Summary:\n", summary_bh)
    print("\nSPY Buy-and-Hold Summary:\n", summary_spy)

    return portfolio_cum, summary, rolling_weights, leverage_factor, spy_data

portfolio_cum, summary, rolling_weights, leverage_factor = multi_ticker_momentum_alpaca(
    tickers=tickers,
    start="2018-01-01",
    end="2025-11-02",
    max_ticker_weight=0.25,
    max_sector_weight=0.1,
    sector_map=sector_map,
    plot=True
)


