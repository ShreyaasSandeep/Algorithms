import pandas as pd
import numpy as np
from data_fetch import fetch_alpaca_data_batch

def get_spy_drawdown(start_date, end_date, rolling_weights_index, crisis_drawdown_threshold=-0.05):
        """Optimized function to get crisis indicator"""
        
        # Cache implementation
        cache_key = f"spy_{start_date}_{end_date}"
        if not hasattr(get_spy_drawdown, "cache"):
            get_spy_drawdown.cache = {}
        
        if cache_key in get_spy_drawdown.cache:
            crisis = get_spy_drawdown.cache[cache_key]
        else:
            try:
                spy_data = fetch_alpaca_data_batch(["SPY"], start_date, end_date)
                spy = spy_data[spy_data['symbol'] == "SPY"].set_index('timestamp')['close']
                
                # Fast drawdown calculation
                spy_cummax = spy.cummax()
                drawdown = spy / spy_cummax - 1
                crisis = (drawdown < crisis_drawdown_threshold).astype(int)
                
                # Store in cache
                get_spy_drawdown.cache[cache_key] = crisis
                
            except Exception as e:
                print(f"Warning: Could not fetch SPY data: {e}")
                crisis = pd.Series(0, index=pd.date_range(start_date, end_date, freq='D'))
                get_spy_drawdown.cache[cache_key] = crisis
        
        # Align with rolling_weights index efficiently
        if not crisis.index.equals(rolling_weights_index):
            crisis = crisis.reindex(rolling_weights_index, method='ffill').fillna(0)
        
        return crisis, crisis == 1

def fast_rolling_product(returns, window):
    """Fast rolling product using numpy cumprod"""
    returns_np = returns.values
        
    if len(returns_np) < window:
        return pd.Series(np.zeros(len(returns)), index=returns.index)
        
    # Add small epsilon to avoid issues with zeros
    returns_np = np.where(returns_np == 0, 1e-10, returns_np)
        
    # Calculate rolling product
    growth = 1 + returns_np
    cumprod = np.cumprod(growth)
        
    # Create shifted cumprod (window days behind)
    cumprod_shifted = np.ones_like(cumprod)
    cumprod_shifted[window:] = cumprod[:-window]
        
    # Rolling product (growth over window)
    rolling_growth = cumprod / cumprod_shifted
    rolling_returns = rolling_growth - 1
        
    # Shift by 1 (to use yesterday's return for today's adjustment)
    rolling_returns_shifted = np.roll(rolling_returns, 1)
    rolling_returns_shifted[0] = 0
        
    return pd.Series(rolling_returns_shifted, index=returns.index)

def construct_portfolio(all_data, tickers, sector_map,
                        target_vol=0.5, vol_lookback=20,
                        max_ticker_weight=0.1, max_sector_weight=0.05,
                        max_leverage=2.0,
                        crisis_drawdown_threshold=-0.05,
                        crisis_leverage_multiplier=0.1):

    strategy_returns = all_data.pivot(index='timestamp', columns='symbol', values='strategy_net').fillna(0)
    
    rolling_vol = strategy_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)
    rolling_vol = rolling_vol.clip(lower=0.001)
    rolling_weights = 1.0 / rolling_vol
    row_sums = rolling_weights.sum(axis=1)
    rolling_weights = rolling_weights.div(row_sums, axis=0).clip(upper=max_sector_weight * max_ticker_weight)

    for date in rolling_weights.index:
        weights_day = rolling_weights.loc[date]
        sector_totals = {}

        for ticker in weights_day.index:
            sector = sector_map.get(ticker, "Other")
            sector_totals[sector] = sector_totals.get(sector, 0) + weights_day[ticker]

        for sector, total in sector_totals.items():
            if total > max_sector_weight:
                factor = max_sector_weight / total
                for ticker in [t for t, s in sector_map.items() if s == sector]:
                    if ticker in rolling_weights.columns:
                        rolling_weights.loc[date, ticker] *= factor

    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0)

    #Crisis detection using SPY drawdown
    
    crisis, crisis_mask = get_spy_drawdown(
        all_data['timestamp'].min(),
        all_data['timestamp'].max(),
        rolling_weights.index,
        crisis_drawdown_threshold
    )

    port = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)

    port_vol = port.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    port_vol = port_vol.clip(lower=0.05)

    leverage = (target_vol / port_vol).clip(0, max_leverage)
    leverage *= np.where(crisis == 1, crisis_leverage_multiplier, 1.0)
    port_levered = port * leverage.shift(1)

    rolling_20d_ret = fast_rolling_product(port, 20)

    port_levered *= np.where(rolling_20d_ret < -0.05, 0.5, 1.0)

    port_cum = (1 + port_levered).cumprod()

    return port_cum, rolling_weights, leverage
