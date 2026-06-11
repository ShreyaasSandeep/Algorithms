import pandas as pd
import numpy as np
from data_fetch import fetch_alpaca_data_batch

def construct_portfolio(all_data, tickers, sector_map,
                        target_vol=0.5, vol_lookback=20,
                        max_ticker_weight=0.1, max_sector_weight=0.05,
                        max_leverage=2.0,
                        crisis_drawdown_threshold=-0.05,
                        crisis_leverage_multiplier=0.2):

    strategy_returns = all_data.pivot(index='timestamp', columns='symbol', values='strategy_net').fillna(0)
    
    #Inverse volatility weighting
    rolling_vol = strategy_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)
    rolling_vol = rolling_vol.clip(lower=0.001)
    rolling_weights = 1 / rolling_vol
    effective_ticker_cap = max_sector_weight *max_ticker_weight
    rolling_weights = rolling_weights.div(rolling_weights.sum(axis=1), axis=0).clip(upper=effective_ticker_cap)

    #Sector weight constraints
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
    spy = fetch_alpaca_data_batch(["SPY"],
                                  all_data['timestamp'].min(),
                                  all_data['timestamp'].max())
    spy = spy[spy['symbol']=="SPY"].set_index('timestamp')['close']
    spy_cummax = spy.cummax()

    crisis = (spy / spy_cummax - 1 < crisis_drawdown_threshold).astype(int)
    crisis = crisis.reindex(rolling_weights.index,method="ffill")
    crisis_mask = crisis == 1

    #Calculating portfolio returns
    port = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)

    # Adjust leverage based on volatility and crisis conditions
    port_vol = port.rolling(vol_lookback,min_periods=vol_lookback).std()* np.sqrt(252)
    port_vol = port_vol.clip(lower=0.05)

    leverage = (target_vol / port_vol).clip(0, max_leverage)
    leverage *= np.where(crisis == 1, crisis_leverage_multiplier, 1.0)

    # Apply leverage to portfolio returns
    port_levered = port * leverage.shift(1)

    # Additional crisis-based leverage adjustment
    rolling_20d_ret = (1 + port).rolling(20).apply(np.prod, raw=True)- 1
    port_levered *= np.where(rolling_20d_ret < -0.05, 0.5, 1.0)

    #Cumulative returns
    port_cum = (1 + port_levered).cumprod()

    return port_cum, rolling_weights, leverage
