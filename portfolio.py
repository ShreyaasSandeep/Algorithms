import pandas as pd
import numpy as np
from data_fetch import fetch_alpaca_data_batch

def get_spy_drawdown(start_date, end_date, rolling_weights_index, crisis_drawdown_threshold=-0.1):
        """Function to get crisis indicator"""
        
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
        
    rolling_returns_shifted = np.roll(rolling_returns, 1)
    rolling_returns_shifted[0] = 0
        
    return pd.Series(rolling_returns_shifted, index=returns.index)

def calculate_peak_to_trough_drawdown(returns, lookback_days=None):
    """
    Calculate peak-to-trough drawdown from a rolling peak.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns series
    lookback_days : int, optional
        If provided, only consider peaks within this window
    
    Returns:
    --------
    pd.Series : Current drawdown from the most recent peak
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    if lookback_days:
        # Rolling peak within lookback window
        rolling_peak = cum_returns.rolling(window=lookback_days, min_periods=1).max()
    else:
        # Global peak (all-time high)
        rolling_peak = cum_returns.cummax()
    
    # Calculate drawdown from peak
    drawdown = (cum_returns / rolling_peak) - 1
    
    return drawdown

def get_peak_to_trough_sizing_multiplier(returns, 
                                         entry_threshold=-0.05, 
                                         exit_threshold=-0.02,
                                         sizing_scheme='progressive'):
    """
    Generate sizing multipliers based on peak-to-trough drawdown state.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns series
    entry_threshold : float
        Drawdown level that triggers crisis mode (default: -5%)
    exit_threshold : float
        Drawdown level that exits crisis mode (default: -2%)
    sizing_scheme : str
        'progressive', 'step', or 'linear'
    
    Returns:
    --------
    tuple : (sizing_multiplier, in_crisis_mode, drawdown_series)
    """
    # Calculate drawdown from peak
    cum_returns = (1 + returns).cumprod()
    rolling_peak = cum_returns.cummax()
    drawdown = (cum_returns / rolling_peak) - 1
    
    in_crisis = pd.Series(False, index=returns.index)
    sizing_multiplier = pd.Series(1.0, index=returns.index)
    
    crisis_started = False
    current_trough = 0
    
    for i, idx in enumerate(returns.index):
        dd = drawdown.iloc[i]
        
        if not crisis_started:
            # Check if we've entered a drawdown beyond entry threshold
            if dd < entry_threshold:
                crisis_started = True
                current_trough = dd
                in_crisis.iloc[i] = True
                
                # Calculate initial sizing based on trough depth
                sizing_multiplier.iloc[i] = _get_sizing_from_drawdown(dd, sizing_scheme)
        else:
            # We're in crisis mode
            in_crisis.iloc[i] = True
            
            # Update trough if we go lower
            if dd < current_trough:
                current_trough = dd
                sizing_multiplier.iloc[i] = _get_sizing_from_drawdown(current_trough, sizing_scheme)
            else:
                # Recovery phase 
                if dd > exit_threshold:
                    # Exit crisis mode when we recover above exit threshold
                    crisis_started = False
                    in_crisis.iloc[i] = False
                    sizing_multiplier.iloc[i] = 1.0
                else:
                    # Still in drawdown but recovering - progressive increase
                    recovery_factor = 1 + (dd / exit_threshold)
                    sizing_multiplier.iloc[i] = max(0.1, min(1.0, recovery_factor))
    
    return sizing_multiplier, in_crisis, drawdown

def _get_sizing_from_drawdown(drawdown, sizing_scheme='progressive'):
    """Get sizing multiplier based on drawdown depth"""
    
    if sizing_scheme == 'progressive':
        # Smooth progressive scaling
        if drawdown < -0.15:
            return 0.1
        elif drawdown < -0.10:
            # Linear interpolation between 0.1 and 0.3
            return 0.1 + 0.2 * (abs(drawdown) - 0.15) / 0.05
        elif drawdown < -0.08:
            # Linear interpolation between 0.3 and 0.5
            return 0.3 + 0.2 * (abs(drawdown) - 0.10) / 0.02
        else:
            # Map drawdown to 0.5-1.0 range
            return max(0.5, 1 + drawdown / 0.08)
            
    elif sizing_scheme == 'step':
        # Discrete step function
        if drawdown < -0.15:
            return 0.1
        elif drawdown < -0.10:
            return 0.25
        elif drawdown < -0.08:
            return 0.5
        elif drawdown < -0.05:
            return 0.75
        else:
            return 1.0
            
    else:  # linear
        # Linear scaling from 1.0 at 0% to 0.1 at -15%
        return max(0.1, 1 + drawdown / 0.1667)

def get_drawdown_sizing_with_hysteresis(returns, 
                                        entry_threshold=-0.05,
                                        exit_threshold=-0.02,
                                        min_multiplier=0.1,
                                        max_multiplier=1.0):
    """
    Simplified version with hysteresis to avoid whipsawing.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    entry_threshold : float
        Drawdown to start reducing size
    exit_threshold : float
        Drawdown to return to full size
    min_multiplier : float
        Minimum sizing multiplier
    max_multiplier : float
        Maximum sizing multiplier
    
    Returns:
    --------
    pd.Series : Sizing multiplier for each period
    """
    # Calculate drawdown from peak
    cum_returns = (1 + returns).cumprod()
    rolling_peak = cum_returns.cummax()
    drawdown = (cum_returns / rolling_peak) - 1
    
    sizing = pd.Series(max_multiplier, index=returns.index)
    reduced_mode = False
    
    for i, idx in enumerate(returns.index):
        dd = drawdown.iloc[i]
        
        if not reduced_mode:
            if dd < entry_threshold:
                reduced_mode = True
                # Map drawdown to sizing (deeper drawdown = smaller size)
                depth_factor = max(min_multiplier, 1 + (dd / entry_threshold))
                sizing.iloc[i] = max(min_multiplier, min(max_multiplier, depth_factor))
        else:
            if dd > exit_threshold:
                reduced_mode = False
                sizing.iloc[i] = max_multiplier
            else:
                # Continue reduced sizing, scaling with recovery
                recovery_factor = 1 + (dd / exit_threshold)
                sizing.iloc[i] = max(min_multiplier, min(max_multiplier, recovery_factor))
    
    return sizing

def get_multi_asset_crisis_indicator(start_date, end_date, rolling_weights_index,
                                     crisis_assets=None,
                                     crisis_drawdown_threshold=-0.05,
                                     min_assets_in_crisis=5,
                                     persistence_days=1):
    """
    Detect systemic crises using multiple uncorrelated assets.
    
    Parameters:
    -----------
    start_date, end_date : datetime or string
        Date range for data
    rolling_weights_index : pd.Index
        Index to align the crisis signal to
    crisis_assets : list
        List of tickers to monitor for crisis detection
        Default: ['SPY', 'TLT', 'GLD', 'HYG', 'VXX']
    crisis_drawdown_threshold : float
        Drawdown threshold to consider an asset in crisis (default: -5%)
    min_assets_in_crisis : int
        Minimum number of assets that must be in drawdown to trigger crisis
    persistence_days : int
        Number of days to stay in crisis mode once triggered (default: 5)
    
    Returns:
    --------
    tuple : (crisis_series, crisis_mask)
        crisis_series: 0/1 indicator of crisis periods
        crisis_mask: boolean mask of crisis periods
    """
    
    # Set default crisis assets if not provided
    if crisis_assets is None:
        crisis_assets = ['SPY', 'TLT', 'GLD', 'HYG', 'VXX']
    
    # Cache implementation
    cache_key = f"multi_crisis_{start_date}_{end_date}_{'_'.join(crisis_assets)}"
    if not hasattr(get_multi_asset_crisis_indicator, "cache"):
        get_multi_asset_crisis_indicator.cache = {}
    
    if cache_key in get_multi_asset_crisis_indicator.cache:
        crisis = get_multi_asset_crisis_indicator.cache[cache_key]
    else:
        try:
            # Fetch all crisis assets
            all_data = fetch_alpaca_data_batch(crisis_assets, start_date, end_date)
            
            crisis_signals = []
            asset_drawdowns = {}
            
            for asset in crisis_assets:
                # Get asset data
                asset_data = all_data[all_data['symbol'] == asset].set_index('timestamp')['close']
                
                # Calculate drawdown
                asset_cummax = asset_data.cummax()
                drawdown = asset_data / asset_cummax - 1
                asset_drawdowns[asset] = drawdown
                
                # Generate crisis signal for this asset
                crisis_signal = (drawdown < crisis_drawdown_threshold).astype(int)
                crisis_signals.append(crisis_signal)
            
            # Combine signals - crisis if enough assets are in drawdown
            crisis_df = pd.concat(crisis_signals, axis=1)
            crisis_df.columns = crisis_assets
            
            # Count how many assets are in crisis each day
            assets_in_crisis = crisis_df.sum(axis=1)
            
            # Trigger crisis if at least 'min_assets_in_crisis' are in drawdown
            crisis_raw = (assets_in_crisis >= min_assets_in_crisis).astype(int)
            
            if persistence_days > 1:
                # Use rolling window to propagate crisis signals forward
                crisis_rolling = crisis_raw.rolling(window=persistence_days, min_periods=1).max()
                crisis = crisis_rolling.astype(int)
            else:
                crisis = crisis_raw
            
            # Store in cache
            get_multi_asset_crisis_indicator.cache[cache_key] = crisis
            
            print(f"Multi-asset crisis indicator created using: {', '.join(crisis_assets)}")
            print(f"  - Crisis threshold: {crisis_drawdown_threshold*100:.0f}% drawdown")
            print(f"  - Min assets required: {min_assets_in_crisis}")
            print(f"  - Persistence: {persistence_days} days")
            print(f"  - Crisis periods detected: {crisis.sum()} days ({crisis.mean()*100:.2f}% of time)")
            
        except Exception as e:
            print(f"Warning: Could not fetch multi-asset crisis data: {e}")
            print(f"Falling back to SPY-only crisis detection")
            
            # Fallback to SPY-only
            try:
                spy_data = fetch_alpaca_data_batch(["SPY"], start_date, end_date)
                spy = spy_data[spy_data['symbol'] == "SPY"].set_index('timestamp')['close']
                spy_cummax = spy.cummax()
                drawdown = spy / spy_cummax - 1
                crisis_raw = (drawdown < crisis_drawdown_threshold).astype(int)
                
                # Apply persistence to SPY fallback as well
                if persistence_days > 1:
                    crisis_rolling = crisis_raw.rolling(window=persistence_days, min_periods=1).max()
                    crisis = crisis_rolling.astype(int)
                else:
                    crisis = crisis_raw
                    
            except Exception as e2:
                print(f"Error with fallback: {e2}")
                # Ultimate fallback: no crisis
                date_range = pd.date_range(start_date, end_date, freq='D')
                crisis = pd.Series(0, index=date_range)
            
            get_multi_asset_crisis_indicator.cache[cache_key] = crisis
    
    # Align with rolling_weights index efficiently
    if not crisis.index.equals(rolling_weights_index):
        crisis = crisis.reindex(rolling_weights_index, method='ffill').fillna(0)
    
    return crisis, crisis == 1

def construct_portfolio(all_data, tickers, sector_map,
                        target_vol=0.5, vol_lookback=20,
                        max_ticker_weight=0.1, max_sector_weight=0.05,
                        max_leverage=2.0,
                        crisis_drawdown_threshold=-0.05,
                        crisis_leverage_multiplier=0.6,
                        use_peak_to_trough=True,
                        drawdown_sizing_scheme='progressive',
                        drawdown_exit_threshold=None,
                        use_multi_asset_crisis=True,
                        crisis_assets=None,
                        min_assets_in_crisis=5,
                        multi_asset_crisis_multiplier=None):

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

    # Calculate portfolio returns
    port = (strategy_returns * rolling_weights.shift(1)).sum(axis=1)
    
    if use_peak_to_trough:
        # Set default exit threshold if not provided
        if drawdown_exit_threshold is None:
            drawdown_exit_threshold = crisis_drawdown_threshold / 2
        
        # Calculate sizing multiplier based on peak-to-trough drawdown
        sizing_multiplier, crisis_mask, drawdown_series = get_peak_to_trough_sizing_multiplier(
            port,
            entry_threshold=crisis_drawdown_threshold,
            exit_threshold=drawdown_exit_threshold,
            sizing_scheme=drawdown_sizing_scheme
        )
        
        # Use multi-asset crisis detection
        crisis_assets = ['SPY', 'TLT', 'GLD', 'HYG', 'VXX']
        
        crisis, spy_crisis_mask = get_multi_asset_crisis_indicator(
            all_data['timestamp'].min(),
            all_data['timestamp'].max(),
            rolling_weights.index,
            crisis_assets=crisis_assets,
            crisis_drawdown_threshold=crisis_drawdown_threshold,
            min_assets_in_crisis=5
        )
        
        multi_asset_multiplier = crisis_leverage_multiplier * 1.5  
        multi_asset_multiplier = min(0.5, multi_asset_multiplier) 
        
        spy_multiplier = np.where(spy_crisis_mask, multi_asset_multiplier, 1.0)
        
        # Combine both filters
        combined_multiplier = np.minimum(sizing_multiplier, spy_multiplier)
        
    else:
        # SPY-based crisis detection
        crisis, crisis_mask = get_spy_drawdown(
            all_data['timestamp'].min(),
            all_data['timestamp'].max(),
            rolling_weights.index,
            crisis_drawdown_threshold
        )
        combined_multiplier = np.where(crisis == 1, crisis_leverage_multiplier, 1.0)

    # Calculate volatility and leverage
    port_vol = port.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    port_vol = port_vol.clip(lower=0.05)

    leverage = (target_vol / port_vol).clip(0, max_leverage)
    
    # Apply combined sizing filter
    leverage *= combined_multiplier
    
    port_levered = port * leverage.shift(1)

    # Additional 20-day drawdown filter
    rolling_20d_ret = fast_rolling_product(port, 20)
    port_levered *= np.where(rolling_20d_ret < -0.05, 0.5, 1.0)

    port_cum = (1 + port_levered).cumprod()

    return port_cum, rolling_weights, leverage