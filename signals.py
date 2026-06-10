import pandas as pd
import numpy as np
from modelling import rolling_sgd_predictions
from config import sector_map
import talib

#Hysteresis function to prevent frequent position changes
def apply_hysteresis(signal, upper=0, lower=-0.04):
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

#Enforce minimum holding period to reduce overtrading and transaction costs
def enforce_min_holding(positions, min_hold=5):
    final_pos = positions.copy()
    last_change = 0
    for i in range(1, len(positions)):
        if final_pos.iloc[i] != final_pos.iloc[i - 1]:
            if i - last_change < min_hold:
                final_pos.iloc[i] = final_pos.iloc[i - 1]
            else:
                last_change = i
    return final_pos

#Calculate ADX and related features for trend strength and regime detection
def calculate_adx(group, period=7):
        high = group['high'].values
        low = group['low'].values
        close = group['close'].values
        
        adx = talib.ADX(high, low, close, timeperiod=period)
        
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
        
        adx_slope = talib.ROCR(adx, timeperiod=5) - 1
        
        return pd.DataFrame({
            'ADX': adx,
            'PLUS_DI': plus_di,
            'MINUS_DI': minus_di,
            'ADX_slope': adx_slope
        }, index=group.index)

def calculate_vcpm(group, correlation_period=8, volume_period=128):
    """Volume-Confirmed Price Momentum - with safety bounds"""
    returns = group['close'].pct_change()
    volume_ratio = group['volume'] / (group['volume'].rolling(volume_period, min_periods=10).mean() + 1e-8)
    volume_ratio = volume_ratio.clip(0.1, 5)  # Tighter bounds
    
    # Rolling correlation with error handling
    price_vol_corr = returns.rolling(correlation_period, min_periods=5).corr(volume_ratio)
    price_vol_corr = price_vol_corr.clip(-1, 1).fillna(0)
    
    # VCPM signal
    vcpm_bullish = ((price_vol_corr > 0.5) & (volume_ratio > 0.8)).astype(int)
    vcpm_bearish = ((price_vol_corr < -0.5) & (volume_ratio > 0.8)).astype(int)
    
    return pd.DataFrame({
        'vcpm_correlation': price_vol_corr,
        'vcpm_bullish': vcpm_bullish,
        'vcpm_bearish': vcpm_bearish,
    }, index=group.index)


def compute_signals(all_data, target_vol=0.5,
                    cost_rate=0.001, slippage_rate=0.0005):

    #Prepare the DataFrame and calculate features
    df = all_data.copy().sort_values(['symbol', 'timestamp'])

    df['returns'] = df.groupby('symbol')['close'].pct_change()
    df['momentum_long'] = df.groupby('symbol')['close'].pct_change(20)
    df['momentum_short'] = df.groupby('symbol')['close'].pct_change(5)

    vol_short = df.groupby('symbol')['returns'].transform(lambda x: x.rolling(5, min_periods = 1).std())
    vol_long  = df.groupby('symbol')['returns'].transform(lambda x: x.rolling(20, min_periods=1).std())
    df['volatility_ratio'] = (vol_short / (vol_long + 1e-8)).fillna(1)

    df['signal_long'] = df.groupby('symbol')['momentum_long'].transform(
        lambda x: ((x - x.mean()) / (x.std() + 1e-8)).ewm(span=60).mean()
    )
    df['signal_short'] = df.groupby('symbol')['momentum_short'].transform(
        lambda x: ((x - x.mean()) / (x.std() + 1e-8)).ewm(span=20).mean()
    )

    df['rank_momentum'] = df.groupby('timestamp')['momentum_long'].rank(pct=True)
    df['sector_rank_momentum'] = df.groupby(
        ['timestamp', df['symbol'].map(sector_map)]
    )['momentum_long'].rank(pct=True)

    delta = df.groupby('symbol')['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust = False).mean()
    roll_down = down.ewm(span=14, adjust = False).mean()
    RS = roll_up / (roll_down + 1e-8)
    df['RSI_signal'] = (100 - (100 / (1 + RS)) - 50) / 50

    adx_features = df.groupby('symbol').apply(calculate_adx, period=7, include_groups=False).reset_index(level=0, drop=True)    
    df['ADX'] = adx_features['ADX']
    df['PLUS_DI'] = adx_features['PLUS_DI']
    df['MINUS_DI'] = adx_features['MINUS_DI']
    df['ADX_slope'] = adx_features['ADX_slope']
    df['ADX_normalized'] = df['ADX'] / 100.0
    
    df['ADX_regime'] = np.select(
        [df['ADX'] < 20, df['ADX'] < 40, df['ADX'] >= 40],
        [0, 1, 2],
        default=1
    )

    di_diff = (df['PLUS_DI'] - df['MINUS_DI']) / (df['PLUS_DI'] + df['MINUS_DI'] + 1e-8)
    df['DI_bias'] = np.tanh(di_diff)
    
    df['trend_signal'] = df['ADX_normalized'] * df['DI_bias']    

    df['SMA_200'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(200, min_periods=1).mean())
    df['EMA_200'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())

    sma_signal = np.where(df['close'] > df['SMA_200'], 1, 0.5)
    ema_signal = np.where(df['close'] > df['EMA_200'], 1, 0.5)

    trend_strength = (df['signal_long'].abs() - df['signal_long'].abs().min()) / \
                     (df['signal_long'].abs().max() - df['signal_long'].abs().min() + 1e-8)
    df['weighted_filter'] = trend_strength * ema_signal + (1 - trend_strength) * sma_signal

    bb_lookback = 20
    bb_k = 2
    df['SMA_BB'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(bb_lookback).mean())
    df['STD_BB'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(bb_lookback).std())
    df['BB_upper'] = df['SMA_BB'] + bb_k * df['STD_BB']
    df['BB_lower'] = df['SMA_BB'] - bb_k * df['STD_BB']
    df['BB_zscore'] = (df['close'] - df['SMA_BB']) / (df['STD_BB'] + 1e-8)

    # ============================================================
    # HILBERT TRANSFORM - Market Regime Detection
    # ============================================================
    # Calculate Hilbert Transform components per symbol
    def calculate_hilbert(group):
        close_prices = group['close'].values
        
        # Instantaneous trend line (predicts where price is heading)
        ht_trendline = talib.HT_TRENDLINE(close_prices)
        
        # Trend mode: 0 = Cyclical/Choppy, 1 = Strong Trend
        ht_trendmode = talib.HT_TRENDMODE(close_prices)
        
        # Optional: Hilbert Transform Sine Wave (phase prediction)
        # ht_sine, ht_leadsine = talib.HT_SINE(close_prices)
        
        return pd.DataFrame({
            'HT_Trendline': ht_trendline,
            'HT_Trendmode': ht_trendmode,
            # 'HT_Sine': ht_sine,
            # 'HT_LeadSine': ht_leadsine
        }, index=group.index)

    # Apply Hilbert transform to each symbol
    hilbert_features = df.groupby('symbol', group_keys=False).apply(calculate_hilbert)
    df['HT_Trendline'] = hilbert_features['HT_Trendline']
    df['HT_Trendmode'] = hilbert_features['HT_Trendmode']

    # Create a regime confidence score (0 to 1)
    # When HT_Trendmode = 1 (trending), confidence = 0.8 + ADX contribution
    # When HT_Trendmode = 0 (choppy), confidence = 0.2
    df['trend_regime_confidence'] = np.where(
        df['HT_Trendmode'] == 1,
        0.6 + (df['ADX_normalized'] * 0.3),  # Max 0.9 when ADX high
        0.2  # Low confidence in choppy markets
    )

    # Detect trend exhaustion: Price crossing below HT_Trendline signals trend weakening
    df['trend_exhaustion'] = (
        (df['close'] < df['HT_Trendline']) & 
        (df['close'].shift(1) >= df['HT_Trendline'].shift(1))
    ).astype(int)

    # Detect trend initiation: Price crossing above HT_Trendline
    df['trend_initiation'] = (
        (df['close'] > df['HT_Trendline']) & 
        (df['close'].shift(1) <= df['HT_Trendline'].shift(1))
    ).astype(int)

    eps = 1e-8
    vol_lookback_mean = 20
    vol_lookback_std = 20
    df['vol_mean'] = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(vol_lookback_mean, min_periods=1).mean()
    )
    df['vol_std'] = df.groupby('symbol')['volume'].transform(
        lambda x: x.rolling(vol_lookback_std, min_periods=1).std().fillna(0)
    )
    df['volume_spike_ratio_raw'] = df['volume'] / (df['vol_mean'] + eps)
    df['volume_spike_log'] = np.log1p(df['volume_spike_ratio_raw'])
    df['volume_zscore'] = ((df['volume'] - df['vol_mean']) /
                           (df['vol_std'] + eps))

    df['volume_spike_combined'] = (
        0.6 * (df['volume_spike_log'] / (df['volume_spike_log'].std() + eps)) +
        0.4 * (df['volume_zscore'] / (df['volume_zscore'].std() + eps))
    )

    df['volume_spike_rank'] =df.groupby('timestamp')['volume_spike_combined'].transform(
        lambda x: x.rank(pct=True)
    )

    df['efficiency_ratio'] = df.groupby('symbol')['close'].transform(
    lambda x: abs(x - x.shift(20)) / (abs(x.diff()).rolling(20).sum() + 1e-8)
    )

    df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * 
    (df.groupby('symbol')['high'].transform(lambda x: np.log(x / x.shift()))**2).rolling(20).mean() * 252)

    df['cmf'] = df.groupby('symbol').apply(
    lambda g: (g['volume'] * (2 * g['close'] - g['high'] - g['low']) / 
               (g['high'] - g['low'] + 1e-8)).rolling(20).sum() / 
              g['volume'].rolling(20).sum()
    ).reset_index(level=0, drop=True)

    vcpm_features = df.groupby('symbol').apply(calculate_vcpm, include_groups=False).reset_index(level=0, drop=True)
    df['vcpm_correlation'] = vcpm_features['vcpm_correlation']
    df['vcpm_bullish'] = vcpm_features['vcpm_bullish']
    df['vcpm_bearish'] = vcpm_features['vcpm_bearish']

    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['next_open_return'] = df['next_open'] / df['close'] - 1 

    features = [
        'signal_long', 'signal_short', 'RSI_signal', 'weighted_filter', 'BB_zscore', 
        'volatility_ratio', 'volume_spike_rank', 'rank_momentum', 'sector_rank_momentum',
        'ADX_normalized', 'ADX_regime', 'DI_bias', 'trend_signal', 'ADX_slope',
        'efficiency_ratio', 'parkinson_vol', 'cmf', 'vcpm_correlation', 'vcpm_bullish', 
        'vcpm_bearish', 'HT_Trendmode', 'trend_regime_confidence', 'trend_exhaustion', 
        'trend_initiation'
    ]
    df = df.dropna(subset=features + ['next_open_return']).copy()

    #Generate signals using the rolling SGD model and apply execution lag
    df = rolling_sgd_predictions(df, features)

    df['combined_signal_for_execution'] = df.groupby('symbol')['combined_signal'].shift(1)

    df['position_hysteresis'] = df.groupby('symbol')['combined_signal_for_execution'].transform(lambda x: apply_hysteresis(x.values))
    df['position_filtered'] = df['position_hysteresis'] * df['weighted_filter']

    scaling = (target_vol / df['parkinson_vol'].fillna(target_vol)).clip(0, 3)
    df['position_final'] = df['position_filtered'] * scaling

    df['position_final'] = df.groupby('symbol')['position_final'].transform(
        lambda x: enforce_min_holding(x, min_hold=60)
    )

    df['strategy'] = df['position_final'].shift(1) * (df['next_open'] / df['close'] - 1)

    #Calculate transaction costs based on position changes and estimated spread
    est_spread = df.groupby("symbol")["returns"].transform(lambda x: x.rolling(5, min_periods=1).std()) * 0.5
    est_spread = est_spread.clip(lower=0.0001)

    total_cost = cost_rate + slippage_rate + est_spread
    pos_change = df.groupby("symbol")["position_final"].diff().abs()

    df['strategy_net'] = df['strategy'] - pos_change * total_cost

    return df
