import pandas as pd
import numpy as np
from modelling import rolling_sgd_predictions
from config import sector_map

def apply_hysteresis(signal, upper=-0, lower=-0.04):
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

def compute_signals(all_data, target_vol=0.5,
                    cost_rate=0.001, slippage_rate=0.0005):

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

    df['next_open'] = df.groupby('symbol')['open'].shift(-1)
    df['next_open_return'] = df['next_open'] / df['close'] - 1 

    features = [
        'signal_long', 'signal_short', 'RSI_signal',
        'weighted_filter', 'BB_zscore', 'volatility_ratio',
        'volume_spike_rank', 'rank_momentum', 'sector_rank_momentum'
    ]
    df = df.dropna(subset=features + ['next_open_return']).copy()

    df = rolling_sgd_predictions(df, features)
    df['combined_signal_for_execution'] = df.groupby('symbol')['combined_signal'].shift(1)

    df['position_hysteresis'] = df.groupby('symbol')['combined_signal_for_execution'].transform(lambda x: apply_hysteresis(x.values))
    df['position_filtered'] = df['position_hysteresis'] * df['weighted_filter']

    realised_vol = df.groupby('symbol')['returns'].transform(lambda x: x.rolling(20, min_periods=1).std() * np.sqrt(252))
    scaling = (target_vol / realised_vol).clip(0, 3)
    df['position_final'] = df['position_filtered'] * scaling

    df['position_final'] = df.groupby('symbol')['position_final'].transform(
        lambda x: enforce_min_holding(x, min_hold=60)
    )

    df['strategy'] = df['position_final'].shift(1) * (df['next_open'] / df['close'] - 1)

    est_spread = df.groupby("symbol")["returns"].transform(lambda x: x.rolling(5, min_periods=1).std()) * 0.5
    est_spread = est_spread.clip(lower=0.0001)

    total_cost = cost_rate + slippage_rate + est_spread
    pos_change = df.groupby("symbol")["position_final"].diff().abs()

    df['strategy_net'] = df['strategy'] - pos_change * total_cost

    return df
