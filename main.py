from config import tickers, sector_map
from backtest import multi_ticker_momentum_alpaca

portfolio_cum, summary, weights, leverage = multi_ticker_momentum_alpaca(
    tickers=tickers,
    start="2018-01-01",
    end="2026-06-15",
    sector_map=sector_map,
    target_vol=0.5,
    cost_rate=0.001,
    slippage_rate=0.0005,
    vol_lookback=20,
    max_ticker_weight=0.05,
    max_sector_weight=0.05,
    max_leverage=1.5,
    crisis_drawdown_threshold=-0.1,
    crisis_leverage_multiplier=0.5,
    plot=True
)
