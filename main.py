from config import tickers, sector_map
from backtest import multi_ticker_momentum_alpaca

portfolio_cum, summary, weights, leverage = multi_ticker_momentum_alpaca(
    tickers=tickers,
    start="2024-01-01",
    end="2026-05-16",
    sector_map=sector_map,
    max_ticker_weight=0.1,
    max_sector_weight=0.05,
    crisis_drawdown_threshold=-0.05,
    plot=True
)
