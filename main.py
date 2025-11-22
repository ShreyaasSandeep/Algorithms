from config import tickers, sector_map
from backtest import multi_ticker_momentum_alpaca

portfolio_cum, summary, weights, leverage = multi_ticker_momentum_alpaca(
    tickers=tickers,
    start="2018-01-01",
    end="2025-11-21",
    sector_map=sector_map,
    max_ticker_weight=0.25,
    max_sector_weight=0.1,
    crisis_drawdown_threshold=-0.05,
    plot=True
)
