import pandas as pd
from data_fetch import fetch_alpaca_data_batch
from signals import compute_signals
from portfolio import construct_portfolio
from utils import compute_performance
import matplotlib.pyplot as plt

def multi_ticker_momentum_alpaca(tickers, start, end,
                                 sector_map,
                                 target_vol=0.5, cost_rate=0.001, slippage_rate=0.0005,
                                 vol_lookback=20, max_ticker_weight=0.25,
                                 max_sector_weight=0.1, max_leverage=2.0,
                                 crisis_drawdown_threshold=-0.10,
                                 crisis_leverage_multiplier=0.2,
                                 plot=True):

    all_data = fetch_alpaca_data_batch(tickers, start, end)
    all_data = compute_signals(all_data, target_vol=target_vol,
                               cost_rate=cost_rate, slippage_rate=slippage_rate)

    portfolio_cum, weights, leverage = construct_portfolio(
        all_data, tickers, sector_map,
        target_vol=target_vol, vol_lookback=vol_lookback,
        max_ticker_weight=max_ticker_weight,
        max_sector_weight=max_sector_weight,
        max_leverage=max_leverage,
        crisis_drawdown_threshold=crisis_drawdown_threshold,
        crisis_leverage_multiplier=crisis_leverage_multiplier
    )

    spy = fetch_alpaca_data_batch(["SPY"], start, end)
    spy = spy[spy['symbol'] == "SPY"]
    spy = spy.rename(columns={"timestamp": "Date", "close": "Close"})
    spy = spy.set_index("Date")
    spy["returns"] = spy["Close"].pct_change()
    spy_cum = (1 + spy["returns"]).cumprod()


    summary = compute_performance(portfolio_cum, benchmark=spy_cum)
    bh = pd.DataFrame({
        t: fetch_alpaca_data_batch([t], start, end).set_index('timestamp')['close'].pct_change()
        for t in tickers
    }).dropna()

    bh_port = (1 + bh.mean(axis=1)).cumprod()
    summary_bh = compute_performance(bh_port)
    summary_spy = compute_performance(spy_cum)

    if plot:
        df_plot = pd.DataFrame({
            "Momentum Portfolio": portfolio_cum,
            "Equal-Weight BH": bh_port,
            "SPY Buy-and-Hold": spy_cum
        })
        df_plot.plot(figsize=(12, 6), logy=True, title="Momentum Portfolio vs BH vs SPY")
        plt.show()

    print("Portfolio Summary:\n", summary)
    print("\nEqual-Weight BH Summary:\n", summary_bh)
    print("\nSPY Summary:\n", summary_spy)

    return portfolio_cum, summary, weights, leverage
