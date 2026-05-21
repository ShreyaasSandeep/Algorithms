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

    #Fetching historical data for all tickers and computing signals
    all_data = fetch_alpaca_data_batch(tickers, start, end)
    all_data = compute_signals(all_data, target_vol=target_vol,
                               cost_rate=cost_rate, slippage_rate=slippage_rate)

    #Constructing portfolio
    portfolio_cum, weights, leverage = construct_portfolio(
        all_data, tickers, sector_map,
        target_vol=target_vol, vol_lookback=vol_lookback,
        max_ticker_weight=max_ticker_weight,
        max_sector_weight=max_sector_weight,
        max_leverage=max_leverage,
        crisis_drawdown_threshold=crisis_drawdown_threshold,
        crisis_leverage_multiplier=crisis_leverage_multiplier
    )

    #Fetching SPY data for benchmark comparison
    spy = fetch_alpaca_data_batch(["SPY"], start, end)
    spy = spy[spy['symbol'] == "SPY"]
    spy = spy.rename(columns={"timestamp": "Date", "close": "Close"})
    spy = spy.set_index("Date")
    spy["returns"] = spy["Close"].pct_change()
    spy_cum = (1 + spy["returns"]).cumprod()

    #Computing performance metrics for portfolio
    summary = compute_performance(portfolio_cum, benchmark=spy_cum)

    #Calculating returns under an equal-weight buy-and-hold strategy for comparison
    bh = (
    all_data.pivot(index="timestamp", columns="symbol", values="close").pct_change(fill_method=None)
    )

    bh_port = (1 + bh.mean(axis=1)).cumprod()

    #Computing performance metrics for buy-and-hold strategy and SPY benchmark
    summary_bh = compute_performance(bh_port, benchmark=spy_cum)
    summary_spy = compute_performance(spy_cum, benchmark=spy_cum)

    #Fetching DBMF data for additional benchmark comparison and computing performance metrics
    quant = fetch_alpaca_data_batch(["DBMF"], start, end)
    quant = quant[quant["symbol"] == "DBMF"]
    quant = quant.rename(columns={"timestamp": "Date", "close": "Close"})
    quant = quant.set_index("Date")
    quant["returns"] = quant["Close"].pct_change()
    quant_cum = (1 + quant["returns"]).cumprod()
    summary_quant = compute_performance(quant_cum, benchmark=spy_cum)

    #Plotting log-scale graph comparing portfolio performance against benchmarks to show relative performance
    if plot:
        df_plot = pd.DataFrame({
            "Momentum Portfolio": portfolio_cum,
            "Equal-Weight BH": bh_port,
            "SPY Buy-and-Hold": spy_cum,
            "DBMF Buy-and-Hold": quant_cum
            
        })
        df_plot.plot(figsize=(12, 6), logy=True, title="Momentum Portfolio vs BH vs SPY vs DBMF")
        plt.show()

    #Printing performance summaries for portfolio and benchmarks
    print("Portfolio Summary:\n", summary)
    print("\nEqual-Weight BH Summary:\n", summary_bh)
    print("\nSPY Summary:\n", summary_spy)
    print("\nDBMF Summary:\n", summary_quant)

    return portfolio_cum, summary, weights, leverage
