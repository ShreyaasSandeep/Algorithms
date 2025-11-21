import numpy as np
import pandas as pd

def compute_performance(series, benchmark=None):
    ret = series.pct_change().dropna()
    cumulative = (1 + ret).cumprod()

    annual_return = cumulative.iloc[-1] ** (252 / len(ret)) - 1
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol

    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = annual_return / downside_vol if downside_vol != 0 else np.nan

    max_dd = (1 - cumulative / cumulative.cummax()).max()
    calmar = annual_return / max_dd if max_dd != 0 else np.nan

    win_rate = (ret > 0).mean()

    beta = np.nan
    if benchmark is not None:
        bench_ret = benchmark.pct_change().dropna()
        aligned = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
        aligned.columns = ["portfolio", "benchmark"]
        if len(aligned) > 5:
            cov = np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1]
            var = np.var(aligned["benchmark"])
            beta = cov / var if var != 0 else np.nan

    return pd.Series({
        "CAGR": annual_return,
        "Volatility": annual_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Beta vs SPY": beta,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
    })
