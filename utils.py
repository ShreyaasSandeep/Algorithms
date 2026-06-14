import numpy as np
import pandas as pd
import requests
from config import FRED_API_KEY

def get_fred_data(series_id='DGS3MO', api_key=FRED_API_KEY, start_date=None, end_date=None):
    """Direct FRED API call - works with any Python version"""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Parse into DataFrame
    observations = []
    for obs in data["observations"]:
        if obs["value"] != ".":
            observations.append({
                "date": pd.to_datetime(obs["date"]).strftime('%Y-%m-%d'),
                "rate": float(obs["value"]) / 100
            })
    
    df = pd.DataFrame(observations)
    df = df.set_index("date").sort_index()
    return df["rate"]

def sharpe_ratio(strategy_returns, risk_free_series_id='DGS3MO', api_key=FRED_API_KEY, periods_per_year=252):
    """
    Calculate annual Sharpe ratio using FRED data for risk-free rate
    
    Parameters:
    - strategy_returns: pandas Series of strategy returns (indexed by date)
    - risk_free_series_id: FRED series ID for risk-free rate (e.g., 'DGS3MO' for 3-month Treasury)
    - api_key: FRED API key
    - periods_per_year: 252 for daily, 12 for monthly, 52 for weekly
    
    Returns:
    - Annual Sharpe ratio (float)
    """
    strategy_returns_copy = strategy_returns.copy()
    strategy_returns_copy.index = strategy_returns_copy.index.strftime('%Y-%m-%d')

    start_date = strategy_returns_copy.index[0]
    end_date = strategy_returns_copy.index[-1]
    
    risk_free_rates = get_fred_data(risk_free_series_id, api_key, start_date, end_date)
    
    # Align risk-free rates to strategy returns dates
    aligned_risk_free = risk_free_rates.reindex(strategy_returns_copy.index, method='ffill')
    
    # Convert annual risk-free rate to per-period rate
    risk_free_per_period = (1 + aligned_risk_free) ** (1/periods_per_year) - 1
    
    # Calculate excess returns and Sharpe ratio
    excess_returns = strategy_returns_copy - risk_free_per_period
    sharpe_daily = excess_returns.mean() / excess_returns.std()
    
    # Annualize and return
    return sharpe_daily * (periods_per_year ** 0.5)

def compute_performance(start_date, end_date, series, benchmark=None, freq=252):
    #Calculate returns and cumulative returns
    ret = series.pct_change().dropna()
    cumulative = (1 + ret).cumprod()
    

    #Calculate performance metrics
    total_return = cumulative.iloc[-1] - 1
    cagr = cumulative.iloc[-1] ** (freq / len(ret)) - 1
    vol = ret.std() * np.sqrt(freq)
    avg_return = ret.mean()
    volatility_adjusted_return = avg_return / ret.std() if ret.std() != 0 else np.nan

    dd = 1 - cumulative / cumulative.cummax()
    max_dd = dd.max()
    ulcer_index = np.sqrt(np.mean(dd**2))

    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(freq)
    sharpe = sharpe_ratio(ret, periods_per_year=freq)
    sortino = cagr / downside_vol if downside_vol != 0 else np.nan
    calmar = cagr / max_dd if max_dd != 0 else np.nan

    win_rate = (ret > 0).mean()
    avg_win = ret[ret > 0].mean() if (ret > 0).sum() > 0 else np.nan
    avg_loss = ret[ret < 0].mean() if (ret < 0).sum() > 0 else np.nan
    profit_factor = abs(avg_win / avg_loss) if avg_loss not in [0, np.nan] else np.nan
    num_trades = len(ret)

    skew = ret.skew()
    kurtosis = ret.kurtosis()
    tail_ratio = abs(ret.quantile(0.95) / ret.quantile(0.05)) if ret.quantile(0.05) != 0 else np.nan
    worst_day = ret.min()
    best_day = ret.max()

    variance_raw = ret.diff().dropna().std()
    variability_ratio = ret.std() / variance_raw if variance_raw != 0 else np.nan

    beta = alpha = information_ratio = r2 = tracking_error = np.nan
    if benchmark is not None:
        #Align returns with benchmark
        bench_ret = benchmark.pct_change().dropna()
        aligned = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
        aligned.columns = ["portfolio", "benchmark"]

        if len(aligned) > 5:
            cov = np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1]
            var = np.var(aligned["benchmark"])
            beta = cov / var if var != 0 else np.nan

            alpha = (avg_return - beta * aligned["benchmark"].mean()) * freq

            tracking_error = (aligned["portfolio"] - aligned["benchmark"]).std() * np.sqrt(freq)

            information_ratio = (cagr - (1 + bench_ret).prod() ** (freq / len(bench_ret)) + 1) / tracking_error \
                if tracking_error != 0 else np.nan

            r2 = np.corrcoef(aligned["portfolio"], aligned["benchmark"])[0, 1] ** 2

    return pd.Series({
        "Total Return": total_return,
        "CAGR": cagr,
        "Avg Return per Period": avg_return,
        "Volatility": vol,
        "Volatility-Adjusted Return": volatility_adjusted_return,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Ulcer Index": ulcer_index,
        "Skew": skew,
        "Kurtosis": kurtosis,
        "Win Rate": win_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Profit Factor": profit_factor,
        "Num Periods": num_trades,
        "Best Day": best_day,
        "Worst Day": worst_day,
        "Tail Ratio (95/5)": tail_ratio,
        "Variability Ratio": variability_ratio,
        "Beta vs Benchmark": beta,
        "Alpha (annualized)": alpha,
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio,
        "R² vs Benchmark": r2,
    })
