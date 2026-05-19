import alpaca_trade_api as tradeapi
import pandas as pd
import os, pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import API_KEY, API_SECRET, BASE_URL

#Initializing Alpaca API client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Request bars from Alpaca API for a ticker with dividend/split adjustments
def fetch_one(symbol, start, end, timeframe='1D'):
    bars = api.get_bars(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment='all'
    ).df
    #If we have data, add symbol column for identification
    if not bars.empty:
        bars['symbol'] = symbol
    return bars

#Fetching data for multiple tickers simultaneously with caching and error handling
def fetch_alpaca_data_batch(tickers, start, end, timeframe='1D',
                            max_workers=8, cache_dir="cache", cache_expiry_days=7):
    #Ensuring cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    #Formatting start and end dates for cache file naming
    start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')

    #Lock for synchronizing cache writes across threads
    cache_lock = threading.Lock()

    #Checks if cache file exists and has not expired
    def is_cache_valid(cache_file):
        if not os.path.exists(cache_file):
            return False
        age_days = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache_file))).days
        return age_days < cache_expiry_days

    def task(ticker):
        #Builds cache file path for the ticker and date range
        cache_file = f"{cache_dir}/{ticker}_{start_str}_{end_str}_{timeframe}.pkl"

        #Loads file from cache if valid, otherwise fetches from API and saves to cache
        if is_cache_valid(cache_file):
            try:
                obj = pickle.load(open(cache_file, "rb"))
                if not obj.empty:
                    return obj
            except Exception as e:
                print(f"[{ticker}] failed to read cache, will refetch: {e}")

        bars = fetch_one(ticker, start_str, end_str, timeframe=timeframe)
        if bars is None or bars.empty:
            return pd.DataFrame()
        
        #Ensuring symbol column exists for consistency
        bars = bars.copy()
        if 'symbol' not in bars.columns:
            bars['symbol'] = ticker

        try:
            with cache_lock:
                tmp_file = cache_file + ".tmp"
                with open(tmp_file, "wb") as f:
                    pickle.dump(bars, f)
                os.replace(tmp_file, cache_file)
        except Exception as e:
            print(f"[{ticker}] error writing cache: {e}")

        return bars

    #Tracking all fetched data and any errors that occur during fetching
    all_data = []
    errors = {}

    #Using ThreadPoolExecutor to fetch data for multiple tickers in parallel with error handling
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(task, t): t for t in tickers}
        for fut in as_completed(future_to_ticker):
            t = future_to_ticker[fut]
            try:
                res = fut.result()
                if isinstance(res, pd.DataFrame) and not res.empty:
                    all_data.append(res)
                else:
                    print(f"[{t}] no data returned (empty DataFrame).")
            except Exception as e:
                print(f"[{t}] failed after retries: {e}")
                errors[t] = str(e)

    if not all_data:
        raise ValueError(f"No data fetched from Alpaca. Errors: {errors}")

    #Combining all fetched data into a single DataFrame and ensuring timestamp is in datetime format
    df = pd.concat(all_data).reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
