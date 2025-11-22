import yfinance as yf
API_KEY = "PKOYSXCCALC4OTPQ146W"
API_SECRET = "H8WojE6vXjVzwrW2CKSjXSomIJ5kyVBg5Gf9pS2o"
BASE_URL = "https://paper-api.alpaca.markets/v2"

tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "JPM", "JNJ", "XOM", "UNH",
    "PG", "NVDA", "HD", "V", "MA", "PFE", "BA", "KO", "PEP",
    "CAT", "T", "CVX"
]

sector_map = {}
for t in tickers:
    try:
        info = yf.Ticker(t).info
        sector_map[t] = info.get("sector", "Unknown")
    except:
        sector_map[t] = "Unknown"

#print(sector_map)
