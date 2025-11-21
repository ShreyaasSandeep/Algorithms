API_KEY = "PKOYSXCCALC4OTPQ146W"
API_SECRET = "H8WojE6vXjVzwrW2CKSjXSomIJ5kyVBg5Gf9pS2o"
BASE_URL = "https://paper-api.alpaca.markets/v2"

tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "JPM", "JNJ", "XOM", "UNH",
    "PG", "NVDA", "HD", "V", "MA", "PFE", "BA", "KO", "PEP",
    "CAT", "T", "CVX"
]

sector_map = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Communication Services",
    "AMZN": "Consumer Discretionary", "JPM": "Financials", "JNJ": "Healthcare",
    "XOM": "Energy", "UNH": "Healthcare", "PG": "Consumer Staples",
    "NVDA": "Technology", "HD": "Consumer Discretionary", "V": "Financials",
    "MA": "Financials", "PFE": "Healthcare", "BA": "Industrials",
    "KO": "Consumer Staples", "PEP": "Consumer Staples", "CAT": "Industrials",
    "T": "Communication Services", "CVX": "Energy"
}
