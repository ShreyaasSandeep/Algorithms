import yfinance as yf
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL")
FRED_API_KEY = os.getenv("FRED_API_KEY")

#Check if keys loaded successfully
if not API_KEY:
    print("Warning: API_KEY not found in environment variables")
if not FRED_API_KEY:
    print("Warning: FRED_API_KEY not found in environment variables")

tickers = [
    "AAPL","MSFT","GOOG","GOOGL","NVDA","IBM","ORCL","CSCO","HPQ","DELL",
    "ADBE","INTC","QCOM","TXN","MU","AMD","CRM","PYPL","INTU","AMAT",
    "WDAY","ADSK","FISV","NOW","APH", "META","NFLX","DIS","CMCSA","T","VZ",
    "TMUS","EA","CHTR", "AMZN","HD","MCD","SBUX","NKE","LOW","TGT","TJX",
    "BKNG","GM","F","TSLA","EBAY","ROST","ULTA","LVS","MAR","HLT","DPZ","DG",
    "PG","KO","PEP","WMT","COST","MDLZ","MO","PM","KHC","CL", "K","KR","EL",
    "GIS","SYY","JPM","BAC","WFC","C","MS","GS","BLK","USB","PNC","SCHW","AIG",
    "MET","AFL","SPGI","ICE","CB","TRV", "JNJ","PFE","MRK","UNH","ABBV","AMGN",
    "GILD","BMY","LLY","MDT","TMO","SYK","ZTS","CVS","CI","BA","CAT","MMM","GE",
    "HON","UNP","UPS","FDX","LMT","RTX","DE","NOC","EMR","ETN","GD","DAL","AAL",
    "UAL","CSX","NSC", "LIN","APD","ECL","SHW","NUE","FCX","DD","MLM","VMC","PLD",
    "AMT","EQIX","SPG","O","VTR","WELL","AVB","EQR","NEE","DUK","SO","D","AEP","EXC",
    "SRE","WEC","XEL",
]

sector_map = {}

def get_sector_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return ticker, info.get("sector", "Unknown")
    except:
        return ticker, "Unknown"

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(get_sector_info, t): t for t in tickers}
    for future in as_completed(futures):
        ticker, sector = future.result()
        sector_map[ticker] = sector