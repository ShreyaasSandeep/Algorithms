import yfinance as yf
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL")

#Check if keys loaded successfully
if not API_KEY:
    print("Warning: API_KEY not found in environment variables")

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

#Generates a mapping of tickers to their respective sectors
sector_map = {}
for t in tickers:
    try:
        info = yf.Ticker(t).info
        sector_map[t] = info.get("sector", "Unknown")
    except:
        sector_map[t] = "Unknown"

