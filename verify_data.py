import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi

load_dotenv()

alpaca = tradeapi.REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),  # should be https://paper-api.alpaca.markets
    api_version='v2'
)

try:
    ticker = "AAPL"
    start = (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"
    end = datetime.utcnow().isoformat() + "Z"

    print(f"📊 Fetching {ticker} data from {start} to {end}...")
    bars = alpaca.get_bars(ticker, timeframe="1Min", start=start, end=end, feed="iex").df

    if bars.empty:
        print("❌ No data returned.")
    else:
        print(f"✅ {len(bars)} bars returned.")
        print(bars.head())
except Exception as e:
    print(f"🚨 Exception: {e}")
