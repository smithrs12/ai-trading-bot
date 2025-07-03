# main.py

import os
import time
import pytz
import torch
import joblib
import random
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from alpaca_trade_api.rest import REST, TimeFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
import asyncio
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# === DEBUG Mode Toggle ===
DEBUG = True

def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# === Initialization ===
load_dotenv()
pacific = timezone('US/Pacific')

# === Load Environment Variables ===
GSPREAD_JSON_PATH = os.getenv("GSPREAD_JSON_PATH")
GSHEET_ID = os.getenv("GSHEET_ID")

# === Ensure model directories exist ===
os.makedirs("models/short", exist_ok=True)
os.makedirs("models/medium", exist_ok=True)

# === API Keys & Setup ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
analyzer = SentimentIntensityAnalyzer()

# Google Sheets Setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(GSPREAD_JSON_PATH, scope)
gc = gspread.authorize(credentials)
sheet = gc.open("MetaModelLog").sheet1

# === Fallback Universe (Large Cap, Diverse Sectors) ===
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AMD", "NFLX", "CRM",
    "BRK.B", "V", "MA", "JPM", "BAC", "WFC", "C", "GS", "MS",
    "XOM", "CVX", "COP", "SLB", "PSX", "EOG",
    "PFE", "JNJ", "LLY", "MRK", "ABT", "BMY", "CVS", "UNH",
    "HD", "LOW", "COST", "TGT", "WMT", "PG", "PEP", "KO", "PM",
    "UNP", "CSX", "UPS", "FDX", "CAT", "DE", "GE", "HON",
    "NKE", "SBUX", "MCD", "CMG", "DIS", "WBD", "ROKU",
    "ORCL", "IBM", "INTC", "QCOM", "AVGO", "TXN", "MU", "ADBE", "SNOW", "SHOP",
    "PLD", "O", "SPG", "AMT", "CCI",
    "BA", "LMT", "RTX", "NOC", "TDG",
    "ZM", "ROKU", "DOCU", "UBER", "LYFT", "ABNB", "SQ", "PYPL", "COIN", "SOFI",
    "TLRY", "CGC", "CRON", "ACB",
]

# === Utility ===
def send_discord_alert(message):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except:
        pass

# === Reinforcement Learning ===
class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

q_net = QNetwork(input_dim=4)  # adjust input_dim based on q_state features
if os.path.exists("q_net.pth"):
    q_net.load_state_dict(torch.load("q_net.pth"))
    print("🧠 Q-network loaded from q_net.pth")
else:
    print("⚠️ No existing Q-network found. Starting fresh.")
q_net.eval()

def q_state(ticker, position):
    try:
        barset = api.get_bars(ticker, TimeFrame.Minute, limit=5).df
        if barset.empty:
            return torch.zeros(4)
        change = (barset.close.iloc[-1] - barset.close.iloc[0]) / barset.close.iloc[0]
        avg_volume = barset.volume.mean()
        last_volume = barset.volume.iloc[-1]
        rsi = RSIIndicator(close=barset.close).rsi().iloc[-1] / 100.0
        state = torch.tensor([change, avg_volume / 1e6, last_volume / 1e6, rsi], dtype=torch.float32)
        return state
    except:
        return torch.zeros(4)

# === Meta Model Trainer ===
def train_meta_model():
    try:
        df = pd.read_csv("meta_model_log.csv")
        if len(df) < 50:
            print("⚠️ Not enough data to train meta model.")
            return
        X = df.drop(["timestamp", "ticker", "label"], axis=1)
        y = df["label"].map({"approved": 1, "rejected": 0})
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, "meta_model.pkl")
        print("✅ Trained and saved meta model.")
    except Exception as e:
        print(f"❌ Meta model training failed: {e}")
        send_discord_alert(f"⚠️ Meta model training failed: {e}")

def train_model(ticker, df, model_class, model_path):
    try:
        X = df.drop("target", axis=1)
        y = df["target"]
        model = model_class()
        try:
            model.fit(X, y)
        except Exception as e:
            handle_model_training_failure(ticker)
            print(f"❌ Model training failed for {ticker}: {e}")
            send_discord_alert(f"⚠️ Training failed for {ticker}: {e}")
            return None

        joblib.dump(model, model_path)
        log(f"✅ Trained model for {ticker}")
        return model

    except Exception as e:
        handle_model_training_failure(ticker)
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Training failed for {ticker}: {e}\n{tb}")
        send_discord_alert(f"⚠️ Model training failed for {ticker}")
        return None

def walk_forward_validation(X, y, model):
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)

        return sum(scores) / len(scores)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"⚠️ Walk-forward validation error: {e}\n{tb}")
        send_discord_alert(f"⚠️ Walk-forward validation error: {e}")
        return 0

def get_data(ticker, days=2):
    try:
        end_dt = datetime.now(pytz.utc)
        start_dt = end_dt - timedelta(days=days)
        bars = api.get_bars(
            ticker,
            TimeFrame.Minute,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            feed='iex'
        ).df

        if bars.empty:
            return None

        bars = bars[bars['volume'] > 0].copy()

        # Normalize column names to match the rest of your pipeline
        bars.rename(columns={
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume"
        }, inplace=True)

        # Add TA indicators
        bars["RSI"] = RSIIndicator(close=bars["Close"], window=14).rsi()
        bars["MACD"] = MACD(close=bars["Close"]).macd_diff()
        bars["OBV"] = OnBalanceVolumeIndicator(close=bars["Close"], volume=bars["Volume"]).on_balance_volume()

        # Assign 5-minute prediction target
        bars["Target"] = (bars["Close"].shift(-5) > bars["Close"]).astype(int)

        bars.dropna(inplace=True)
        return bars

    except Exception as e:
        print(f"Data fetch failed for {ticker}: {e}")
        return None

# === Prediction ===
def dual_horizon_predict(ticker, df):
    try:
        if df is None or df.empty:
            return None, None
        X = df.drop(columns=["Target"])

        short_model_path = f"models/short/{ticker}.pkl"
        mid_model_path = f"models/medium/{ticker}.pkl"

        if not os.path.exists(short_model_path) or not os.path.exists(mid_model_path):
            return None, None

        short_model = joblib.load(short_model_path)
        mid_model = joblib.load(mid_model_path)

        short_proba = short_model.predict_proba([X.iloc[-1]])[0][1]
        mid_proba = mid_model.predict_proba([X.iloc[-1]])[0][1]
        return short_proba, mid_proba
    except Exception as e:
        print(f"Prediction failed for {ticker}: {e}")
        return None, None

# === Main Loop ===
def is_market_open():
    print("⏳ Checking if market is open...", flush=True)
    clock = api.get_clock()
    print(f"✅ Market open status: {clock.is_open}", flush=True)
    return clock.is_open
    
def is_near_market_close():
    now = datetime.now(pytz.timezone("US/Eastern"))
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return (market_close - now).total_seconds() <= 300  # 5 minutes

# === Q-Learning Update ===
def update_q_network_from_log():
    try:
        if not os.path.exists("trade_outcomes.csv"):
            print("📭 No trade outcome log found.")
            return [], []

        df = pd.read_csv("trade_outcomes.csv", header=None)
        df.columns = ["timestamp", "ticker", "price", "score", "action", "result", "sentiment", "regime"]
        df = df.dropna(subset=["ticker", "price", "action"])

        rewards = {
            "profit": 1.0,
            "loss": -1.0,
            "decay": -0.3,
            "": 0.0
        }

        learning_rate = 0.01
        gamma = 0.95

        log_q_values = []
        trade_summary = []
        meta_data = []

        for _, row in df.iterrows():
            state = q_state(row["ticker"], 1).unsqueeze(0)
            old_q = q_net(state)
            reward = rewards.get(row["result"], 0.0)
            new_q = old_q + learning_rate * (reward + gamma * old_q.detach() - old_q)
            q_net.zero_grad()
            loss = torch.nn.functional.mse_loss(old_q, new_q.detach())
            loss.backward()
            for param in q_net.parameters():
                param.data -= learning_rate * param.grad

            log_q_values.append((row["timestamp"], row["ticker"], old_q.item(), reward))
            trade_summary.append((row["ticker"], row["price"], row["result"]))

            meta_data.append([
                row["ticker"],
                row["score"],
                row["sentiment"],
                row["regime"],
                datetime.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%S.%f").hour,
                datetime.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%S.%f").weekday(),
                1 if row["result"] == "profit" else 0
            ])

        torch.save(q_net.state_dict(), "q_net.pth")
        print("✅ Q-network updated from trade outcomes.")

        with open("q_evolution_log.csv", "a") as f:
            for ts, ticker, q_val, r in log_q_values:
                f.write(f"{ts},{ticker},{q_val:.4f},{r}\n")

        # Save meta training data
        meta_df = pd.DataFrame(meta_data, columns=["ticker", "score", "sentiment", "regime", "label"])
        meta_df.to_csv("meta_training_data.csv", mode="a", index=False, header=not os.path.exists("meta_training_data.csv"))

        # Auto-train meta model
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            if len(meta_df) >= 10:
                X = meta_df[["score", "sentiment", "regime"]]
                y = meta_df["label"]
                model = RandomForestClassifier(n_estimators=100)
                model.fit(X, y)
                joblib.dump(model, "meta_model.pkl")
                print("✅ Meta model retrained.")
        except Exception as e:
            print(f"⚠️ Meta model training failed: {e}")

        os.remove("trade_outcomes.csv")
        print("🧹 trade_outcomes.csv purged after update.")

        return trade_summary, log_q_values

    except Exception as e:
        print(f"❌ Failed to update Q-network: {e}")
        return [], []

# === VWAP + Volume Filter ===
def passes_volume_vwap_filter(ticker):
    try:
        df = api.get_bars(ticker, TimeFrame.Minute, limit=20).df
        if df.empty or len(df) < 5:
            return False
        vwap = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        current_close = df["close"].iloc[-1]
        current_volume = df["volume"].iloc[-1]
        avg_volume = df["volume"].mean()
        return current_close >= vwap.iloc[-1] and current_volume >= 1.5 * avg_volume
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ VWAP/Volume check error for {ticker}: {e}\n{tb}")
        send_discord_alert(f"⚠️ VWAP/Volume check error for {ticker}: {e}")
        return False

# === Sector Rotation Limits ===
sector_allocations = {}
MAX_SECTOR_EXPOSURE = 10

# === Dynamic Watchlist Sector Limits ===
MAX_PER_SECTOR_WATCHLIST = 12  # max tickers per sector in watchlist

def check_sector_allocation(ticker):
    try:
        sector = get_sector(ticker)
        count = sector_allocations.get(sector, 0)
        if count >= MAX_SECTOR_EXPOSURE:
            print(f"🛑 Sector limit reached for {sector}")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Sector check failed for {ticker}: {e}")
        return True

def update_sector_allocation(ticker):
    sector = get_sector(ticker)
    sector_allocations[sector] = sector_allocations.get(sector, 0) + 1

# === Market Regime Detection ===
def get_market_regime():
    try:
        spy = api.get_bars("SPY", TimeFrame.Day, limit=20).df
        if spy is None or spy.empty or len(spy) < 10:
            return 0  # neutral

        recent_returns = (spy.close.pct_change().dropna()[-5:]).mean()

        if recent_returns > 0.005:
            return 1  # bull
        elif recent_returns < -0.005:
            return -1  # bear
        else:
            return 0  # sideways
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Failed to determine market regime: {e}\n{tb}")
        send_discord_alert(f"⚠️ Market regime detection error: {e}")
        return 0

# === Data Fetching Utilities ===
def calculate_support_resistance(df, window=20):
    try:
        if df is None or df.empty or len(df) < window:
            return None, None

        df = df.tail(window)
        support = df['low'].min()
        resistance = df['high'].max()
        return support, resistance
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Failed to calculate support/resistance: {e}\n{tb}")
        send_discord_alert(f"⚠️ Support/Resistance calc error: {e}")
        return None, None
        
def get_sector(symbol):
    try:
        asset = api.get_asset(symbol)
        if hasattr(asset, 'sector') and asset.sector:
            return asset.sector
    except:
        pass

    try:
        if os.path.exists("sector_lookup.csv"):
            df = pd.read_csv("sector_lookup.csv")
            match = df[df["symbol"] == symbol.upper()]
            if not match.empty:
                return match.iloc[0]["sector"]
    except Exception as e:
        print(f"❌ Sector fallback failed for {symbol}: {e}")

    return "Unknown"
    
def get_data_alpaca(ticker, timeframe=TimeFrame.Minute, limit=100):
    try:
        bars = api.get_bars(
            ticker,
            timeframe,
            limit=limit,
            feed='iex'  # Force IEX feed
        ).df
        return bars if not bars.empty else None
    except Exception as e:
        print(f"❌ Failed to get Alpaca data for {ticker}: {e}")
        return None

# === Sentiment Using FinBERT (Async with Caching + Fallback) ===
_finbert_pipeline = None

def get_finbert_pipeline():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        print("🧠 Loading FinBERT model (first-time)...")
        _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _finbert_pipeline

sentiment_cache = {}

async def analyze_sentiment_async(titles):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: get_finbert_pipeline()(titles))

def fallback_vader_score(titles):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(title)['compound'] for title in titles]
    return sum(scores) / len(scores) if scores else 0

def get_news_sentiment(ticker):
    try:
        if ticker in sentiment_cache:
            return sentiment_cache[ticker]["news"]

        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        titles = [article["title"] for article in articles if article.get("title")]
        if not titles:
            return 0

        try:
            sentiments = finbert_pipeline(titles)
            scores = [1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0 for s in sentiments]
            score = sum(scores) / len(scores)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"⚠️ FinBERT failed for NewsAPI: {e}\n{tb}")
            send_discord_alert(f"⚠️ FinBERT fallback to VADER (News): {e}")
            score = fallback_vader_score(titles)

        sentiment_cache.setdefault(ticker, {})["news"] = score
        return score
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ News sentiment error for {ticker}: {e}\n{tb}")
        send_discord_alert(f"⚠️ News sentiment error for {ticker}: {e}")
        return 0

def get_reddit_sentiment(ticker):
    try:
        if ticker in sentiment_cache:
            return sentiment_cache[ticker].get("reddit", 0)

        import praw
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        titles = [submission.title for submission in reddit.subreddit("stocks").search(ticker, limit=10)]
        if not titles:
            return 0

        try:
            sentiments = finbert_pipeline(titles)
            scores = [1 if s['label'] == 'positive' else -1 if s['label'] == 'negative' else 0 for s in sentiments]
            score = sum(scores) / len(scores)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"⚠️ FinBERT failed for Reddit: {e}\n{tb}")
            send_discord_alert(f"⚠️ FinBERT fallback to VADER (Reddit): {e}")
            score = fallback_vader_score(titles)

        sentiment_cache.setdefault(ticker, {})["reddit"] = score
        return score
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Reddit sentiment error for {ticker}: {e}\n{tb}")
        send_discord_alert(f"⚠️ Reddit sentiment error for {ticker}: {e}")
        return 0

# === Push Meta Logs to Google Sheets ===
def push_meta_logs_to_sheets(meta_data):
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GSPREAD_JSON_PATH, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(GSHEET_ID).worksheet("MetaModelLog")

        formatted = [
            [d["timestamp"], d["ticker"], d["score"], d["sentiment"], d["regime"], d["label"]]
            for d in meta_data
        ]
        sheet.append_rows(formatted, value_input_option="RAW")
        print("✅ Meta logs pushed to Google Sheets.")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Failed to push meta logs to Sheets: {e}\n{tb}")
        send_discord_alert(f"⚠️ Google Sheets push failed: {e}")

# === Model Failure Retry Queue ===
model_failure_queue = []
blacklisted_tickers = set()

def handle_model_training_failure(ticker):
    if ticker not in blacklisted_tickers:
        model_failure_queue.append(ticker)
        print(f"🔁 Queued {ticker} for model retraining retry.")
    else:
        print(f"🚫 Skipping {ticker}: blacklisted due to repeated failures.")

def blacklist_ticker(ticker):
    blacklisted_tickers.add(ticker)
    print(f"⛔ Blacklisted {ticker} after repeated failures.")
        
# === Model Staleness and Retraining Logic ===
def needs_retraining(model_path, max_age_hours=24):
    if not os.path.exists(model_path):
        return True
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    return (datetime.now() - mod_time).total_seconds() > max_age_hours * 3600

# === Updated train_model() with scaler and higher max_iter ===
def train_model(ticker, X, y, model_path):
    try:
        if len(X) < 50:
            print(f"⚠️ Not enough data to train {ticker}", flush=True)
            return None

        print(f"🧪 [{ticker}] Starting training with {len(X)} rows", flush=True)
        start = time.time()

        model = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier()),
            ('lr', LogisticRegression(max_iter=1000)),
            ('xgb', XGBClassifier(eval_metric="logloss", verbosity=0))
        ], voting='soft')

        model.fit(X, y)

        print(f"🧪 [{ticker}] Model fit completed in {time.time() - start:.2f}s", flush=True)

        acc = walk_forward_validation(X, y, RandomForestClassifier())
        if acc < 0.5:
            print(f"⚠️ Low validation accuracy for {ticker}: {acc:.2f}", flush=True)
            handle_model_training_failure(ticker)
            return None

        joblib.dump(model, model_path)
        log(f"✅ Trained model for {ticker} with acc={acc:.2f}")
        return model

    except Exception as e:
        handle_model_training_failure(ticker)
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Training failed for {ticker}: {e}\n{tb}", flush=True)
        send_discord_alert(f"⚠️ Model training failed for {ticker}")
        return None

# === Append Meta Logs After Trade Decision ===
def log_meta_decision(ticker, score, sentiment, regime, label):
    meta_data = [{
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "score": score,
        "sentiment": sentiment,
        "regime": regime,
        "label": label
    }]
    push_meta_logs_to_sheets(meta_data)

# === Discord Summary ===
def send_end_of_day_summary(trades, q_logs):
    try:
        account = api.get_account()
        cash = float(account.cash)

        summary = ["📊 **End of Day Summary**"]
        summary.append(f"💰 Account Cash: ${cash:,.2f}")

        if trades:
            summary.append("\n**Top Trades:**")
            for ticker, price, result in trades[:5]:
                summary.append(f"• {ticker} at ${price} ➝ {result}")

        if q_logs:
            avg_q = sum(q for _, _, q, _ in q_logs) / len(q_logs)
            summary.append(f"\n🧠 Avg Q-Value: {avg_q:.4f} from {len(q_logs)} updates")

        payload = {"content": "\n".join(summary)}
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
        print("📬 End of day summary sent to Discord.")
    except Exception as e:
        print(f"❌ Failed to send Discord summary: {e}")

# === Sector Allocation State ===
sector_allocations = {}
MAX_SECTOR_EXPOSURE = 3  # max number of positions per sector

# === End-of-Day Handler ===
def end_of_day_cleanup():
    global sector_allocations
    sector_allocations = {}
    print("🔁 Sector allocations reset for new trading day.")
    print("🌙 Running end-of-day cleanup...")

    trades, q_logs = update_q_network_from_log()
    send_end_of_day_summary(trades, q_logs)
    train_meta_model()

    print("📤 Cleanup complete.")

def check_sector_allocation(ticker):
    try:
        sector = get_sector(ticker)
        count = sector_allocations.get(sector, 0)
        if count >= MAX_SECTOR_EXPOSURE:
            print(f"🛑 Sector limit reached for {sector}")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Sector check failed for {ticker}: {e}")
        return True  # fail-safe

def update_sector_allocation(ticker):
    sector = get_sector(ticker)
    sector_allocations[sector] = sector_allocations.get(sector, 0) + 1

# === Trade Execution Logic (Completed with Logging and Sentiment) ===
def execute_trade(ticker, score):
    if not passes_volume_vwap_filter(ticker):
        print(f"⚠️ Skipping {ticker}: does not pass VWAP/Volume filter.")
        return

    if not check_sector_allocation(ticker):
        print(f"⚠️ Skipping {ticker}: exceeds sector allocation limit.")
        return

    df = get_data_alpaca(ticker)
    support, resistance = calculate_support_resistance(df)
    if support and resistance:
        current_price = df['close'].iloc[-1]
        if current_price < support * 0.98 or current_price > resistance * 1.02:
            print(f"⚠️ {ticker} price outside support/resistance bounds. Skipping.")
            return
    try:
        sentiment_news = get_news_sentiment(ticker)
        sentiment_reddit = get_reddit_sentiment(ticker)
        sentiment_score = (sentiment_news + sentiment_reddit) / 2
        regime = get_market_regime()

        # Meta Model Approval Gate
        meta_features = {
            "ticker": ticker,
            "score": score,
            "sentiment": sentiment_score,
            "regime": regime,
            "hour": datetime.now().hour,
            "dayofweek": datetime.now().weekday(),
        }
        meta_df = pd.DataFrame([meta_features])

        if os.path.exists("meta_model.pkl"):
            try:
                meta_model = joblib.load("meta_model.pkl")
                approve = meta_model.predict(meta_df)[0]
                label = "approved" if approve == 1 else "rejected"

                if approve == 0:
                    print(f"🛑 Meta model rejected trade for {ticker}")
                    log_meta_decision(ticker, score, sentiment_score, regime, label)
                    return
                else:
                    print(f"🧠 Meta model approved trade for {ticker}")
                    log_meta_decision(ticker, score, sentiment_score, regime, label)

            except Exception as e:
                print(f"⚠️ Meta model error: {e}")
                # In case of error, assume "approved" to be conservative
                log_meta_decision(ticker, score, sentiment_score, regime, "approved")

        try:
            position = api.get_position(ticker)
            has_position = True
        except:
            position = None
            has_position = False

        bar = api.get_latest_bar(ticker)
        current_price = bar.c

        if has_position:
            avg_price = float(position.avg_entry_price)
            gain = (current_price - avg_price) / avg_price
            qty_held = int(position.qty)
            held_duration = (datetime.now() - datetime.strptime(position.lastday_price, "%Y-%m-%dT%H:%M:%S.%fZ")).total_seconds() / 60

            if gain >= 0.05:
                api.submit_order(symbol=ticker, qty=qty_held, side="sell", type="market", time_in_force="gtc")
                print(f"🏁 Sold {ticker} at +5% profit | Gain: {gain:.2%}")
                send_discord_alert(f"🏁 Sold {ticker} at +5% profit | Gain: {gain:.2%}")
                with open("trade_outcomes.csv", "a") as f:
                    f.write(f"{datetime.now().isoformat()},{ticker},{current_price},{score},sell,profit,{sentiment_score},{regime}\n")
                return

            if gain <= -0.03:
                api.submit_order(symbol=ticker, qty=qty_held, side="sell", type="market", time_in_force="gtc")
                print(f"🛑 Stop loss hit on {ticker} | Loss: {gain:.2%}")
                send_discord_alert(f"🛑 Stop loss hit on {ticker} | Loss: {gain:.2%}")
                with open("trade_outcomes.csv", "a") as f:
                    f.write(f"{datetime.now().isoformat()},{ticker},{current_price},{score},sell,loss,{sentiment_score},{regime}\n")
                return

            if gain < 0.01 and held_duration > 60:
                api.submit_order(symbol=ticker, qty=qty_held, side="sell", type="market", time_in_force="gtc")
                print(f"💸 Sold {ticker} due to profit decay | Gain: {gain:.2%} | Held: {held_duration:.0f} min")
                send_discord_alert(f"💸 Sold {ticker} due to profit decay | Gain: {gain:.2%} | Held: {held_duration:.0f} min")
                with open("trade_outcomes.csv", "a") as f:
                    f.write(f"{datetime.now().isoformat()},{ticker},{current_price},{score},sell,decay,{sentiment_score},{regime}\n")
                return

            state = q_state(ticker, 1).unsqueeze(0)
            q_val = q_net(state).item()
            if q_val > 0.3 and gain > 0 and score > 0.6:
                print(f"⏸️ RL prefers to hold {ticker} (Q={q_val:.2f}, Gain={gain:.2%})")
                return

        try:
            capital = float(api.get_account().cash)
            win_rate = 0.6
            reward_risk = 1.5
            kelly_fraction = win_rate - (1 - win_rate) / reward_risk
            kelly_fraction = max(0.01, min(kelly_fraction, 0.2))
            dollar_position = capital * kelly_fraction
            qty = max(1, int(dollar_position / current_price))
        except Exception as e:
            print(f"⚠️ Kelly sizing failed, defaulting to 1 share: {e}")
            qty = 1

        if not has_position:
            update_sector_allocation(ticker)
            api.submit_order(symbol=ticker, qty=qty, side="buy", type="market", time_in_force="gtc")
            print(f"✅ Bought {qty} of {ticker} at ${current_price:.2f} | Confidence: {score:.2f}")
            send_discord_alert(f"✅ Bought {qty} of {ticker} at ${current_price:.2f} | Confidence: {score:.2f}")
            with open("trade_outcomes.csv", "a") as f:
                f.write(f"{datetime.now().isoformat()},{ticker},{current_price},{score},buy,,{sentiment_score},{regime}\n")
        else:
            print(f"📌 Already holding {ticker}, skipping buy.")

    except Exception as e:
        print(f"❌ Trade execution failed for {ticker}: {e}")
        send_discord_alert(f"❌ Trade execution failed for {ticker}: {e}")
        
# === Dynamic Watchlist Selection (Updated) ===
def get_dynamic_watchlist():
    print("🧠 Generating dynamic watchlist...", flush=True)
    tradable = FALLBACK_UNIVERSE  # <-- Force fallback universe for IEX testing

    top = []
    sector_counts = {}

    for symbol in random.sample(tradable, 50):  # fewer samples to reduce load
        try:
            df = get_data_alpaca(symbol, limit=30)
            if df is None or df.empty:
                continue

            change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            volume_avg = df['volume'].mean()
            support, resistance = calculate_support_resistance(df)
            current_price = df['close'].iloc[-1]

            print(f"🔍 Trying {symbol} | Change: {change:.2%} | Volume: {volume_avg:.0f}")

            if (
                change > 0
                and volume_avg > 500
                and support and resistance
            ):
                sector = get_sector(symbol)
                if sector_counts.get(sector, 0) >= MAX_PER_SECTOR_WATCHLIST:
                    continue

                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                top.append((symbol, change))

                if len(top) >= 20:
                    break

        except Exception as e:
            print(f"⚠️ Error processing {symbol}: {e}")
            continue

    top.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in top]

def auto_liquidate():
    try:
        positions = api.list_positions()
        if not positions:
            print("📭 No positions to liquidate.")
            return

        for pos in positions:
            ticker = pos.symbol
            qty = int(pos.qty)
            current_price = float(pos.current_price)
            avg_entry_price = float(pos.avg_entry_price)
            gain = (current_price - avg_entry_price) / avg_entry_price
            sentiment = get_news_sentiment(ticker)
            regime = get_market_regime()

            api.submit_order(symbol=ticker, qty=qty, side="sell", type="market", time_in_force="gtc")
            print(f"🔻 Liquidated {qty} shares of {ticker} at ${current_price:.2f}")

            result = "profit" if gain > 0 else "loss"
            with open("trade_outcomes.csv", "a") as f:
                f.write(f"{datetime.now().isoformat()},{ticker},{current_price},AUTO,sell,{result},{sentiment},{regime}\n")

            send_discord_alert(f"🔻 Auto-liquidated {qty} of {ticker} at ${current_price:.2f} | {result.upper()}")

    except Exception as e:
        print(f"❌ Auto-liquidation failed: {e}")
        send_discord_alert(f"❌ Auto-liquidation failed: {e}")

# === Main Loop with Integrated Retraining ===
def run_trading_loop():
    print("🚀 Entered run_trading_loop", flush=True)
    if is_near_market_close():
        print("⏱️ Near market close — skipping trade cycle.", flush=True)
        auto_liquidate()
        return

    used_sectors = set()
    cooldown = {}

    while True:
        now = datetime.now(pacific)
        if now.hour == 13 and now.minute >= 0:
            print("⏹️ Market closing soon, stopping bot.", flush=True)
            send_discord_alert("📉 End of trading day.")
            break

        if now.hour < 6 or (now.hour == 6 and now.minute < 30):
            print("⏳ Waiting for market to open...", flush=True)
            time.sleep(60)
            continue

        tickers = get_dynamic_watchlist()
        print(f"🎯 Evaluating tickers: {tickers}", flush=True)

        for ticker in tickers:
            try:
                print(f"📊 Processing {ticker}", flush=True)
                df_short = get_data(ticker, days=2)
                df_mid = get_data(ticker, days=15)

                if df_short is None or df_short.empty or len(df_short) < 20:
                    print(f"⚠️ Not enough data for {ticker}, skipping.", flush=True)
                    continue

                latest_row = df_short.iloc[-1]

                # Confirm required columns
                if not all(col in df_short.columns for col in ["Close", "Volume", "High", "Low"]):
                    print(f"⚠️ Missing required columns for {ticker}, skipping.", flush=True)
                    continue

                # Volume & VWAP filters
                price_momentum = (df_short['Close'].iloc[-1] - df_short['Close'].iloc[-5]) / df_short['Close'].iloc[-5]
                recent_volume = df_short['Volume'].rolling(20).mean().iloc[-2]
                current_volume = latest_row["Volume"]
                vwap = (df_short['Volume'] * (df_short['High'] + df_short['Low']) / 2).sum() / df_short['Volume'].sum()

                if price_momentum < 0.005 or current_volume < 1.2 * recent_volume or latest_row["Close"] < vwap:
                    print(f"⏩ Skipping {ticker} due to weak momentum or volume.", flush=True)
                    continue

                # Technical filters
                support, resistance = calculate_support_resistance(df_short)
                if support and resistance:
                    current_price = latest_row["Close"]
                    if current_price > resistance or current_price < support:
                        print(f"🛑 {ticker} near extreme S/R level, skipping.")
                        continue

                # Sentiment override
                sentiment_score = get_sentiment_score(ticker)
                if sentiment_score < -0.2:
                    print(f"😡 Negative sentiment for {ticker}, skipping.")
                    continue

                # Train models
                df_short["Target"] = (df_short["Close"].shift(-5) > df_short["Close"]).astype(int)
                X_short = df_short.drop(columns=["Target"])
                y_short = df_short["Target"]
                short_model_path = f"models/short/{ticker}.pkl"
                train_model(ticker, X_short, y_short, short_model_path)

                if df_mid is not None and not df_mid.empty:
                    df_mid["Target"] = (df_mid["Close"].shift(-5) > df_mid["Close"]).astype(int)
                    X_mid = df_mid.drop(columns=["Target"])
                    y_mid = df_mid["Target"]
                    mid_model_path = f"models/medium/{ticker}.pkl"
                    train_model(ticker, X_mid, y_mid, mid_model_path)

                # Predict with ensemble
                proba_short, prediction_short = predict_weighted_proba(ticker, latest_row.drop("Target"))
                if proba_short is None:
                    continue

                # RL HOLD check
                if ticker in get_current_positions():
                    _price = float(get_current_positions()[ticker]["avg_entry_price"])
                    gain = (latest_row["Close"] - _price) / _price
                    state = q_state(ticker, 1).unsqueeze(0)
                    hold_value = q_net(state).item()
                    if hold_value > 0.3 and gain > 0 and proba_short > 0.55:
                        print(f"⏸️ RL prefers to HOLD {ticker} (Q={hold_value:.2f})")
                        continue

                # Execute Buy
                if proba_short > 0.53 and prediction_short == 1:
                    print(f"🚀 BUY confirmed for {ticker}")
                    execute_trade(ticker, prediction_short, proba_short, cooldown, latest_row, df_short)

                # Optional: notify on confident sell
                elif proba_short < 0.45 and prediction_short == 0:
                    print(f"📉 SELL or avoid {ticker}")

                else:
                    print(f"⏸️ HOLD/No action for {ticker}")

                print(f"✅ Done processing {ticker}", flush=True)

            except Exception as e:
                print(f"❌ Error during loop for {ticker}: {e}", flush=True)

        time.sleep(300)
        print("🔄 Looping after 5 min sleep...", flush=True)
        
if __name__ == "__main__":
    print("🟢 main.py started")
    send_discord_alert("✅ Trading bot launched on Render.")

    while not is_market_open():
        print("⏳ Market is closed. Waiting to start trading...")
        time.sleep(60)

    run_trading_loop()
    end_of_day_cleanup()
