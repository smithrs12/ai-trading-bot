# -*- coding: utf-8 -*-
# ai_trading_bot.py

# [FULLY INTEGRATED WITH ALL REQUESTED ENHANCEMENTS]
# Includes: Reinforcement Learning, Market Regime Detection, Sentiment Scoring (Reddit + News),
# Trade Cooldown, Position Sizing, Walk-forward Validation, Persistent Q-table, Discord Alerts,
# Trade Logging, Backtesting Switch (manual), Feature Importance Logging

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import time
import requests
import warnings
import stat
import joblib
import json
import random
import pytz
import finnhub
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ✅ Load environment variables
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")  # e.g. https://paper-api.alpaca.s

# ✅ Initialize Alpaca API
alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ✅ Google Sheets API scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# ✅ Meta model retraining check
def should_retrain_meta_model(max_age_hours=24):
    model_path = "meta_model.pkl"
    if not os.path.exists(model_path):
        return True
    last_modified = os.path.getmtime(model_path)
    age_hours = (time.time() - last_modified) / 3600
    return age_hours > max_age_hours

def retrain_meta_model_from_sheet():
    try:
        sheet = gsheet_meta.open("meta_model_training").sheet1
        records = sheet.get_all_records()
        if len(records) < 10:
            print("⚠️ Not enough data to retrain meta model.")
            return
        df = pd.DataFrame(records)

        # ✅ Add this validation block
        expected_cols = {
            "proba_short", "proba_mid", "sentiment",
            "price_change", "atr", "vwap_diff", "volume_ratio", "final_outcome"
        }
        if not expected_cols.issubset(df.columns):
            print("❌ Meta model training failed: Missing columns in training data")
            send_discord_message("❌ Meta model training failed: Missing columns in training data")
            return

        train_meta_model(df)
        print("✅ Meta model retrained successfully.")
        send_discord_message("🧠 Meta model retrained.")
        
    except Exception as e:
        print(f"❌ Failed to retrain meta model: {e}")
        send_discord_message(f"❌ Meta model retraining failed: {e}")

# Trade log credentials
creds_trade = ServiceAccountCredentials.from_json_keyfile_name("trade_log_credentials.json", scope)
gsheet_trade = gspread.authorize(creds_trade)

# Meta model credentials
creds_meta = ServiceAccountCredentials.from_json_keyfile_name("meta_credentials.json", scope)
gsheet_meta = gspread.authorize(creds_meta)

warnings.filterwarnings("ignore")
load_dotenv()

DEBUG = True
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

if os.path.exists(".env"):
    os.chmod(".env", stat.S_IRUSR | stat.S_IWUSR)

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "NVDA", "AMD",
    "META", "SNAP", "DIS", "TSLA", "F", "RIVN", "CHWY",
    "PLTR", "UBER", "COIN", "SHOP", "INTC", "MARA", "SOFI", "SIRI",
    "CHPT", "OPEN", "PINS", "LCID", "CGC"
]

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "META": "Communication", "SNAP": "Communication", "DIS": "Communication",
    "TSLA": "Consumer", "F": "Consumer", "RIVN": "Consumer", "CHWY": "Consumer",
    "PLTR": "Tech", "UBER": "Transport", "COIN": "Finance", "SHOP": "Tech",
    "INTC": "Semiconductors", "MARA": "Crypto", "SOFI": "Finance", "SIRI": "Media",
    "CHPT": "Energy", "OPEN": "RealEstate", "PINS": "Social", "LCID": "EV", "CGC": "Cannabis"
}

MAX_POSITION_PCT = 0.2
CONFIDENCE_THRESHOLD = 0.8
STOP_LOSS_THRESHOLD = 0.05
TRADE_LOG_FILE = "trade_history.csv"
MODEL_DIR = "models"
FEATURE_DIR = "feature_importance"
TRADE_CACHE_FILE = "last_trades.json"
Q_TABLE_FILE = "q_table.json"
BACKTESTING = False

def is_market_open():
    try:
        clock = alpaca.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"⚠️ Market status check failed: {e}")
        return False
     
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(FEATURE_DIR): os.makedirs(FEATURE_DIR)

def is_premarket_risk_period():
    now = datetime.now().time()
    market_open = dt_time(6, 30)  # 6:30 AM PT
    market_cutoff = dt_time(6, 40)
    return market_open <= now < market_cutoff

def log_trade_to_gsheet(ts, ticker, action, qty, price):
    try:
        sheet = gsheet_trade.open("trade_history").sheet1  # Update sheet name if needed
        sheet.append_row([ts, ticker, action, qty, price])
    except Exception as e:
        print(f"⚠️ Failed to log trade to Google Sheets: {e}")

def log_meta_model_metrics(ticker, acc, prec, rec, date_str=None):
    try:
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet = gsheet_meta.open("meta_model_log").sheet1  # Update sheet name if needed
        sheet.append_row([date_str, ticker, acc, prec, rec])
    except Exception as e:
        print(f"⚠️ Failed to log meta model metrics: {e}")

def log_meta_training_row(features_dict):
    try:
        sheet = gsheet_meta.open("meta_model_training").sheet1
        sheet.append_row([features_dict[k] for k in features_dict])
    except Exception as e:
        print(f"⚠️ Failed to log meta training row: {e}")

def train_meta_model(df):
    features = df.drop(columns=["final_outcome"])
    labels = df["final_outcome"]
    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(features, labels)
    joblib.dump(model, "meta_model.pkl")

def log_trade(ts, ticker, action, qty, price):
    account = api.get_account()
    row = pd.DataFrame([[
        ts, ticker, action, qty, price, account.buying_power, account.equity
    ]], columns=[
        "timestamp", "ticker", "action", "qty", "price", "buying_power", "equity"
    ])
    row.to_csv(TRADE_LOG_FILE,
               mode='a',
               header=not os.path.exists(TRADE_LOG_FILE),
               index=False)

    # Also log to Google Sheet
    log_trade_to_gsheet(ts, ticker, action, qty, price)

def log_pnl(ticker, qty, price, direction, entry_price, model_type):
    pnl = (price - entry_price) * qty if direction == "SELL" else 0
    row = pd.DataFrame([[
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ticker,
        direction,
        qty,
        price,
        entry_price,
        pnl,
        model_type
    ]], columns=[
        "timestamp", "ticker", "direction", "qty", "price", "entry_price", "pnl", "model_type"
    ])
    row.to_csv("pnl_tracker.csv", mode='a', header=not os.path.exists("pnl_tracker.csv"), index=False)

def send_discord_message(msg):
    if DISCORD_WEBHOOK_URL:
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        except:
            pass

def retry_submit_order(symbol, qty, side, max_attempts=3, delay=3):
    attempt = 1
    while attempt <= max_attempts:
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            print(f"✅ Trade submitted for {symbol} on attempt {attempt}")
            return True
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for {symbol}: {e}")
            if attempt == max_attempts:
                send_discord_message(f"❌ Failed to execute trade for {symbol} after {max_attempts} attempts.")
                return False
            time.sleep(delay)
            attempt += 1

def get_market_regime():
    spy = get_data("SPY", days=7)  # Limit to 7 days to avoid 1m restriction
    if spy is None:
        print("⚠️ Failed to load SPY data. Using last known regime.")
        return regime_cache["type"]

    ma20 = spy["Close"].rolling(20).mean()
    volatility = spy["Close"].pct_change().rolling(20).std()

    if spy.Close.iloc[-1] > ma20.iloc[-1] and volatility.iloc[-1] < 0.02:
        regime = "bull"
    elif spy.Close.iloc[-1] < ma20.iloc[-1] and volatility.iloc[-1] > 0.03:
        regime = "bear"
    else:
        regime = "sideways"

    regime_cache["last"] = datetime.now()
    regime_cache["type"] = regime

    print(f"📉 Current market regime: {regime}")
    return regime

# Q-Table Logic
q_table = json.load(open(Q_TABLE_FILE)) if os.path.exists(Q_TABLE_FILE) else {}

def reward_function(action, entry_price=None, exit_price=None):
    if action == 1:
        return 0  # No reward at buy time
    elif action == 0 and entry_price is not None and exit_price is not None:
        return (exit_price - entry_price) / entry_price
    return -1  # Penalty for invalid input
    
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

q_net = QNetwork()
q_optimizer = optim.Adam(q_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def q_state(ticker, action):
    return torch.tensor([
        hash(ticker) % 100000 / 100000.0,  # Normalize ticker hash
        float(action)
    ], dtype=torch.float32)

def update_q_nn(ticker, action, reward):
    state = q_state(ticker, action).unsqueeze(0)
    predicted = q_net(state)
    target = torch.tensor([[reward]], dtype=torch.float32)
    loss = loss_fn(predicted, target)
    q_optimizer.zero_grad()
    loss.backward()
    q_optimizer.step()
    q_table[ticker] = q_net(state).item()

# Regime Detection
regime_cache = {"last": None, "type": "unknown"}
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw

# Set up the sentiment analyzer and News API client (these must come first)
analyzer = SentimentIntensityAnalyzer()
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
reddit = praw.Reddit(client_id=os.getenv("REDDIT_CLIENT_ID"),
                     client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                     user_agent=os.getenv("REDDIT_USER_AGENT"))


# ✅ Add this new helper function BEFORE get_sentiment_score()
def get_news_sentiment(ticker):
    try:
        url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', [])]
        if not headlines:
            raise ValueError("No headlines found.")
        scores = [
            analyzer.polarity_scores(title)["compound"] for title in headlines
        ]
        return np.mean(scores)
    except Exception as e:
        if "No headlines found." not in str(e):
            print(f"⚠️ News sentiment error for {ticker}: {e}")
    return 0


# ✅ Your existing combined sentiment scoring function
def get_sentiment_score(ticker):
    score = 0
    count = 0

    # --- News Sentiment ---
    try:
        score += get_news_sentiment(ticker)
        count += 1
    except Exception as e:
        print(f"⚠️ News sentiment error for {ticker}: {e}")

    # --- Reddit Sentiment ---
    try:
        subreddit = reddit.subreddit("stocks")
        for submission in subreddit.search(ticker, limit=5):
            score += analyzer.polarity_scores(submission.title)["compound"]
            count += 1
    except Exception as e:
        print(f"⚠️ Reddit sentiment error for {ticker}: {e}")

    return score / count if count > 0 else 0

def get_risk_events(ticker):
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        events = []

        # Earnings calendar
        earnings = finnhub_client.earnings_calendar(_from=today, to=today)
        for event in earnings.get("earningsCalendar", []):
            if event.get("symbol") == ticker:
                events.append("Earnings Today")

        # Analyst ratings
        ratings = finnhub_client.recommendation_trends(ticker)
        if ratings:
            latest = ratings[0]
            if latest.get("sell", 0) > latest.get("buy", 0):
                events.append("Bearish Analyst Rating")

        return events
    except Exception as e:
        print(f"⚠️ Risk event check failed for {ticker}: {e}")
        return []

def is_high_risk_news_day():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        if os.path.exists("risk_events.json"):
            with open("risk_events.json") as f:
                data = json.load(f)
            return today in data.get("high_risk_days", [])
    except Exception as e:
        print(f"⚠️ Failed to check news suppression: {e}")
    return False
        
def calculate_vwap(df):
    try:
        pv = df["Close"] * df["Volume"]
        cumulative_pv = pv.cumsum()
        cumulative_volume = df["Volume"].cumsum()
        df["vwap"] = cumulative_pv / cumulative_volume
        return df
    except Exception as e:
        print(f"⚠️ VWAP calculation failed: {e}")
        return df

def get_data(ticker, days=2, interval='1Min'):
    try:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)

        barset = alpaca.get_bars(
            ticker,
            timeframe=interval,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            adjustment='raw',
            limit=None
        ).df

        if barset.empty or ticker not in barset.index.get_level_values(0):
            print(f"⚠️ No Alpaca data for {ticker}")
            return None

        df = barset[barset.index.get_level_values(0) == ticker].copy()
        df.index = df.index.tz_localize(None)
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })

        # Calculate features
        df["sma"] = df["Close"].rolling(14).mean()
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        avg_gain = up.rolling(14).mean()
        avg_loss = down.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_diff"] = df["macd"].diff()

        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["stoch"] = 100 * (df["Close"] - low14) / (high14 - low14)

        tr1 = df["High"] - df["Low"]
        tr2 = abs(df["High"] - df["Close"].shift())
        tr3 = abs(df["Low"] - df["Close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek

        df["bb_bbm"] = df["Close"].rolling(20).mean()
        df["vwap"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

        df.dropna(inplace=True)
        return df

    except Exception as e:
        print(f"❌ Error fetching Alpaca data for {ticker}: {e}")
        return None

def load_trade_cache():
    return json.load(
        open(TRADE_CACHE_FILE)) if os.path.exists(TRADE_CACHE_FILE) else {}


def save_trade_cache(cache):
    json.dump(cache, open(TRADE_CACHE_FILE, "w"))

def kelly_position_size(prob, price, equity, atr=None, ref_atr=0.5):
    edge = 2 * prob - 1
    if edge <= 0 or atr is None or atr == 0:
        return 0

    kelly_fraction = min(edge, MAX_POSITION_PCT)
    base_allocation = equity * kelly_fraction
    base_size = base_allocation // price

    # Volatility adjustment (scale down if ATR is high)
    atr_adjustment = ref_atr / atr
    adjusted_size = int(base_size * atr_adjustment)
    return max(adjusted_size, 0)
    
def is_model_stale(ticker, max_age_hours=6):
    path = os.path.join(MODEL_DIR, f"{ticker}.pkl")
    if not os.path.exists(path):
        return True
    last_modified = os.path.getmtime(path)
    age_hours = (time.time() - last_modified) / 3600
    return age_hours > max_age_hours

def train_model(ticker, df):
    df["future_return"] = df["Close"].shift(-3) / df["Close"] - 1
    df["target"] = (df["future_return"] > 0.01).astype(int)
    df.dropna(inplace=True)

    features = [
        "sma", "rsi", "macd", "macd_diff", "stoch", "atr", "bb_bbm",
        "hour", "minute", "dayofweek"
    ]
    X, y = df[features], df["target"]

    if len(X) < 60 or y.nunique() < 2:
        return None, None

    model = VotingClassifier(estimators=[
        ('xgb', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
        ('log', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100))
    ], voting='soft', weights=[3, 1, 2])

    model.fit(X, y)
    return model, features

def predict_weighted_proba(models, weights, X):
    total_weight = sum(weights)
    if total_weight == 0 or X is None or X.empty:
        print("⚠️ Invalid input to predict_weighted_proba: empty features or zero total weight.")
        return 0.5

    try:
        probs = []
        for model in models:
            prob = model.predict_proba(X)[0][1]
            probs.append(prob)

        weighted_avg = sum(w * p for w, p in zip(weights, probs)) / total_weight
        return weighted_avg

    except Exception as e:
        print(f"❌ Weighted proba prediction failed: {e}")
        return 0.5  # fallback confidence

def dual_horizon_predict(ticker, model, features, short_days=2, mid_days=15):
    df_short = get_data(ticker, days=short_days)
    df_mid = get_data(ticker, days=mid_days)

    required_cols = set(features + ["Close"])

    for label, df in [("short", df_short), ("mid", df_mid)]:
        if df is None or len(df) < 10:
            print(f"⚠️ {label} timeframe data for {ticker} is missing or too short.")
            return None, None, None
        if not required_cols.issubset(df.columns):
            print(f"⚠️ {label} timeframe data for {ticker} missing required columns: {required_cols - set(df.columns)}")
            return None, None, None

    X_short = df_short[features].iloc[-1:]
    X_mid = df_mid[features].iloc[-1:]

    try:
        if isinstance(model, VotingClassifier) and hasattr(model, "estimators") and hasattr(model, "weights"):
            models = [est for _, est in model.estimators]
            weights = model.weights
            proba_short = predict_weighted_proba(models, weights, X_short)
            proba_mid = predict_weighted_proba(models, weights, X_mid)
        else:
            proba_short = model.predict_proba(X_short)[0][1]
            proba_mid = model.predict_proba(X_mid)[0][1]

        return proba_short, proba_mid, df_short.iloc[-1]

    except Exception as e:
        print(f"⚠️ Dual horizon prediction failed for {ticker}: {e}")
        return None, None, None

def predict(ticker, model, features):
    df = get_data(ticker, days=2)

    required_cols = set(features + ["Close"])
    if df is None or len(df) < 10:
        print(f"⚠️ Not enough data for {ticker}.")
        return 0, None, None
    if not required_cols.issubset(df.columns):
        print(f"⚠️ Missing required columns for {ticker}: {required_cols - set(df.columns)}")
        return 0, None, None

    X = df[features].iloc[-1:]

    try:
        if isinstance(model, VotingClassifier) and hasattr(model, "estimators") and hasattr(model, "weights"):
            models = [est for _, est in model.estimators]
            weights = model.weights
            proba = predict_weighted_proba(models, weights, X)
        else:
            proba = model.predict_proba(X)[0][1]

        return int(proba > 0.5), df.iloc[-1], proba
    except Exception as e:
        print(f"⚠️ Prediction failed for {ticker}: {e}")
        return 0, None, None

def is_medium_model_stale(ticker, max_age_hours=24):
    path = os.path.join("models_medium", f"{ticker}_medium.pkl")
    if not os.path.exists(path):
        return True
    last_modified = os.path.getmtime(path)
    age_hours = (time.time() - last_modified) / 3600
    return age_hours > max_age_hours
        
def train_medium_model(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)
    if len(df) < 90:
        print(f"⚠️ Not enough daily data to train medium-term model for {ticker}")
        return None, None
        
    df["return_5d"] = df["Close"].pct_change(5).shift(-5)
    df["target"] = (df["return_5d"] > 0.02).astype(int)
    df.dropna(inplace=True)

    features = ["Open", "High", "Low", "Close", "Volume"]
    X, y = df[features], df["target"]
    if len(X) < 60:
        print(f"⚠️ Not enough training samples for {ticker} (medium-term): {len(X)} rows.")
        return None, None
    if y.nunique() < 2:
        print(f"⚠️ Target lacks diversity for {ticker} (medium-term). Only found class: {y.unique()}")
        return None, None

    xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    log_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100)

    ensemble = VotingClassifier(estimators=[
        ('xgb', xgb_model),
        ('log', log_model),
        ('rf', rf_model)
    ], voting='soft', weights=[3, 1, 2])

    tscv = TimeSeriesSplit(n_splits=5)
    accs, precs, recs = [], [], []

    for train_idx, test_idx in tscv.split(X):
        ensemble.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = ensemble.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], y_pred))
        precs.append(precision_score(y.iloc[test_idx], y_pred, zero_division=0))
        recs.append(recall_score(y.iloc[test_idx], y_pred, zero_division=0))

    print(f"📈 [MEDIUM] {ticker} | Acc: {np.mean(accs):.3f} | Prec: {np.mean(precs):.3f} | Rec: {np.mean(recs):.3f}")
 
    log_meta_model_metrics(ticker, np.mean(accs), np.mean(precs), np.mean(recs))

    model_path = os.path.join("models_medium", f"{ticker}_medium.pkl")
    os.makedirs("models_medium", exist_ok=True)
    joblib.dump(ensemble, model_path)

    # ✅ Log medium-term model performance
    perf_log = {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "medium",
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "samples": len(X)
    }
    perf_df = pd.DataFrame([perf_log])
    perf_file = os.path.join("model_performance.csv")
    perf_df.to_csv(perf_file, mode='a', header=not os.path.exists(perf_file), index=False)

    return ensemble, features

def ensure_models_trained(tickers):
    for ticker in tickers:
        try:
            # ----- SHORT-TERM MODEL -----
            model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")
            needs_short = is_model_stale(ticker) or not os.path.exists(model_path)
            if needs_short:
                df = get_data(ticker, days=2)
                if df is not None:
                    model, features = train_model(ticker, df)
                    if model:
                        joblib.dump(model, model_path)
                        print(f"✅ Trained and saved short-term model for {ticker}")
                    else:
                        print(f"⚠️ Failed to train short-term model for {ticker}")

            # ----- MEDIUM-TERM MODEL -----
            med_path = os.path.join("models_medium", f"{ticker}_medium.pkl")
            needs_med = is_medium_model_stale(ticker) or not os.path.exists(med_path)
            if needs_med:
                model, _ = train_medium_model(ticker)
                if model:
                    print(f"✅ Trained and saved medium-term model for {ticker}")
                else:
                    print(f"⚠️ Failed to train medium-term model for {ticker}")

        except Exception as e:
            print(f"❌ Model pretraining error for {ticker}: {e}")

def predict_medium_term(ticker):
    model_path = os.path.join("models_medium", f"{ticker}_medium.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Medium-term model not found for {ticker}. Attempting to train.")
        model, features = train_medium_model(ticker)
        if model is None:
            return None
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        df.dropna(inplace=True)
        if len(df) < 90:
            return None

        features = ["Open", "High", "Low", "Close", "Volume"]
        X = df[features].iloc[-1:]

        if isinstance(model, VotingClassifier) and hasattr(model, "estimators") and hasattr(model, "weights"):
            models = [est for name, est in model.estimators]
            weights = model.weights
            proba = predict_weighted_proba(models, weights, X)
        else:
            proba = model.predict_proba(X)[0][1]

        return proba

    except Exception as e:
        print(f"⚠️ Medium-term prediction error for {ticker}: {e}")
        return None

def execute_trade(ticker, prediction, proba, proba_mid, latest_row, df):
    global cooldown
    try:
        position = None
        try:
            position = api.get_position(ticker)
        except:
            pass

        current_price = api.get_latest_bar(ticker).c
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        equity = float(api.get_account().equity)
        max_dollars = equity * MAX_POSITION_PCT
        atr = latest_row["atr"]
        price_change = latest_row["Close"] - latest_row["Open"]
        sentiment = get_sentiment_score(ticker)
        qty = kelly_position_size(proba, current_price, equity, atr=atr, ref_atr=0.5)

        # ---- RL-driven HOLD logic ----

        if position:
            _price = float(position.avg_entry_price)
            gain = (current_price - _price) / _price
            state = q_state(ticker, 1).unsqueeze(0)
            hold_value = q_net(state).item()
            if hold_value > 0.3 and gain > 0 and proba > 0.55:
                print(f"⏸️ RL prefers to hold {ticker} (Q={hold_value:.2f})")
                return

        # ---- Hybrid Cooldown Logic ----
        if ticker in cooldown:
            last_trade = cooldown[ticker]
            last_time = datetime.strptime(last_trade["timestamp"], "%Y-%m-%d %H:%M:%S")
            last_conf = last_trade.get("confidence", 0.5)
            seconds_since = (datetime.now() - last_time).total_seconds()

            # Hard block for first 5 minutes
            if seconds_since < 300:
                print(f"⏳ Cooldown: Hard block active for {ticker} ({int(seconds_since)}s elapsed)")
                return

            # Soft cooldown decay for re-entry
            decay_factor = max(0.1, 1 - seconds_since / 900)  # Decays over 15 mins
            adjusted_proba = proba * decay_factor
            if adjusted_proba < 0.6:
                print(f"⏳ Cooldown: Adjusted confidence too low ({adjusted_proba:.2f}) for {ticker}. Skipping.")
                return

        # ---- BUY logic ----
        if prediction == 1 and qty > 0 and (not position or float(position.market_value) < max_dollars):

            # ---- Meta Model Approval ----
            try:
                if os.path.exists("meta_model.pkl"):
                    import xgboost as xgb
                    meta_model = joblib.load("meta_model.pkl")
                    volume_ratio = 1  # fallback
                    if len(df) >= 20:
                        try:
                            volume_ratio = latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2]
                        except Exception as e:
                            print(f"⚠️ Volume ratio calc failed for {ticker}: {e}")

                    meta_features = pd.DataFrame([{
                        "proba_short": proba,
                        "proba_mid": proba_mid,
                        "sentiment": sentiment,
                        "price_change": price_change,
                        "atr": atr,
                        "vwap_diff": latest_row.get("Close", 0) - latest_row.get("vwap", latest_row.get("Close", 0)),
                        "volume_ratio": volume_ratio,
                    }])

                    meta_pred = meta_model.predict(meta_features)[0]
                    if meta_pred == 0:
                        print(f"⛔ Meta model vetoed the trade for {ticker}. Skipping.")
                        return
            except Exception as e:
                print(f"⚠️ Meta model check failed for {ticker}: {e}")

            # ✅ Only runs if no veto or exception
            if not retry_submit_order(ticker, qty, "buy"):
                return

            send_discord_message(
                f"🟢 Bought {qty} shares of {ticker} at ${current_price:.2f} (Conf: {proba:.2f} Sentiment: {sentiment:+.2f})"
            )

            log_trade(timestamp, ticker, "BUY", qty, current_price)
            cooldown[ticker] = {
                "timestamp": timestamp,
                "confidence": float(min(proba + 0.1, 1.0))
            }
            log_pnl(ticker, qty, current_price, "BUY", current_price, "short")
            update_q_nn(ticker, 1, reward_function(1, proba - 0.5))
         
            # ---- Meta Model Logging ----
            try:
                outcome = 1 if prediction == 1 else 0  # This could later be replaced by true PnL outcome
                meta_log = {
                    "proba_short": proba,
                    "proba_mid": proba_mid,
                    "sentiment": sentiment,
                    "price_change": price_change,
                    "atr": atr,
                    "vwap_diff": latest_row.get("Close", 0) - latest_row.get("vwap", latest_row.get("Close", 0)),
                    "volume_ratio": latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2],
                    "final_outcome": outcome
                }
                log_meta_training_row(meta_log)
            except Exception as e:
                print(f"⚠️ Failed to log meta training row for {ticker}: {e}")

        # ---- ADDITIONAL BUY logic (Pyramiding) ----
        pyramiding_count = cooldown.get(ticker, {}).get("adds", 0)

        if (
            prediction == 1 and 
            position and 
            float(position.market_value) < max_dollars and 
            proba > 0.8 and 
            proba_mid > 0.8 and 
            price_change > 0 and 
            pyramiding_count < 2
        ):
            additional_qty = kelly_position_size(proba, current_price, equity, atr=atr, ref_atr=0.5)
            if additional_qty > 0:

                # ---- Meta Model Approval ----
                try:
                    if os.path.exists("meta_model.pkl"):
                        import xgboost as xgb
                        meta_model = joblib.load("meta_model.pkl")
                        volume_ratio = 1  # fallback
                        if len(df) >= 20:
                            try:
                                volume_ratio = latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2]
                            except Exception as e:
                                print(f"⚠️ Volume ratio calc failed for {ticker}: {e}")

                        meta_features = pd.DataFrame([{
                            "proba_short": proba,
                            "proba_mid": proba_mid,
                            "sentiment": sentiment,
                            "price_change": price_change,
                            "atr": atr,
                            "vwap_diff": latest_row.get("Close", 0) - latest_row.get("vwap", latest_row.get("Close", 0)),
                            "volume_ratio": volume_ratio,
                        }])

                        meta_pred = meta_model.predict(meta_features)[0]
                        if meta_pred == 0:
                            print(f"⛔ Meta model vetoed the trade for {ticker}. Skipping.")
                            return
                except Exception as e:
                    print(f"⚠️ Meta model check failed for {ticker}: {e}")

                # ✅ Only runs if no veto or exception
                if not retry_submit_order(ticker, additional_qty, "buy"):
                    return

                send_discord_message(
                    f"🔼 Added {additional_qty} more shares to {ticker} at ${current_price:.2f} (Strong Dual Horizon Signal)"
                )
                log_trade(timestamp, ticker, "BUY_MORE", additional_qty, current_price)
                log_pnl(ticker, additional_qty, current_price, "BUY", current_price, "short")
                cooldown[ticker] = {
                    "timestamp": timestamp,
                    "confidence": float(min(proba + 0.1, 1.0)),
                    "adds": pyramiding_count + 1
                }
                update_q_nn(ticker, 1, reward_function(1, (proba - 0.5) * 1.5))

                # ---- Meta Model Logging ----
                try:
                    outcome = 1 if prediction == 1 else 0
                    meta_log = {
                        "proba_short": proba,
                        "proba_mid": proba_mid,
                        "sentiment": sentiment,
                        "price_change": price_change,
                        "atr": atr,
                        "vwap_diff": latest_row.get("Close", 0) - latest_row.get("vwap", latest_row.get("Close", 0)),
                        "volume_ratio": latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2],
                        "final_outcome": outcome
                    }
                    log_meta_training_row(meta_log)
                except Exception as e:
                    print(f"⚠️ Failed to log meta training row for {ticker}: {e}")
         
        # ---- SELL logic ----
        if prediction == 0 and position:
            try:
                _price = float(position.avg_entry_price)
                regime = get_market_regime()
                volatility = latest_row.get("atr", 0.5)
                confidence_factor = proba

                base_stop_loss = 1.2 * volatility / current_price
                base_profit_target = 2.5 * volatility / current_price

                if regime == "bull":
                    stop_loss_pct = base_stop_loss * (1 - confidence_factor * 0.3)
                    profit_take_pct = base_profit_target * (1 + confidence_factor * 0.5)
                elif regime == "bear":
                    stop_loss_pct = base_stop_loss * (1 + (1 - confidence_factor) * 0.4)
                    profit_take_pct = base_profit_target * (1 - (1 - confidence_factor) * 0.3)
                else:
                    stop_loss_pct = base_stop_loss
                    profit_take_pct = base_profit_target

                stop_loss_pct = min(max(stop_loss_pct, 0.01), 0.07)
                profit_take_pct = min(max(profit_take_pct, 0.03), 0.12)

                stop_loss_price = _price * (1 - stop_loss_pct)
                profit_target_price = _price * (1 + profit_take_pct)
                gain = (current_price - _price) / _price

                # Profit target reached
                if current_price >= profit_target_price and proba < 0.6:
                    if not retry_submit_order(ticker, int(position.qty), "sell"):
                        return
                    send_discord_message(f"💰 Took profit on {position.qty} shares of {ticker} at ${current_price:.2f} (Gain: {gain:.2%})")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, 0.5 - proba))
                    return

                # Dynamic Trailing Stop
                trailing_atr_factor = 1.5 if gain < 0.05 else 2.5
                trailing_stop_price = current_price - (volatility * trailing_atr_factor)
                if current_price < trailing_stop_price:
                    if not retry_submit_order(ticker, int(position.qty), "sell"):
                        return
                    send_discord_message(f"🔻 {ticker} hit dynamic trailing stop at ${current_price:.2f}.")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, 0.5 - proba))
                    return

                # Profit Decay
                time_held_minutes = 0
                entry_time_str = cooldown.get(ticker, {}).get("timestamp")
                if entry_time_str:
                    time_held_minutes = (datetime.now() - datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60

                gain_pct = (current_price - _price) / _price
                if 0 < gain_pct < 0.02 and time_held_minutes > 60:
                    if not retry_submit_order(ticker, int(position.qty), "sell"):
                        return
                    send_discord_message(f"📉 Exiting {ticker} due to fading profits ({gain_pct:.2%}) after {time_held_minutes:.0f} mins.")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, _price, current_price))

            except Exception as e:
                print(f"⚠️ SELL logic failed for {ticker}: {e}")

        return  # end of execute_trade()

    except Exception as e:
        print(f"🚨 execute_trade crashed for {ticker}: {e}")
        send_discord_message(f"🚨 Trade failed for {ticker}: {e}")

def get_dynamic_watchlist(limit=8):
    tickers_to_scan = [
        "F", "SOFI", "SIRI", "MARA", "PLTR", "INTC", "CHPT", "OPEN", "RIVN",
        "LCID", "PINS", "MIRM", "FROG", "CHWY", "UBER", "MRVL", "CGC", "AAPL",
        "NVDA", "AMD", "TSLA", "AMZN", "META", "MSFT", "GOOG", "COIN", "SHOP",
        "SNAP", "DIS", "T"
    ]

    win_rates = {}
    try:
        if os.path.exists(TRADE_LOG_FILE):
            df = pd.read_csv(TRADE_LOG_FILE)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            for ticker in tickers_to_scan:
                trades = df[df["ticker"] == ticker]
                buys = trades[trades["action"].str.contains("BUY")]
                sells = trades[trades["action"] == "SELL"]
                if not buys.empty and not sells.empty:
                    avg_buy = (buys["qty"] * buys["price"]).sum() / buys["qty"].sum()
                    avg_sell = (sells["qty"] * sells["price"]).sum() / sells["qty"].sum()
                    if avg_sell > avg_buy:
                        win_rates[ticker] = win_rates.get(ticker, 0) + 1
                    else:
                        win_rates[ticker] = win_rates.get(ticker, 0)

        ranked = sorted(tickers_to_scan, key=lambda t: win_rates.get(t, 0), reverse=True)
        return ranked[:limit]
    except Exception as e:
        print(f"⚠️ Failed to tune watchlist: {e}")
        return tickers_to_scan[:limit]

def send_end_of_day_summary():
    if not os.path.exists(TRADE_LOG_FILE):
        send_discord_message("📉 No trades executed today.")
        return

    try:
        df = pd.read_csv(TRADE_LOG_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        today = datetime.utcnow().date()
        df_today = df[df["timestamp"].dt.date == today]

        if df_today.empty:
            send_discord_message("📉 No trades executed today.")
            return

        profit = 0
        wins = 0
        losses = 0
        trades = []

        for ticker in df_today["ticker"].unique():
            buys = df_today[(df_today["ticker"] == ticker) & (df_today["action"].str.contains("BUY"))]
            sells = df_today[(df_today["ticker"] == ticker) & (df_today["action"] == "SELL")]

            if not buys.empty and not sells.empty:
                avg_buy = (buys["qty"] * buys["price"]).sum() / buys["qty"].sum()
                avg_sell = (sells["qty"] * sells["price"]).sum() / sells["qty"].sum()
                qty_sold = sells["qty"].sum()
                pl = (avg_sell - avg_buy) * qty_sold
                profit += pl
                trades.append(f"{ticker}: ${pl:.2f}")

                if pl > 0:
                    wins += 1
                else:
                    losses += 1

        total_trades = wins + losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        try:
            account = api.get_account()
            portfolio_value = float(account.portfolio_value)
        except:
            portfolio_value = "N/A"

        summary = f"📊 End of Day Summary ({today}):\n"
        summary += "\n".join(trades) + "\n"
        summary += f"\n💰 Total P/L: ${profit:.2f}"
        summary += f"\n📈 Portfolio Value: ${portfolio_value}"
        summary += f"\n✅ Win Rate: {wins}/{total_trades} ({win_rate:.1f}%)"

        send_discord_message(summary)

    except Exception as e:
        print(f"❌ Failed to send EOD summary: {e}")

def liquidate_positions():
    try:
        positions = api.list_positions()
        for pos in positions:
            symbol = pos.symbol
            qty = int(float(pos.qty))
            if qty > 0:
                try:
                    retry_submit_order(symbol, qty, "sell")
                    send_discord_message(f"💣 Liquidated {qty} shares of {symbol}")
                    log_trade(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, "SELL", qty, float(pos.current_price))
                except Exception as e:
                    print(f"⚠️ Failed to liquidate {symbol}: {e}")
    except Exception as e:
        print(f"❌ Failed to list positions for liquidation: {e}")

cooldown = load_trade_cache()

def detect_support_resistance(df, window=20, tolerance=0.01):
    """
    Detects whether the latest price is near support or resistance levels.
    """
    if len(df) < window:
        return False, False

    recent = df.tail(window)
    close = recent["Close"]
    support = close.min()
    resistance = close.max()

    last_price = close.iloc[-1]
    near_support = abs(last_price - support) / support < tolerance
    near_resistance = abs(last_price - resistance) / resistance < tolerance

    return near_support, near_resistance

# ✅ Pre-train all required models before starting trading loop
TICKERS = get_dynamic_watchlist(limit=8)
ensure_models_trained(TICKERS)

while True:
    try:
        if is_market_open():
            print("🔁 Trading cycle...", flush=True)
            now = datetime.now().time()

            # 🚫 Block trades for first 30 minutes after market open (6:30–7:00 AM PT)
            if dt_time(6, 30) <= now < dt_time(7, 0):
                print("🛑 No trades allowed during first 30 minutes after market open.")
                time.sleep(60)
                continue

            if is_premarket_risk_period():
                print("🛑 Suppressing trades during first 10 minutes of market open.")
                time.sleep(60)
                continue

            if should_retrain_meta_model():
                retrain_meta_model_from_sheet()

            trade_count = 0
            regime = get_market_regime()
            used_sectors = set()
            trade_candidates = []
            model, features = None, None
            TICKERS = get_dynamic_watchlist(limit=8)
            print(f"🔎 Watchlist for this cycle: {TICKERS}")

            # ✅ Indent this block!
            for ticker in TICKERS:
                try:
                    df = get_data(ticker, days=2)
                    if df is None:
                        continue

                    if is_model_stale(ticker):
                        model, features = train_model(ticker, df)
                        if model is None:
                            continue
                        joblib.dump(model, os.path.join(MODEL_DIR, f"{ticker}.pkl"))
                    else:
                        model = joblib.load(os.path.join(MODEL_DIR, f"{ticker}.pkl"))
                        features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [
                            "sma", "rsi", "macd", "macd_diff", "stoch", "atr", "bb_bbm", "hour", "minute", "dayofweek"
                        ]

                    proba_short, proba_mid, latest_row = dual_horizon_predict(ticker, model, features)
                    if proba_short is None or proba_mid is None or latest_row is None:
                        continue

                    sentiment = get_sentiment_score(ticker)
                    df_volume_avg = df["Volume"].rolling(20).mean().iloc[-2] if len(df) >= 20 else 1
                    score = (
                        proba_short * 100 +
                        sentiment * 10 -
                        latest_row.get("atr", 0.5) * 5 +
                        (latest_row.get("Volume", 1) / df_volume_avg) * 2
                    )

                    sector = SECTOR_MAP.get(ticker, "Unknown")
                    prediction = int(proba_short > 0.5)
                    trade_candidates.append((ticker, score, model, features, latest_row, proba_short, proba_mid, prediction, sector))

                except Exception as e:
                    print(f"⚠️ Failed to process {ticker}: {e}")

            # ✅ These also stay indented inside the same try block
            trade_candidates = sorted(trade_candidates, key=lambda x: x[1], reverse=True)

            for cand in trade_candidates[:5]:
                try:
                    ticker, score, model, features, latest_row, proba_short, proba_mid, prediction, sector = cand
                    df = get_data(ticker, days=2)

                    if df is None or latest_row is None:
                        print(f"⚠️ Missing data for {ticker}, skipping.")
                        continue

                    last_candles = df.tail(5)
                    green_candles = (last_candles["Close"] > last_candles["Open"]).sum()
                    if green_candles < 3:
                        print(f"⛔ {ticker} lacks strong recent price action ({green_candles}/5 green). Skipping.")
                        continue

                    recent_volume = df["Volume"].rolling(20).mean().iloc[-2]
                    current_volume = latest_row["Volume"]
                    if current_volume < 1.5 * recent_volume:
                        print(f"⏸️ {ticker} volume not spiking. Skipping.")
                        continue

                    if latest_row.get("vwap") and latest_row["Close"] < latest_row["vwap"]:
                        print(f"⏸️ {ticker} below VWAP. Skipping.")
                        continue

                    last_ts = cooldown.get(ticker, {}).get("timestamp")
                    if last_ts:
                        elapsed = (datetime.now() - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
                        if elapsed < 600:
                            decay = 1 - (elapsed / 600)
                            proba_short *= decay
                            print(f"🕓 Cooldown decay for {ticker}: {proba_short:.2f}")

                    blended_proba = 0.6 * proba_short + 0.4 * proba_mid

                    if regime == "bear" and proba_short < 0.8:
                        print(f"⚠️ Bear regime: skipping {ticker} (conf: {proba_short:.2f})")
                        continue
                    elif regime == "sideways" and proba_short < 0.7:
                        print(f"⏸️ Sideways regime: skipping {ticker} (conf: {proba_short:.2f})")
                        continue

                    near_support, near_resistance = detect_support_resistance(df)
                    if prediction == 1 and near_resistance:
                        print(f"⛔ {ticker} near resistance. Avoiding buy.")
                        continue
                    elif prediction == 0 and near_support:
                        print(f"⛔ {ticker} near support. Avoiding sell.")
                        continue

                    price_change = latest_row["Close"] - latest_row["Open"]
                    if proba_short > 0.75 and price_change <= 0:
                        print(f"⚠️ {ticker} lacks intraday momentum. Skipping.")
                        continue

                    risks = get_risk_events(ticker)
                    if risks:
                        print(f"⚠️ {ticker} risky ({', '.join(risks)}). Skipping.")
                        continue

                    if blended_proba > 0.75:
                        prediction = 1
                    elif blended_proba < 0.4:
                        prediction = 0
                    else:
                        print(f"⏸️ Uncertain signal for {ticker}. Skipping.")
                        continue

                    sentiment = get_sentiment_score(ticker)
                    score = (
                        proba_short * 100 +
                        sentiment * 10 -
                        latest_row["atr"] * 5 +
                        (current_volume / recent_volume) * 2
                    )
                    if score < 20:
                        print(f"⛔ Trade score too low ({score:.2f}) for {ticker}. Skipping.")
                        continue

                    if sector in used_sectors:
                        print(f"⛔ Already traded sector: {sector}. Skipping {ticker}.")
                        continue

                    used_sectors.add(sector)

                    execute_trade(ticker, prediction, proba_short, proba_mid, latest_row, df)
                    trade_count += 1

                except Exception as e:
                    msg = f"🚨 Trade logic error for {ticker}: {e}"
                    print(msg)
                    send_discord_message(msg)

            # ⏳ Auto-liquidate 5 mins before market close (12:55 PM PT)
            if dt_time(12, 55) <= now < dt_time(12, 56):
                print("⚠️ Approaching market close. Liquidating positions.")
                liquidate_positions()
                time.sleep(60)  # Prevent repeat execution
                continue

            # 📊 Send end-of-day summary just before market close
            if dt_time(12, 56) <= now < dt_time(12, 58):
                print("📊 Sending end-of-day summary before market close.")
                send_end_of_day_summary()
                time.sleep(120)  # Prevent repeat
                continue

            # ✅ Exit cleanly after 1:00 PM PT
            if now >= dt_time(13, 0):
                print("✅ Trading session complete. Exiting.")
                break

        else:
            print("⏸️ Market is closed. Waiting...")
            time.sleep(60)
            continue  # 🟢 Skip to next iteration without saving or dumping

        # ✅ Only run if market was open
        try:
            save_trade_cache(cooldown)
            with open(Q_TABLE_FILE, "w") as f:
                json.dump(q_table, f)
        except Exception as e:
            print(f"⚠️ Failed to save cooldown cache or Q-table: {e}")

        # ✅ Move this line outside the try block
        time.sleep(300)

    except Exception as e:  # ✅ Must align with try above!
        print(f"🚨 Fatal error in main trading loop: {e}")
        send_discord_message(f"🚨 Fatal error in main trading loop: {e}")
        time.sleep(60)  # Backoff before retrying
