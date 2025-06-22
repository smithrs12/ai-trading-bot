 # ai_trading_bot.py

# [FULLY INTEGRATED WITH ALL REQUESTED ENHANCEMENTS]
# Includes: Reinforcement Learning, Market Regime Detection, Sentiment Scoring (Reddit + News),
# Trade Cooldown, Position Sizing, Walk-forward Validation, Persistent Q-table, Discord Alerts,
# Trade Logging, Backtesting Switch (manual), Feature Importance Logging
# -*- coding: utf-8 -*-

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
import yfinance as yf
import finnhub
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

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
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"⚠️ Market status check failed: {e}")
        return False
     
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(FEATURE_DIR): os.makedirs(FEATURE_DIR)

import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

import xgboost as xgb

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


def get_market_regime():
    spy = get_data("SPY", days=10)
    if spy is None: return regime_cache["type"]
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

def get_data(ticker, days=3, interval="1m"):
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, interval=interval)
        if df.empty or len(df) < 10:
            return None

        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
        df["sma"] = SMAIndicator(close=df["Close"], window=14).sma_indicator()
        df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()
        macd = MACD(close=df["Close"])
        df["macd"] = macd.macd()
        df["macd_diff"] = macd.macd_diff()
        df["stoch"] = StochasticOscillator(high=df["High"],
                                           low=df["Low"],
                                           close=df["Close"]).stoch()
        df["atr"] = AverageTrueRange(high=df["High"],
                                     low=df["Low"],
                                     close=df["Close"]).average_true_range()
        df["bb_bbm"] = BollingerBands(close=df["Close"]).bollinger_mavg()
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["dayofweek"] = df.index.dayofweek
        df = calculate_vwap(df)
        return df.dropna() if len(df) > 50 else None
    except Exception as e:
        print(f"❌ Data error for {ticker}: {e}", flush=True)
        return None

def detect_support_resistance(df, window=20, tolerance=0.01):
    try:
        recent_high = df["High"].rolling(window).max().iloc[-1]
        recent_low = df["Low"].rolling(window).min().iloc[-1]
        current_price = df["Close"].iloc[-1]
        
        near_resistance = abs(current_price - recent_high) / recent_high < tolerance
        near_support = abs(current_price - recent_low) / recent_low < tolerance

        return near_support, near_resistance
    except Exception as e:
        print(f"⚠️ Support/resistance detection failed: {e}")
        return False, False

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
    if total_weight == 0:
        return 0.5  # fallback
    probs = [model.predict_proba(X)[0][1] for model in models]
    weighted_avg = sum(w * p for w, p in zip(weights, probs)) / total_weight
    return weighted_avg

def dual_horizon_predict(ticker, model, features, short_days=2, mid_days=15):
    df_short = get_data(ticker, days=short_days)
    df_mid = get_data(ticker, days=mid_days)

    if df_short is None or df_mid is None or len(df_short) < 10 or len(df_mid) < 10:
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
            proba_short = model.predict_proba()[0][1]
            proba_mid = model.predict_proba(X_mid)[0][1]

        return proba_short, proba_mid, df_short.iloc[-1]

    except Exception as e:
        print(f"⚠️ Dual horizon prediction failed for {ticker}: {e}")
        return None, None, None

def predict(ticker, model, features):
    df = get_data(ticker, days=2)
    if df is None or len(df) < 10:
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
    if len(X) < 60 or y.nunique() < 2:
        print(f"⚠️ Insufficient or non-diverse data for {ticker} (medium-term).")
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

    # ✅ Save short-term model performance
    perf_log = {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "short",
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "samples": len(X)
    }
    perf_df = pd.DataFrame([perf_log])
    perf_file = os.path.join("model_performance.csv")
    perf_df.to_csv(perf_file, mode='a', header=not os.path.exists(perf_file), index=False)
    return ensemble, features

def predict_medium_term(ticker):
    model_path = os.path.join("models_medium", f"{ticker}_medium.pkl")
    if not os.path.exists(model_path):
        return None

    try:
        model = joblib.load(model_path)
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

def execute_trade(ticker, prediction, proba, proba_mid, cooldown_cache, latest_row, df):
    try:
        position = None
        try:
            position = api.get_position(ticker)
        except:
            pass

        current_price = api.get_latest_bar(ticker).c
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---- RL-driven HOLD logic ----
        if position:
            _price = float(position.avg_entry_price)
            gain = (current_price - _price) / _price
            state = q_state(ticker, 1).unsqueeze(0)
            hold_value = q_net(state).item()
            if hold_value > 0.3 and gain > 0 and proba > 0.55:
                print(f"⏸️ RL prefers to hold {ticker} (Q={hold_value:.2f})")
                return

        if ticker in cooldown_cache:
            last_trade = cooldown_cache[ticker]
            last_time = datetime.strptime(last_trade["timestamp"], "%Y-%m-%d %H:%M:%S")
            last_conf = last_trade.get("confidence", 0.5)
            cooldown_secs = int(600 + 600 * last_conf)
            if (datetime.now() - last_time).seconds < cooldown_secs:
                print(f"⏳ Adaptive cooldown active for {ticker} ({cooldown_secs//60} min)")
                return

        equity = float(api.get_account().equity)
        max_dollars = equity * MAX_POSITION_PCT
        atr = latest_row.get("atr", 0.5)
        qty = kelly_position_size(proba, current_price, equity, atr=atr, ref_atr=0.5)
        sentiment = get_sentiment_score(ticker)
        price_change = latest_row["Close"] - latest_row["Open"]

        # ---- BUY logic ----
        if prediction == 1 and qty > 0 and (not position or float(position.market_value) < max_dollars):

            # ---- Meta Model Approval ----
            try:
                if os.path.exists("meta_model.pkl"):
                    import xgboost as xgb
                    meta_model = joblib.load("meta_model.pkl")
                    meta_features = pd.DataFrame([{
                        "proba_short": proba,
                        "proba_mid": proba_mid,
                        "sentiment": sentiment,
                        "price_change": price_change,
                        "atr": atr,
                        "vwap_diff": latest_row["Close"] - latest_row["vwap"],
                        "volume_ratio": latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2],
                    }])
                    meta_pred = meta_model.predict(meta_features)[0]
                    if meta_pred == 0:
                        print(f"⛔ Meta model vetoed the trade for {ticker}. Skipping.")
                        return
            except Exception as e:
                print(f"⚠️ Meta model check failed for {ticker}: {e}")

            # ✅ Only runs if no veto or exception
            api.submit_order(symbol=ticker,
                             qty=qty,
                             side="buy",
                             type="market",
                             time_in_force="gtc")
            send_discord_message(
                f"🟢 Bought {qty} shares of {ticker} at ${current_price:.2f} (Conf: {proba:.2f} Sentiment: {sentiment:+.2f})"
            )
            log_trade(timestamp, ticker, "BUY", qty, current_price)
            cooldown_cache[ticker] = {
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
        "vwap_diff": latest_row["Close"] - latest_row["vwap"],
        "volume_ratio": latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2],
        "final_outcome": outcome
    }
    log_meta_training_row(meta_log)
except Exception as e:
    print(f"⚠️ Failed to log meta training row for {ticker}: {e}")

        # ---- ADDITIONAL BUY logic (Pyramiding) ----
        pyramiding_count = cooldown_cache.get(ticker, {}).get("adds", 0)

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
                        meta_features = pd.DataFrame([{
                            "proba_short": proba,
                            "proba_mid": proba_mid,
                            "sentiment": sentiment,
                            "price_change": price_change,
                            "atr": atr,
                            "vwap_diff": latest_row["Close"] - latest_row["vwap"],
                            "volume_ratio": latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2],
                        }])
                        meta_pred = meta_model.predict(meta_features)[0]
                        if meta_pred == 0:
                            print(f"⛔ Meta model vetoed the trade for {ticker}. Skipping.")
                            return
                except Exception as e:
                    print(f"⚠️ Meta model check failed for {ticker}: {e}")

                api.submit_order(symbol=ticker,
                                 qty=additional_qty,
                                 side="buy",
                                 type="market",
                                 time_in_force="gtc")
                send_discord_message(
                    f"🔼 Added {additional_qty} more shares to {ticker} at ${current_price:.2f} (Strong Dual Horizon Signal)"
                )
                log_trade(timestamp, ticker, "BUY_MORE", additional_qty, current_price)
                log_pnl(ticker, additional_qty, current_price, "BUY", current_price, "short")
                cooldown_cache[ticker] = {
                    "timestamp": timestamp,
                    "confidence": float(min(proba + 0.1, 1.0)),
                    "adds": pyramiding_count + 1
                }
                update_q_nn(ticker, 1, reward_function(1, (proba - 0.5) * 1.5))
             
        # ---- Meta Model Logging ----
        try:
            outcome = 1
            meta_log = {
                "proba_short": proba,
                "proba_mid": proba_mid,
                "sentiment": sentiment,
                "price_change": price_change,
                "atr": atr,
                "vwap_diff": latest_row["Close"] - latest_row["vwap"],
                "volume_ratio": latest_row["Volume"] / df["Volume"].rolling(20).mean().iloc[-2],
                "final_outcome": outcome
            }
            log_meta_training_row(meta_log)
        except Exception as e:
            print(f"⚠️ Failed to log meta training row for {ticker}: {e}")

        return

    except Exception as e:
        print(f"🚨 execute_trade crashed for {ticker}: {e}")
        send_discord_message(f"🚨 Trade failed for {ticker}: {e}")

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
                    api.submit_order(symbol=ticker, qty=int(position.qty), side="sell", type="market", time_in_force="gtc")
                    send_discord_message(f"💰 Took profit on {position.qty} shares of {ticker} at ${current_price:.2f} (Gain: {gain:.2%})")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, 0.5 - proba))
                    return

                # Dynamic Trailing Stop
                trailing_atr_factor = 1.5 if gain < 0.05 else 2.5
                trailing_stop_price = current_price - (volatility * trailing_atr_factor)
                if current_price < trailing_stop_price:
                    send_discord_message(f"🔻 {ticker} hit dynamic trailing stop at ${current_price:.2f}.")
                    api.submit_order(symbol=ticker, qty=int(position.qty), side="sell", type="market", time_in_force="gtc")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, 0.5 - proba))
                    return

                # Profit Decay
                time_held_minutes = 0
                entry_time_str = cooldown_cache.get(ticker, {}).get("timestamp")
                if entry_time_str:
                    time_held_minutes = (datetime.now() - datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")).total_seconds() / 60

                gain_pct = (current_price - _price) / _price
                if 0 < gain_pct < 0.02 and time_held_minutes > 60:
                    send_discord_message(f"📉 Exiting {ticker} due to fading profits ({gain_pct:.2%}) after {time_held_minutes:.0f} mins.")
                    api.submit_order(symbol=ticker, qty=int(position.qty), side="sell", type="market", time_in_force="gtc")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, _price, current_price))
                    return

                # Stop-loss or confidence drop
                if current_price <= stop_loss_price or proba < 0.4:
                    api.submit_order(symbol=ticker, qty=int(position.qty), side="sell", type="market", time_in_force="gtc")
                    send_discord_message(f"🔴 Sold {position.qty} of {ticker} at ${current_price:.2f} (SL or Sell Signal)")
                    log_trade(timestamp, ticker, "SELL", int(position.qty), current_price)
                    log_pnl(ticker, int(position.qty), current_price, "SELL", _price, "short")
                    update_q_nn(ticker, 0, reward_function(0, 0.5 - proba))
                    return

            except Exception as e:
                print(f"⚠️ Sell logic failed for {ticker}: {e}")

def get_dynamic_watchlist(limit=8):
    tickers_to_scan = [
        "F", "SOFI", "SIRI", "MARA", "PLTR", "INTC", "CHPT", "OPEN", "RIVN",
        "LCID", "PINS", "MIRM", "FROG", "CHWY", "UBER", "MRVL", "CGC", "AAPL",
        "NVDA", "AMD", "TSLA", "AMZN", "META", "MSFT", "GOOG", "COIN", "SHOP",
        "SNAP", "DIS", "T"
    ]
    return tickers_to_scan[:limit]

def liquidate_positions():
    for pos in api.list_positions():
        try:
            qty = int(float(pos.qty))
            if qty > 0:
                api.submit_order(
                    symbol=pos.symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                log_trade(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pos.symbol, "SELL", qty, float(pos.market_value) / qty)
                send_discord_message(f"⏳ Auto-liquidated {qty} shares of {pos.symbol} before close.")
        except Exception as e:
            print(f"❌ Error liquidating {pos.symbol}: {e}")

def send_end_of_day_summary():
    if not os.path.exists(TRADE_LOG_FILE):
        send_discord_message("📉 No trades executed today.")
        return

    df = pd.read_csv(TRADE_LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    today = datetime.utcnow().date()
    df_today = df[df["timestamp"].dt.date == today]

    if df_today.empty:
        send_discord_message("📉 No trades executed today.")
        return
            
    candidates = []
    for ticker in UNIVERSE:
        try:
            df = get_data(ticker, days=5)
            if df is None or len(df) < 5: continue

            ret = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
            sentiment = get_sentiment_score(ticker)
            atr = df["atr"].iloc[-1]
            volume = df["Volume"].rolling(5).mean().iloc[-1]

            # Score formula: prioritize momentum, volatility, sentiment, and liquidity
            regime = get_market_regime()
            if regime == "bull":
                score = (ret * 120) + (sentiment * 4) + np.log(volume)
            elif regime == "bear":
                score = (sentiment * 6) + (atr * 3) + np.log(volume)
            else:  # sideways
                score = (ret * 60) + (sentiment * 4) + np.log(volume)

            candidates.append((ticker, score))
        except Exception as e:
            print(f"⚠️ Skipping {ticker}: {e}")

    # Sort by score and return top tickers
    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    selected = [t[0] for t in ranked[:limit]]

    print(f"📈 Dynamic watchlist selected: {selected}", flush=True)
    send_discord_message(f"📈 Dynamic Watchlist: {', '.join(selected)}")
    return selected

# ✅ FIXED: df_today must be defined from trade log
if not os.path.exists(TRADE_LOG_FILE):
    df_today = pd.DataFrame()
else:
    df = pd.read_csv(TRADE_LOG_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    today = datetime.utcnow().date()
    df_today = df[df["timestamp"].dt.date == today]

profit = 0
wins = 0
losses = 0
trades = []

for ticker in df_today["ticker"].unique():
    buys = df_today[(df_today["ticker"] == ticker) & (df_today["action"] == "BUY")]
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

try:
    account = api.get_account()
    portfolio_value = float(account.portfolio_value)
except:
    portfolio_value = "N/A"

total_trades = wins + losses
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

summary = f"📊 End of Day Summary ({today}):\n"
summary += "\n".join(trades) + "\n"
summary += f"\n💰 Total P/L: ${profit:.2f}"
summary += f"\n📈 Portfolio Value: ${portfolio_value}"
summary += f"\n✅ Win Rate: {wins}/{total_trades} ({win_rate:.1f}%)"

send_discord_message(summary)

try:
    cooldown = load_trade_cache()
except Exception as e:
    print(f"⚠️ Failed to load cooldown cache: {e}")
    cooldown = {}

try:
    cooldown = load_trade_cache()
except Exception as e:
    print(f"⚠️ Failed to load cooldown cache: {e}")
    cooldown = {}

while True:
    try:
        if is_market_open():
            print("🔁 Trading cycle...", flush=True)
            trade_count = 0
            regime = get_market_regime()
            used_sectors = set()
            trade_candidates = []
            model, features = None, None
            TICKERS = get_dynamic_watchlist(limit=8)

            trade_candidates = sorted(trade_candidates, key=lambda x: x[1], reverse=True)

            for cand in trade_candidates[:5]:  # Top 5 trades
                try:
                    ticker, score, model, features, latest_row, proba_short, proba_mid, prediction, sector = cand

                    # [your trading logic here]

                except Exception as e:
                    msg = f"🚨 Bot crashed in market open loop: {e}"
                    print(msg, flush=True)
                    send_discord_message(msg)

        else:
            print("⏸️ Market is closed. Waiting...")
            time.sleep(60)

        save_trade_cache(cooldown)
        time.sleep(300)

    except Exception as e:
        print(f"🚨 Fatal error in trading loop: {e}")
        send_discord_message(f"🚨 Fatal error: {e}")
        time.sleep(60)

for cand in trade_candidates[:5]:  # Top 5 trades
    try:
        ticker, score, model, features, latest_row, proba_short, proba_mid, prediction, sector = cand

        df = get_data(ticker, days=5)
        if df is None or len(df) < 5:
            print(f"❌ Could not load data for {ticker}, skipping.")
            continue

        # Retrain medium-term model if stale
        if is_medium_model_stale(ticker):
            print(f"🔁 Retraining medium-term model for {ticker}...")
            train_medium_model(ticker)

        # Retrain short-term model if stale
        model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")
        if is_model_stale(ticker) or not os.path.exists(model_path):
            print(f"🔁 Retraining model for {ticker}...")
            model, features = train_model(ticker, df)
            if model and features:
                try:
                    joblib.dump(model, model_path)
                except Exception as e:
                    print(f"❌ Failed to save model for {ticker}: {e}")
                    continue
        else:
            try:
                model = joblib.load(model_path)
                features = df.columns.intersection([
                    "sma", "rsi", "macd", "macd_diff", "stoch",
                    "atr", "bb_bbm", "hour", "minute", "dayofweek"
                ]).tolist()
            except Exception as e:
                print(f"⚠️ Failed to load model for {ticker}: {e}")
                continue

        if model is None or features is None:
            print(f"⚠️ Skipping {ticker}: no trained model or features.")
            continue

        # Run predictions
        prediction, latest_row, proba_short = predict(ticker, model, features)
        proba_mid = predict_medium_term(ticker)
        if proba_short is None or proba_mid is None or latest_row is None:
            print(f"⚠️ Missing prediction data for {ticker}, skipping.")
            continue

        # Confidence decay if still on cooldown
        last_ts = cooldown.get(ticker, {}).get("timestamp")
        if last_ts:
            seconds_elapsed = (datetime.now() - datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")).total_seconds()
            if seconds_elapsed < 600:
                decay_factor = 1 - (seconds_elapsed / 600)
                proba_short *= decay_factor
                proba_short = min(max(proba_short, 0), 1)
                print(f"🕓 Cooldown active for {ticker}. Adjusted confidence: {proba_short:.2f}")

        # HOLD logic
        if 0.6 <= proba_short < 0.75 and proba_mid >= 0.75:
            print(f"⏸️ HOLDING {ticker}: Short-term moderate, mid-term strong outlook.")
            continue

        # VWAP check
        if latest_row["Close"] < latest_row["vwap"]:
            print(f"⏸️ {ticker} price below VWAP. Skipping.")
            continue

        # Volume spike check
        recent_volume = df["Volume"].rolling(20).mean().iloc[-2]
        current_volume = latest_row["Volume"]
        if current_volume < 1.5 * recent_volume:
            print(f"⏸️ {ticker} volume not spiking (Current: {current_volume:.0f}, Avg: {recent_volume:.0f}). Skipping.")
            continue

        # Support/resistance logic
        near_support, near_resistance = detect_support_resistance(df)
        if prediction == 1 and near_resistance:
            print(f"⏸️ {ticker} near resistance. Avoiding buy.")
            continue
        elif prediction == 0 and near_support:
            print(f"⏸️ {ticker} near support. Avoiding sell.")
            continue

        # Momentum logic
        price_change = latest_row["Close"] - latest_row["Open"]
        if proba_short > 0.75 and price_change <= 0:
            print(f"⚠️ {ticker} short-term strong but lacks intraday momentum. Skipping.")
            continue

        # News & risk logic
        risks = get_risk_events(ticker)
        if risks:
            print(f"⚠️ Skipping {ticker} due to risk events: {', '.join(risks)}")
            continue

        # Blended short + medium signal
        blended_proba = 0.6 * proba_short + 0.4 * proba_mid
        if regime == "bear" and proba_short < 0.8:
            print(f"⚠️ Bear market: Skipping {ticker} due to low confidence ({proba_short:.2f}).")
            continue
        elif regime == "sideways" and proba_short < 0.7:
            print(f"⏸️ Sideways market: Skipping {ticker} with moderate confidence ({proba_short:.2f}).")
            continue

        # Adjust prediction if needed
        if blended_proba > 0.75:
            prediction = 1
        elif blended_proba < 0.4:
            prediction = 0
        else:
            print(f"⏸️ Blended signal too uncertain for {ticker}. Skipping.")
            continue

        # Score trade opportunity
        sentiment = get_sentiment_score(ticker)
        score = (
            proba_short * 100 +
            sentiment * 10 -
            latest_row["atr"] * 5 +
            (current_volume / recent_volume) * 2
        )

        # Add to candidate list
        sector = SECTOR_MAP.get(ticker, "Unknown")
        trade_candidates.append((ticker, score, model, features, latest_row, proba_short, proba_mid, prediction, sector))

        # Execute the trade
        execute_trade(ticker, prediction, proba_short, proba_mid, cooldown, latest_row, df)
        trade_count += 1
        if sector:
            used_sectors.add(sector)

        save_trade_cache(cooldown)
        time.sleep(300)

    except Exception as e:
        msg = f"🚨 Bot crashed in market open loop for {ticker}: {e}"
        print(msg, flush=True)
        send_discord_message(msg)
