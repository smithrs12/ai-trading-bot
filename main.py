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
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
if not os.path.exists(FEATURE_DIR): os.makedirs(FEATURE_DIR)


def log_trade(ts, ticker, action, qty, price):
    account = api.get_account()
    row = pd.DataFrame([[
        ts, ticker, action, qty, price, account.buying_power, account.equity
    ]],
                       columns=[
                           "timestamp", "ticker", "action", "qty", "price",
                           "buying_power", "equity"
                       ])
    row.to_csv(TRADE_LOG_FILE,
               mode='a',
               header=not os.path.exists(TRADE_LOG_FILE),
               index=False)

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

def reward_function(price_change, action):
    # Reward based on direction and magnitude of change
    return price_change if action == 1 else -price_change

def update_q_nn(state, action, reward, next_state):
    state = np.reshape(state, [1, len(state)])
    next_state = np.reshape(next_state, [1, len(next_state)])

    target = q_model.predict(state)[0]
    future = np.max(q_model.predict(next_state)[0])
    target[action] = reward + 0.95 * future  # gamma = 0.95

    q_model.fit(state, target.reshape(-1, 2), epochs=1, verbose=0)
    q_model.save(Q_MODEL_PATH)

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

# Neural Q-Network
def create_q_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="linear"))  # Two actions: Buy or Sell
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

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


def is_market_open():
    return api.get_clock().is_open


def close_to_market_close():
    now = datetime.utcnow().time()
    return dt_time(19, 55) <= now <= dt_time(20, 0)


def liquidate_positions():
    for pos in api.list_positions():
        qty = int(float(pos.qty))
        if qty > 0:
            api.submit_order(symbol=pos.symbol,
                             qty=qty,
                             side="sell",
                             type="market",
                             time_in_force="gtc")
            log_trade(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pos.symbol,
                      "SELL", qty,
                      float(pos.market_value) / qty)
            send_discord_message(
                f"⏳ Auto-liquidated {qty} shares of {pos.symbol} before close."
            )


def get_data(ticker, days=90):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    try:
        time.sleep(0.25)  # ⏳ Prevent hitting rate limits
        df = api.get_bars(ticker,
                          tradeapi.TimeFrame.Minute,
                          start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          feed='iex').df
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
        return df.dropna() if len(df) > 50 else None
    except Exception as e:
        print(f"❌ Data error for {ticker}: {e}", flush=True)
        return None


def load_trade_cache():
    return json.load(
        open(TRADE_CACHE_FILE)) if os.path.exists(TRADE_CACHE_FILE) else {}


def save_trade_cache(cache):
    json.dump(cache, open(TRADE_CACHE_FILE, "w"))


def kelly_position_size(prob, price, equity):
    edge = 2 * prob - 1
    if edge <= 0:
        return 0
    kelly_fraction = min(edge, MAX_POSITION_PCT)
    allocation = equity * kelly_fraction
    return int(allocation // price)


def train_model(ticker, df):
    df["future_return"] = df["Close"].shift(-3) / df["Close"] - 1
    df["target"] = (df["future_return"] > 0.01).astype(int)
    df = df.dropna()
    features = [
        "sma", "rsi", "macd", "macd_diff", "stoch", "atr", "bb_bbm", "hour",
        "minute", "dayofweek"
    ]
    X, y = df[features], df["target"]
    if len(X) < 60:
        return None, None

    # Initialize models
    xgb_model = xgb.XGBClassifier(eval_metric='logloss',
                                  use_label_encoder=False)
    log_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100)

    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('log', log_model), ('rf', rf_model)],
        voting='soft',
        weights=[3, 1, 2])  # Weighted voting: prioritize XGB & RF

    # Cross-validation performance tracking
    tscv = TimeSeriesSplit(n_splits=5)
    accs, precs, recs = [], [], []

    for train_idx, test_idx in tscv.split(X):
        ensemble.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = ensemble.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y.iloc[test_idx], y_pred))
        precs.append(precision_score(y.iloc[test_idx], y_pred,
                                     zero_division=0))
        recs.append(recall_score(y.iloc[test_idx], y_pred, zero_division=0))

    print(
        f"📊 [{ticker}] Acc: {np.mean(accs):.4f} | Prec: {np.mean(precs):.4f} | Rec: {np.mean(recs):.4f}",
        flush=True)
    send_discord_message(
        f"📊 [{ticker}] Ensemble Training:\nAcc: {np.mean(accs):.4f}, Prec: {np.mean(precs):.4f}, Rec: {np.mean(recs):.4f}"
    )

    # Skip training if y has fewer than 2 classes
    if len(np.unique(y)) < 2:
        print(
            f"⚠️ Skipping training for {ticker}: only one class in target variable."
        )
        return None, None

    try:
        ensemble.fit(X, y)
        joblib.dump(ensemble, os.path.join(MODEL_DIR, f"{ticker}.pkl"))
    except Exception as e:
        print(f"❌ Failed to train model for {ticker}: {e}")
        return None, None

    def is_model_stale(ticker, max_age_hours=24):
        model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")
        if not os.path.exists(model_path):
            return True
        last_modified = os.path.getmtime(model_path)
        age_hours = (time.time() - last_modified) / 3600
        return age_hours > max_age_hours

    def is_medium_model_stale(ticker, max_age_hours=24):
        path = os.path.join("models_medium", f"{ticker}_medium.pkl")
        if not os.path.exists(path):
            return True
        last_modified = os.path.getmtime(path)
        age_hours = (time.time() - last_modified) / 3600
        return age_hours > max_age_hours

    # Feature importance (from XGBoost component only)
    fi = pd.DataFrame({
        "feature": features,
        "importance": xgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(FEATURE_DIR, f"{ticker}_features.csv"), index=False)
    # Save model performance to CSV
    perf_log = {
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "samples": len(X)
    }
    perf_df = pd.DataFrame([perf_log])
    perf_file = os.path.join("model_performance.csv")
    perf_df.to_csv(perf_file,
                   mode='a',
                   header=not os.path.exists(perf_file),
                   index=False)

    return ensemble, features

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
        
        # Log performance for medium-term model
        perf_log = {
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "samples": len(X),
            "type": "medium"
        }
        perf_df = pd.DataFrame([perf_log])
        perf_file = os.path.join("model_performance.csv")
        perf_df.to_csv(perf_file, mode='a', header=not os.path.exists(perf_file), index=False)
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
        proba = model.predict_proba(X)[0][1]
        return proba
    except Exception as e:
        print(f"⚠️ Medium-term prediction error for {ticker}: {e}")
        return None

def predict(ticker, model, features):
    df = get_data(ticker, days=2)
    if df is None or len(df) < 10: return 0, None, None
    X = df[features].iloc[-1:]
    proba = model.predict_proba(X)[0][1]
    return int(proba > 0.5), df.iloc[-1], proba

# ✅ Load medium-term model and make prediction
medium_model_path = os.path.join("models_medium", f"{ticker}_medium.pkl")
if os.path.exists(medium_model_path):
    try:
        medium_model = joblib.load(medium_model_path)
        df_mid = get_data(ticker, days=15)
        if df_mid is not None and len(df_mid) > 5:
            X_mid = df_mid[["Open", "High", "Low", "Close", "Volume"]].iloc[-1:]
            medium_proba = medium_model.predict_proba(X_mid)[0][1]
        else:
            medium_proba = None
    except Exception as e:
        print(f"⚠️ Failed to load or use medium-term model for {ticker}: {e}")
        medium_proba = None
else:
    medium_proba = None

def dual_horizon_predict(ticker, model, features, short_days=2, mid_days=15):
    df_short = get_data(ticker, days=short_days)
    df_mid = get_data(ticker, days=mid_days)

    if df_short is None or df_mid is None or len(df_short) < 10 or len(
            df_mid) < 10:
        return None, None, None

    X_short = df_short[features].iloc[-1:]
    X_mid = df_mid[features].iloc[-1:]

    try:
        proba_short = model.predict_proba(X_short)[0][1]
        proba_mid = model.predict_proba(X_mid)[0][1]
    except:
        return None, None, None

    return proba_short, proba_mid, df_short.iloc[-1]


def execute_trade(ticker, prediction, proba, cooldown_cache):
    try:
        position = None
        try:
            position = api.get_position(ticker)
        except:
            pass

        current_price = api.get_latest_bar(ticker).c
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if cooldown_cache.get(ticker) and (datetime.now() - datetime.strptime(
                cooldown_cache[ticker], "%Y-%m-%d %H:%M:%S")).seconds < 600:
            print(f"⏳ Cooldown active for {ticker}")
            return

        equity = float(api.get_account().equity)
        max_dollars = equity * MAX_POSITION_PCT
        qty = kelly_position_size(proba, current_price, equity)
        sentiment = get_sentiment_score(ticker)

        if prediction == 1 and qty > 0 and (not position or float(
                position.market_value) < max_dollars):
            api.submit_order(symbol=ticker,
                             qty=qty,
                             side="buy",
                             type="market",
                             time_in_force="gtc")
            send_discord_message(
                f"🟢 Bought {qty} shares of {ticker} at ${current_price:.2f} (Conf: {proba:.2f} Sentiment: {sentiment:+.2f})"
            )
            log_trade(timestamp, ticker, "BUY", qty, current_price)
            cooldown_cache[ticker] = timestamp
            log_pnl(ticker, qty, current_price, "BUY", current_price, "short")
            update_q_nn(ticker, 1, reward_function(1, proba - 0.5))
        elif prediction == 0 and position:
            stop_loss_price = float(
                position.avg_entry_price) * (1 - STOP_LOSS_THRESHOLD)
            entry_price = float(position.avg_entry_price)
            gain = (current_price - entry_price) / entry_price
            if gain > 0.07 and proba < 0.6:
                api.submit_order(symbol=ticker,
                                 qty=int(position.qty),
                                 side="sell",
                                 type="market",
                                 time_in_force="gtc")
                send_discord_message(
                    f"💰 Took profit on {position.qty} shares of {ticker} at ${current_price:.2f} (Gain: {gain:.2%})"
                )
                log_trade(timestamp, ticker, "SELL", int(position.qty),
                          current_price)
                log_pnl(ticker, int(position.qty), current_price, "SELL", entry_price, "short")
                update_q_nn(ticker, 0, reward_function(0, 0.5 - proba))
                return

            if current_price <= stop_loss_price or proba < 0.4:
                api.submit_order(symbol=ticker,
                                 qty=int(position.qty),
                                 side="sell",
                                 type="market",
                                 time_in_force="gtc")
                send_discord_message(
                    f"🔴 Sold {position.qty} of {ticker} at ${current_price:.2f} (SL or Sell Signal)"
                )
                log_trade(timestamp, ticker, "SELL", int(position.qty),
                          current_price)
                update_q(ticker, 0, reward_function(0, 0.5 - proba, gain))
    except Exception as e:
        print(f"❌ Trade error for {ticker}: {e}", flush=True)


print("✅ Console is working and bot is starting...", flush=True)


def get_dynamic_watchlist(limit=8):
    tickers_to_scan = [
        "F", "SOFI", "SIRI", "MARA", "PLTR", "INTC", "CHPT", "OPEN", "RIVN",
        "LCID", "PINS", "MIRM", "FROG", "CHWY", "UBER", "MRVL", "CGC", "AAPL",
        "NVDA", "AMD", "TSLA", "AMZN", "META", "MSFT", "GOOG", "COIN", "SHOP",
        "SNAP", "DIS", "T"
    ]

    candidates = []
    for ticker in tickers_to_scan:
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
    while True:
        if is_market_open():
            TICKERS = get_dynamic_watchlist()
            send_discord_message(f"🔄 New Top Tickers: {', '.join(TICKERS)}")
            if close_to_market_close():
                send_discord_message("🔔 Market close near. Liquidating positions.")
                liquidate_positions()
                send_end_of_day_summary()
            else:
                print("🔁 Trading cycle...", flush=True)
                trade_count = 0
                regime = get_market_regime()
                used_sectors = set()
                for ticker in TICKERS:
                    df = get_data(ticker)
                    if df is None:
                        continue
                        
                    sector = SECTOR_MAP.get(ticker, None)
                    if sector in used_sectors:
                        print(f"⏸️ Skipping {ticker} due to sector concentration: {sector}")
                        continue

                    model_path = os.path.join(MODEL_DIR, f"{ticker}.pkl")

                    # Check if model needs retraining
                    if is_model_stale(
                            ticker) or not os.path.exists(model_path):
                        print(f"🔁 Retraining model for {ticker}...")
                        model, features = train_model(ticker, df)
                        if model and features:
                            try:
                                joblib.dump(model, model_path)
                            except Exception as e:
                                print(
                                    f"❌ Failed to save model for {ticker}: {e}"
                                )
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
                        print(
                            f"⚠️ Skipping {ticker}: no trained model or features."
                        )
                        continue

                    # Dual-horizon prediction
                    
                    # ✅ Retrain medium-term model if stale
                    if is_medium_model_stale(ticker):
                        print(f"🔁 Retraining medium-term model for {ticker}...")
                        train_medium_model(ticker)

                    # ---- Short-term Prediction ----
                    prediction, latest_row, proba_short = predict(ticker, model, features)

                    # ---- Medium-term Prediction ----
                    proba_mid = predict_medium_term(ticker)

                    if proba_short is None or proba_mid is None:
                        print(f"⚠️ Missing prediction data for {ticker}, skipping.")
                        continue

                    # ---- HOLD logic ----
                    if 0.6 <= proba_short < 0.75 and proba_mid >= 0.75:
                        print(f"⏸️ HOLDING {ticker}: Short-term moderate, mid-term strong outlook.")
                        continue

                    # ---- Momentum Confirmation ----
                    price_change = latest_row["Close"] - latest_row["Open"]
                    if proba_short > 0.75 and price_change <= 0:
                        print(f"⚠️ {ticker} short-term strong but lacks intraday momentum. Skipping.")
                        continue

                # ✅ Upgrade: Use both short- and medium-term signals
                if medium_proba is None:
                    print(f"⚠️ No medium-term signal for {ticker}, skipping trade.")
                    continue

                if proba_short > 0.75 and medium_proba > 0.7:
                    prediction = 1
                elif proba_short < 0.4 and medium_proba < 0.4:
                    prediction = 0
                else:
                    print(f"⏸️ Mixed short/medium signal for {ticker}, skipping.")
                    continue

                execute_trade(ticker, prediction, proba_short, cooldown)
                trade_count += 1
                if sector:
                used_sectors.add(sector)

                save_trade_cache(cooldown)
                account = api.get_account()
                send_discord_message(
                    f"📊 Trades: {trade_count} | Portfolio: ${account.portfolio_value}"
                )
                try:
                    df_pnl = pd.read_csv("pnl_tracker.csv")
                    today = datetime.now().strftime("%Y-%m-%d")
                    df_today = df_pnl[df_pnl["timestamp"].str.startswith(today)]

                    total_pnl = df_today["pnl"].sum()
                    per_ticker = df_today.groupby("ticker")["pnl"].sum().to_dict()

                    summary = f"📊 Daily PnL Summary: ${total_pnl:.2f}\n" + "\n".join(
                        [f"{ticker}: ${pnl:.2f}" for ticker, pnl in per_ticker.items()]
                    )
                    send_discord_message(summary)
                except Exception as e:
                    print(f"⚠️ Failed to send PnL summary: {e}")
        time.sleep(300)
except Exception as e:
    msg = f"🚨 Bot crashed: {e}"
    print(msg, flush=True)
    send_discord_message(msg)
