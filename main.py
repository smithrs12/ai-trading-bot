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
from ta.trend import MACD, EMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from datetime import datetime, timedelta
from pytz import timezone
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from alpaca_trade_api.rest import REST, TimeFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
import asyncio
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# === ENHANCED CONFIGURATION ===
DEBUG = True
load_dotenv()
pacific = timezone('US/Pacific')

# Enhanced model directories
os.makedirs("models/short", exist_ok=True)
os.makedirs("models/medium", exist_ok=True)
os.makedirs("models/meta", exist_ok=True)
os.makedirs("models/q_learning", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("performance", exist_ok=True)

# === STRATEGIC THRESHOLDS ===
THRESHOLDS = {
    'SHORT_BUY_THRESHOLD': 0.53,
    'SHORT_SELL_AVOID_THRESHOLD': 0.45,
    'PRICE_MOMENTUM_MIN': 0.005,  # 0.5%
    'VOLUME_SPIKE_MIN': 1.2,
    'SENTIMENT_HOLD_OVERRIDE': -0.5,
    'MAX_PER_SECTOR_WATCHLIST': 12,
    'MAX_PER_SECTOR_PORTFOLIO': 0.3,  # 30% max per sector
    'WATCHLIST_LIMIT': 20,
    'ATR_STOP_MULTIPLIER': 1.2,
    'ATR_PROFIT_MULTIPLIER': 2.5,
    'MAX_PORTFOLIO_RISK': 0.02,  # 2% max risk per trade
    'MAX_DAILY_DRAWDOWN': 0.05,  # 5% max daily drawdown
    'VOLATILITY_GATE_THRESHOLD': 0.05  # 5% volatility threshold
}

def log(msg):
    if DEBUG:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
        # Also log to file
        with open("logs/trading_bot.log", "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

# === API SETUP ===
ALPACA_API_KEY = os.getenv("AlPACA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    log("❌ Missing Alpaca API credentials")
    exit(1)

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=ALPACA_BASE_URL)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None
analyzer = SentimentIntensityAnalyzer()

# Google Sheets Setup
GSPREAD_JSON_PATH = os.getenv("GSPREAD_JSON_PATH")
GSHEET_ID = os.getenv("GSHEET_ID")
gc = None
if GSPREAD_JSON_PATH and GSHEET_ID and os.path.exists(GSPREAD_JSON_PATH):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name(GSPREAD_JSON_PATH, scope)
        gc = gspread.authorize(credentials)
        log("✅ Google Sheets connected")
    except Exception as e:
        log(f"⚠️ Google Sheets setup failed: {e}")

# === ENHANCED UNIVERSE WITH SECTOR MAPPING ===
SECTOR_UNIVERSE = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "AMD", "NFLX", "CRM", "ORCL", "IBM", "INTC", "QCOM", "AVGO", "TXN", "MU", "ADBE", "SNOW", "SHOP"],
    "Finance": ["BRK.B", "V", "MA", "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "TFC", "COF"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "PSX", "EOG", "MPC", "VLO", "OXY", "HAL"],
    "Healthcare": ["PFE", "JNJ", "LLY", "MRK", "ABT", "BMY", "CVS", "UNH", "ABBV", "TMO", "DHR", "MDT"],
    "Consumer": ["HD", "LOW", "COST", "TGT", "WMT", "PG", "PEP", "KO", "PM", "NKE", "SBUX", "MCD", "CMG", "DIS"],
    "Industrial": ["UNP", "CSX", "UPS", "FDX", "CAT", "DE", "GE", "HON", "BA", "LMT", "RTX", "NOC"],
    "REIT": ["PLD", "O", "SPG", "AMT", "CCI", "EQIX", "PSA", "EXR"],
    "Communication": ["T", "VZ", "CMCSA", "CHTR", "TMUS", "DISH"],
    "Materials": ["LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "AA"],
    "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "XEL", "PEG", "SRE"]
}

FALLBACK_UNIVERSE = [ticker for sector_tickers in SECTOR_UNIVERSE.values() for ticker in sector_tickers]

# === ENHANCED GLOBAL STATE ===
class TradingState:
    def __init__(self):
        self.sector_allocations = {}
        self.cooldown_timers = {}
        self.position_history = {}
        self.accuracy_tracker = {}
        self.pnl_tracker = {}
        self.trade_outcomes = []
        self.sentiment_cache = {}
        self.support_resistance_cache = {}
        self.volume_profile_cache = {}
        self.daily_drawdown = 0.0
        self.starting_equity = 0.0
        self.regime_state = "neutral"  # bull, bear, neutral
        
    def reset_daily(self):
        self.sector_allocations = {}
        self.cooldown_timers = {}
        self.sentiment_cache = {}
        self.support_resistance_cache = {}
        self.volume_profile_cache = {}
        self.daily_drawdown = 0.0
        self.starting_equity = 0.0

trading_state = TradingState()

# === ENHANCED Q-LEARNING NETWORK ===
class EnhancedQNetwork(torch.nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256):
        super(EnhancedQNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 3)  # Buy, Hold, Sell
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

q_net = EnhancedQNetwork()
q_optimizer = torch.optim.Adam(q_net.parameters(), lr=0.001)
q_criterion = torch.nn.MSELoss()

if os.path.exists("models/q_learning/enhanced_q_net.pth"):
    try:
        checkpoint = torch.load("models/q_learning/enhanced_q_net.pth", map_location='cpu')
        q_net.load_state_dict(checkpoint['model_state_dict'])
        q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log("🧠 Enhanced Q-network loaded")
    except:
        log("⚠️ Failed to load Q-network, starting fresh")
else:
    log("⚠️ No existing Enhanced Q-network found. Starting fresh.")
q_net.eval()

# === UTILITY FUNCTIONS ===
def send_discord_alert(message, urgent=False):
    """Send Discord alert with optional urgency"""
    try:
        if not DISCORD_WEBHOOK_URL:
            return
            
        if urgent:
            message = f"🚨 **URGENT** 🚨\n{message}"
        
        payload = {"content": message}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            log(f"📬 Discord alert sent: {message[:50]}...")
        else:
            log(f"⚠️ Discord alert failed: {response.status_code}")
            
    except Exception as e:
        log(f"❌ Discord alert error: {e}")

def log_to_google_sheets(data, sheet_name="TradingLog"):
    """Log data to Google Sheets"""
    try:
        if not gc or not GSHEET_ID:
            return
            
        sheet = gc.open_by_key(GSHEET_ID).worksheet(sheet_name)
        
        # Convert data to list format
        if isinstance(data, dict):
            row_data = [
                datetime.now().isoformat(),
                *data.values()
            ]
        elif isinstance(data, list):
            row_data = [datetime.now().isoformat()] + data
        else:
            row_data = [datetime.now().isoformat(), str(data)]
        
        sheet.append_row(row_data)
        log(f"📊 Logged to Google Sheets: {sheet_name}")
        
    except Exception as e:
        log(f"❌ Google Sheets logging failed: {e}")

def get_enhanced_data(ticker, limit=100, timeframe=TimeFrame.Minute, days_back=None):
    """Get enhanced market data with technical indicators"""
    try:
        # Determine timeframe and limit based on model type
        if days_back == 2:  # Short-term model
            timeframe = TimeFrame.Minute
            limit = 2 * 24 * 60 // 5  # 2 days of 5-minute candles
        elif days_back == 15:  # Medium-term model
            timeframe = TimeFrame.Day
            limit = 15  # 15 days of daily bars
        
        # Fetch raw data
        bars = api.get_bars(
            ticker,
            timeframe,
            limit=limit,
            feed='iex'
        ).df
        
        if bars.empty:
            return None
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in bars.columns for col in required_cols):
            log(f"⚠️ Missing required columns for {ticker}")
            return None
        
        # Filter out zero volume bars
        bars = bars[bars['volume'] > 0].copy()
        
        if len(bars) < 20:
            return None
        
        # Add technical indicators
        bars = add_technical_indicators(bars)
        
        return bars
        
    except Exception as e:
        log(f"❌ Data fetch failed for {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """Add comprehensive technical indicators to dataframe"""
    try:
        if df is None or df.empty:
            return df
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else df['close'].rolling(len(df)//2).mean()
        
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # RSI (14-period)
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD (diff)
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # On-Balance Volume (OBV)
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # ATR (Average True Range) - REQUIRED
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # Bollinger Bands
        if len(df) >= 20:
            bb = BollingerBands(close=df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # VWAP calculation - REQUIRED
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Price momentum - REQUIRED
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        df['price_momentum'] = df['momentum_5']  # Primary momentum indicator
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['support_resistance_ratio'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df
        
    except Exception as e:
        log(f"❌ Technical indicator calculation failed: {e}")
        return df

def is_market_open():
    """Check if market is currently open"""
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        log(f"❌ Market status check failed: {e}")
        return False

def is_near_market_close(minutes_before=30):
    """Check if we're near market close"""
    try:
        now = datetime.now(pytz.timezone("US/Eastern"))
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_close = (market_close - now).total_seconds() / 60
        return 0 <= time_to_close <= minutes_before
    except Exception as e:
        log(f"❌ Market close check failed: {e}")
        return False

def get_current_positions():
    """Get current positions with enhanced information"""
    try:
        positions = api.list_positions()
        position_dict = {}
        
        for pos in positions:
            position_dict[pos.symbol] = {
                'qty': int(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'side': pos.side
            }
        
        return position_dict
        
    except Exception as e:
        log(f"❌ Failed to get positions: {e}")
        return {}

def calculate_dynamic_kelly_position_size(ticker, win_rate=0.6, avg_win=0.05, avg_loss=0.03, confidence=0.5):
    """Calculate position size using Dynamic Kelly Criterion"""
    try:
        # Get available capital
        account = api.get_account()
        available_cash = float(account.cash)
        portfolio_value = float(account.equity)
        
        # Adjust Kelly based on model confidence
        confidence_multiplier = max(0.5, min(1.5, confidence * 2))
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds received (avg_win/avg_loss), p = win probability, q = loss probability
        b = avg_win / avg_loss
        p = win_rate * confidence_multiplier  # Adjust win rate by confidence
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety constraints and confidence adjustment
        kelly_fraction = max(0.005, min(kelly_fraction * confidence_multiplier, THRESHOLDS['MAX_PORTFOLIO_RISK']))
        
        # Get current price
        bar = api.get_latest_bar(ticker)
        current_price = bar.c
        
        # Calculate position size
        dollar_amount = portfolio_value * kelly_fraction
        shares = max(1, int(dollar_amount / current_price))
        
        log(f"💰 Dynamic Kelly sizing for {ticker}: {kelly_fraction:.1%} (conf: {confidence:.2f}) = {shares} shares (${dollar_amount:.2f})")
        
        return shares
        
    except Exception as e:
        log(f"❌ Dynamic Kelly sizing failed for {ticker}: {e}")
        return 1  # Default to 1 share

def check_drawdown_limit():
    """Check if daily drawdown limit is exceeded"""
    try:
        account = api.get_account()
        current_equity = float(account.equity)
        
        if trading_state.starting_equity == 0:
            trading_state.starting_equity = current_equity
            return True
        
        drawdown = (trading_state.starting_equity - current_equity) / trading_state.starting_equity
        trading_state.daily_drawdown = drawdown
        
        if drawdown > THRESHOLDS['MAX_DAILY_DRAWDOWN']:
            log(f"🛑 Daily drawdown limit exceeded: {drawdown:.2%}")
            send_discord_alert(f"🛑 Daily drawdown limit exceeded: {drawdown:.2%}", urgent=True)
            return False
        
        return True
        
    except Exception as e:
        log(f"❌ Drawdown check failed: {e}")
        return True

def detect_market_regime():
    """Detect current market regime (bull/bear/neutral)"""
    try:
        # Get SPY data for regime detection
        spy_data = get_enhanced_data("SPY", limit=50)
        if spy_data is None or spy_data.empty:
            return "neutral"
        
        # Calculate multiple timeframe returns
        returns_5d = (spy_data['close'].iloc[-1] - spy_data['close'].iloc[-5]) / spy_data['close'].iloc[-5]
        returns_20d = (spy_data['close'].iloc[-1] - spy_data['close'].iloc[-20]) / spy_data['close'].iloc[-20]
        
        # Calculate volatility
        volatility = spy_data['returns'].rolling(20).std().iloc[-1]
        
        # Regime classification
        if returns_5d > 0.02 and returns_20d > 0.05 and volatility < 0.02:
            regime = "bull"
        elif returns_5d < -0.02 and returns_20d < -0.05:
            regime = "bear"
        else:
            regime = "neutral"
        
        trading_state.regime_state = regime
        log(f"📊 Market regime detected: {regime} (5d: {returns_5d:.2%}, 20d: {returns_20d:.2%}, vol: {volatility:.2%})")
        
        return regime
        
    except Exception as e:
        log(f"❌ Regime detection failed: {e}")
        return "neutral"

# === DUAL-HORIZON PREDICTION SYSTEM ===
class DualHorizonPredictor:
    def __init__(self):
        self.short_models = {}  # 2-day data, 5-minute candles, 5-bar lookahead
        self.medium_models = {}  # 15-day data, daily bars, 5-day lookahead
        self.ensemble_weights = {"short": 0.6, "medium": 0.4}
        
    def create_voting_ensemble(self):
        return VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    eval_metric="logloss",
                    verbosity=0
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=2
                )),
                ('lr', LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver='liblinear'
                ))
            ],
            voting='soft'
        )
    
    def prepare_features(self, df, horizon='short'):
        """Enhanced feature engineering for dual horizon"""
        if df is None or df.empty or len(df) < 30:
            return None, None
            
        df = df.copy()
        
        # Target variable based on horizon
        if horizon == 'short':
            # 5-bar lookahead for short-term
            df['target'] = (df['close'].shift(-5) > df['close'] * 1.002).astype(int)
        else:  # medium
            # 5-day lookahead for medium-term
            df['target'] = (df['close'].shift(-5) > df['close'] * 1.01).astype(int)
        
        # Select features - ALL REQUIRED FEATURES
        feature_cols = [
            'returns', 'log_returns', 'momentum_5', 'momentum_10', 'momentum_20',
            'price_momentum',  # REQUIRED
            'volume_ratio', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'obv',  # REQUIRED
            'bb_position', 'price_vs_vwap',  # VWAP REQUIRED
            'volatility', 'atr',  # ATR REQUIRED
            'support_resistance_ratio'
        ]
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]
        if len(available_features) < 12:  # Need at least 12 features
            return None, None
        
        df_clean = df[available_features + ['target']].dropna()
        
        if len(df_clean) < 20:
            return None, None
            
        X = df_clean[available_features]
        y = df_clean['target']
        
        return X, y
    
    def train_models(self, ticker, short_data, medium_data):
        """Train both short and medium horizon models"""
        try:
            models_trained = 0
            
            # Short horizon model (2-day data, 5-minute candles)
            if short_data is not None:
                X_short, y_short = self.prepare_features(short_data, 'short')
                if X_short is not None and len(X_short) >= 30:
                    short_model = self.create_voting_ensemble()
                    
                    # Use TimeSeriesSplit for training
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(short_model, X_short, y_short, cv=tscv, scoring='accuracy')
                    
                    short_model.fit(X_short, y_short)
                    self.short_models[ticker] = short_model
                    joblib.dump(short_model, f"models/short/{ticker}_ensemble.pkl")
                    
                    log(f"✅ Trained short horizon model for {ticker} (CV accuracy: {scores.mean():.3f})")
                    models_trained += 1
            
            # Medium horizon model (15-day data, daily bars)
            if medium_data is not None:
                X_medium, y_medium = self.prepare_features(medium_data, 'medium')
                if X_medium is not None and len(X_medium) >= 10:
                    medium_model = self.create_voting_ensemble()
                    
                    # Use TimeSeriesSplit for training
                    tscv = TimeSeriesSplit(n_splits=2)
                    scores = cross_val_score(medium_model, X_medium, y_medium, cv=tscv, scoring='accuracy')
                    
                    medium_model.fit(X_medium, y_medium)
                    self.medium_models[ticker] = medium_model
                    joblib.dump(medium_model, f"models/medium/{ticker}_ensemble.pkl")
                    
                    log(f"✅ Trained medium horizon model for {ticker} (CV accuracy: {scores.mean():.3f})")
                    models_trained += 1
            
            return models_trained > 0
            
        except Exception as e:
            log(f"❌ Model training failed for {ticker}: {e}")
            return False
    
    def predict(self, ticker, current_short_data, current_medium_data):
        """Make dual horizon predictions - BOTH MUST AGREE"""
        try:
            short_prob = None
            medium_prob = None
            
            # Load models if not in memory
            if ticker not in self.short_models:
                short_path = f"models/short/{ticker}_ensemble.pkl"
                if os.path.exists(short_path):
                    self.short_models[ticker] = joblib.load(short_path)
            
            if ticker not in self.medium_models:
                medium_path = f"models/medium/{ticker}_ensemble.pkl"
                if os.path.exists(medium_path):
                    self.medium_models[ticker] = joblib.load(medium_path)
            
            # Short horizon prediction
            if ticker in self.short_models and current_short_data is not None:
                try:
                    short_prob = self.short_models[ticker].predict_proba([current_short_data])[0][1]
                except Exception as e:
                    log(f"⚠️ Short prediction failed for {ticker}: {e}")
            
            # Medium horizon prediction
            if ticker in self.medium_models and current_medium_data is not None:
                try:
                    medium_prob = self.medium_models[ticker].predict_proba([current_medium_data])[0][1]
                except Exception as e:
                    log(f"⚠️ Medium prediction failed for {ticker}: {e}")
            
            # DUAL-HORIZON REQUIREMENT: Both models must agree
            if short_prob is not None and medium_prob is not None:
                # Both must be above buy threshold OR both below avoid threshold
                both_bullish = (short_prob > THRESHOLDS['SHORT_BUY_THRESHOLD'] and 
                               medium_prob > THRESHOLDS['SHORT_BUY_THRESHOLD'])
                both_bearish = (short_prob < THRESHOLDS['SHORT_SELL_AVOID_THRESHOLD'] and 
                               medium_prob < THRESHOLDS['SHORT_SELL_AVOID_THRESHOLD'])
                
                if both_bullish:
                    combined_prob = (
                        short_prob * self.ensemble_weights['short'] + 
                        medium_prob * self.ensemble_weights['medium']
                    )
                    return short_prob, medium_prob, combined_prob, True  # Agreement
                elif both_bearish:
                    return short_prob, medium_prob, 0.0, True  # Agreement to avoid
                else:
                    return short_prob, medium_prob, 0.5, False  # No agreement
            
            return None, None, None, False
                
        except Exception as e:
            log(f"❌ Prediction failed for {ticker}: {e}")
            return None, None, None, False

dual_predictor = DualHorizonPredictor()

# === ENHANCED VWAP + VOLUME SPIKE FILTERING ===
class VWAPVolumeFilter:
    def __init__(self):
        self.volume_threshold_multiplier = THRESHOLDS['VOLUME_SPIKE_MIN']  # 1.2x
        self.vwap_deviation_threshold = 0.002
        
    def check_momentum_filter(self, df):
        """Check price momentum > 0.005 (0.5%)"""
        if df is None or df.empty or 'price_momentum' not in df.columns:
            return False
        
        current_momentum = df['price_momentum'].iloc[-1]
        return current_momentum > THRESHOLDS['PRICE_MOMENTUM_MIN']
    
    def check_volume_spike_filter(self, df):
        """Check current_volume > 1.2 * recent_volume"""
        if df is None or df.empty or len(df) < 20:
            return False
            
        current_volume = df['volume'].iloc[-1]
        recent_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        return current_volume > (recent_volume * self.volume_threshold_multiplier)
    
    def check_vwap_confirmation(self, df):
        """Check close > VWAP"""
        if df is None or df.empty or 'vwap' not in df.columns:
            return False
            
        current_price = df['close'].iloc[-1]
        current_vwap = df['vwap'].iloc[-1]
        
        return current_price > current_vwap
    
    def check_volatility_gate(self, df, sentiment_score):
        """Don't buy during volatility spike if sentiment is negative"""
        try:
            if df is None or df.empty or 'volatility' not in df.columns:
                return True
            
            current_volatility = df['volatility'].iloc[-1]
            
            # If high volatility AND negative sentiment, gate the trade
            if (current_volatility > THRESHOLDS['VOLATILITY_GATE_THRESHOLD'] and 
                sentiment_score < THRESHOLDS['SENTIMENT_HOLD_OVERRIDE']):
                log(f"🚫 Volatility gate triggered: vol={current_volatility:.3f}, sentiment={sentiment_score:.3f}")
                return False
            
            return True
            
        except Exception as e:
            log(f"❌ Volatility gate error: {e}")
            return True
    
    def passes_all_filters(self, ticker, sentiment_score=0):
        """Main filter function - ALL FILTERS MUST PASS"""
        try:
            df = get_enhanced_data(ticker, limit=50)
            if df is None:
                return False, "No data"
            
            # Check momentum filter
            if not self.check_momentum_filter(df):
                return False, "Momentum filter failed"
            
            # Check volume spike filter
            if not self.check_volume_spike_filter(df):
                return False, "Volume spike filter failed"
            
            # Check VWAP confirmation
            if not self.check_vwap_confirmation(df):
                return False, "VWAP confirmation failed"
            
            # Check volatility gate
            if not self.check_volatility_gate(df, sentiment_score):
                return False, "Volatility gate failed"
            
            return True, "All filters passed"
            
        except Exception as e:
            log(f"❌ Filter check failed for {ticker}: {e}")
            return False, f"Error: {e}"

vwap_filter = VWAPVolumeFilter()

# === SUPPORT/RESISTANCE LEVEL CONFIRMATION ===
class SupportResistanceAnalyzer:
    def __init__(self):
        self.lookback_period = 50
        self.touch_threshold = 0.01  # 1% threshold for level touches
        self.min_touches = 2
        
    def find_pivot_points(self, df, window=5):
        """Find pivot highs and lows"""
        if df is None or df.empty or len(df) < window * 2:
            return [], []
            
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Pivot high
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                highs.append((i, df['high'].iloc[i]))
            
            # Pivot low
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                lows.append((i, df['low'].iloc[i]))
        
        return highs, lows
    
    def identify_levels(self, df):
        """Identify support and resistance levels"""
        try:
            if df is None or df.empty:
                return [], []
                
            highs, lows = self.find_pivot_points(df)
            
            # Group similar levels
            resistance_levels = []
            support_levels = []
            
            # Process resistance levels
            high_prices = [price for _, price in highs]
            for price in high_prices:
                touches = sum(1 for p in high_prices if abs(p - price) / price <= self.touch_threshold)
                if touches >= self.min_touches:
                    resistance_levels.append(price)
            
            # Process support levels
            low_prices = [price for _, price in lows]
            for price in low_prices:
                touches = sum(1 for p in low_prices if abs(p - price) / price <= self.touch_threshold)
                if touches >= self.min_touches:
                    support_levels.append(price)
            
            # Remove duplicates and sort
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
            support_levels = sorted(list(set(support_levels)))
            
            return support_levels, resistance_levels
            
        except Exception as e:
            log(f"❌ Support/Resistance identification error: {e}")
            return [], []
    
    def check_level_confirmation(self, ticker, current_price):
        """Check if current price confirms support/resistance levels"""
        try:
            # Use cached data if available
            if ticker in trading_state.support_resistance_cache:
                support_levels, resistance_levels = trading_state.support_resistance_cache[ticker]
            else:
                df = get_enhanced_data(ticker, limit=200)
                if df is None:
                    return True  # Default to allow if no data
                    
                support_levels, resistance_levels = self.identify_levels(df)
                trading_state.support_resistance_cache[ticker] = (support_levels, resistance_levels)
            
            # Check if price is near a strong resistance (avoid buying)
            for resistance in resistance_levels[:3]:  # Check top 3 resistance levels
                if abs(current_price - resistance) / resistance <= 0.005:  # Within 0.5%
                    log(f"🛑 {ticker} near resistance level ${resistance:.2f}")
                    return False
            
            # Check if price is bouncing off support (good for buying)
            for support in support_levels[-3:]:  # Check bottom 3 support levels
                if abs(current_price - support) / support <= 0.005:  # Within 0.5%
                    log(f"✅ {ticker} bouncing off support level ${support:.2f}")
                    return True
            
            return True  # No significant levels nearby
            
        except Exception as e:
            log(f"❌ Support/Resistance confirmation error for {ticker}: {e}")
            return True

sr_analyzer = SupportResistanceAnalyzer()

# === ENHANCED SENTIMENT ANALYSIS WITH TRANSFORMER ENSEMBLE ===
class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.finbert_pipeline = None
        self.sentiment_weights = {
            'news': 0.7,
            'reddit': 0.3
        }
        
    def get_finbert_pipeline(self):
        if self.finbert_pipeline is None:
            try:
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis", 
                    model="ProsusAI/finbert",
                    device=-1  # Use CPU
                )
                log("🧠 FinBERT pipeline loaded")
            except Exception as e:
                log(f"⚠️ Failed to load FinBERT: {e}")
                self.finbert_pipeline = None
        return self.finbert_pipeline
    
    def analyze_with_transformer_ensemble(self, texts):
        """Analyze sentiment using Transformer Ensemble (FinBERT + VADER)"""
        try:
            finbert_score = 0
            vader_score = 0
            
            # FinBERT analysis
            pipeline = self.get_finbert_pipeline()
            if pipeline is not None:
                try:
                    results = pipeline(texts[:5])  # Limit to avoid memory issues
                    scores = []
                    
                    for result in results:
                        if result['label'] == 'positive':
                            scores.append(result['score'])
                        elif result['label'] == 'negative':
                            scores.append(-result['score'])
                        else:  # neutral
                            scores.append(0)
                    
                    finbert_score = np.mean(scores) if scores else 0
                except Exception as e:
                    log(f"⚠️ FinBERT analysis failed: {e}")
            
            # VADER analysis
            try:
                vader_scores = []
                for text in texts[:10]:  # Analyze more with VADER (faster)
                    score = analyzer.polarity_scores(text)['compound']
                    vader_scores.append(score)
                vader_score = np.mean(vader_scores) if vader_scores else 0
            except Exception as e:
                log(f"⚠️ VADER analysis failed: {e}")
            
            # Ensemble scoring (weighted average)
            ensemble_score = (finbert_score * 0.7 + vader_score * 0.3)
            
            return ensemble_score
            
        except Exception as e:
            log(f"❌ Transformer ensemble analysis failed: {e}")
            return 0
    
    def get_news_sentiment(self, ticker):
        """Get news sentiment for ticker"""
        try:
            if ticker in trading_state.sentiment_cache:
                return trading_state.sentiment_cache[ticker].get('news', 0)
            
            if not newsapi:
                return 0
            
            # Fetch news
            articles = newsapi.get_everything(
                q=f"{ticker} stock",
                sort_by='publishedAt',
                page_size=20,
                language='en'
            )
            
            if not articles or not articles.get('articles'):
                return 0
            
            titles = [article.get('title', '') for article in articles['articles'] if article.get('title')]
            descriptions = [article.get('description', '') for article in articles['articles'] if article.get('description')]
            
            all_texts = titles + descriptions
            if not all_texts:
                return 0
            
            sentiment_score = self.analyze_with_transformer_ensemble(all_texts)
            
            # Cache the result
            if ticker not in trading_state.sentiment_cache:
                trading_state.sentiment_cache[ticker] = {}
            trading_state.sentiment_cache[ticker]['news'] = sentiment_score
            
            return sentiment_score
            
        except Exception as e:
            log(f"❌ News sentiment error for {ticker}: {e}")
            return 0
    
    def get_combined_sentiment(self, ticker):
        """Get combined sentiment score"""
        try:
            news_sentiment = self.get_news_sentiment(ticker)
            
            # For now, just use news sentiment
            # Reddit sentiment would require additional API setup
            combined_score = news_sentiment
            
            return combined_score
            
        except Exception as e:
            log(f"❌ Combined sentiment error for {ticker}: {e}")
            return 0
    
    def sentiment_override_check(self, ticker, base_signal):
        """Check if sentiment should override the base trading signal"""
        try:
            sentiment_score = self.get_combined_sentiment(ticker)
            
            # SENTIMENT HOLD OVERRIDE: If VADER < -0.5 or FinBERT = "Negative"
            if sentiment_score < THRESHOLDS['SENTIMENT_HOLD_OVERRIDE']:
                log(f"😡 Sentiment hold override for {ticker}: {sentiment_score:.3f}")
                return False
            
            # Strong positive sentiment boost
            if sentiment_score > 0.3 and base_signal > THRESHOLDS['SHORT_BUY_THRESHOLD']:
                log(f"😊 Positive sentiment boost for {ticker}: {sentiment_score:.3f}")
                return True
            
            # Moderate negative sentiment caution
            if sentiment_score < -0.1 and base_signal < 0.55:
                log(f"😐 Negative sentiment caution for {ticker}: {sentiment_score:.3f}")
                return False
            
            return base_signal > THRESHOLDS['SHORT_BUY_THRESHOLD']
            
        except Exception as e:
            log(f"❌ Sentiment override error for {ticker}: {e}")
            return base_signal > THRESHOLDS['SHORT_BUY_THRESHOLD']

sentiment_analyzer = EnhancedSentimentAnalyzer()

# === META MODEL APPROVAL SYSTEM WITH AUTO-RETRAINING ===
class MetaModelApprovalSystem:
    def __init__(self):
        self.meta_model = None
        self.feature_columns = [
            'base_signal_strength', 'sentiment_score', 'market_regime', 
            'volume_spike', 'vwap_position', 'rsi', 'macd_signal',
            'support_resistance_score', 'sector_momentum', 'time_of_day',
            'day_of_week', 'volatility', 'recent_accuracy', 'atr_ratio',
            'price_momentum', 'obv_trend'
        ]
        self.approval_threshold = 0.6
        self.min_training_samples = 100
        self.last_retrain_date = None
        
    def extract_meta_features(self, ticker, base_signal, market_data):
        """Extract features for meta model decision"""
        try:
            features = {}
            
            # Base signal strength
            features['base_signal_strength'] = base_signal
            
            # Sentiment
            features['sentiment_score'] = sentiment_analyzer.get_combined_sentiment(ticker)
            
            # Market regime
            features['market_regime'] = self.get_market_regime_numeric()
            
            # Volume analysis
            if market_data is not None and not market_data.empty:
                current_volume = market_data['volume'].iloc[-1]
                avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
                features['volume_spike'] = min(current_volume / avg_volume, 5.0) if avg_volume > 0 else 1.0
                
                # VWAP position
                if 'vwap' in market_data.columns:
                    features['vwap_position'] = market_data['close'].iloc[-1] / market_data['vwap'].iloc[-1] - 1
                else:
                    features['vwap_position'] = 0
                
                # Technical indicators
                features['rsi'] = market_data.get('rsi', pd.Series([50])).iloc[-1] / 100.0
                features['macd_signal'] = 1 if market_data.get('macd', pd.Series([0])).iloc[-1] > 0 else 0
                features['volatility'] = market_data['volatility'].iloc[-1] if 'volatility' in market_data.columns else 0.02
                
                # ATR ratio
                if 'atr' in market_data.columns:
                    features['atr_ratio'] = market_data['atr'].iloc[-1] / market_data['close'].iloc[-1]
                else:
                    features['atr_ratio'] = 0.02
                
                # Price momentum
                features['price_momentum'] = market_data.get('price_momentum', pd.Series([0])).iloc[-1]
                
                # OBV trend
                if 'obv' in market_data.columns and len(market_data) >= 5:
                    obv_trend = (market_data['obv'].iloc[-1] - market_data['obv'].iloc[-5]) / market_data['obv'].iloc[-5]
                    features['obv_trend'] = obv_trend
                else:
                    features['obv_trend'] = 0
                    
            else:
                features.update({
                    'volume_spike': 1.0, 'vwap_position': 0, 'rsi': 0.5,
                    'macd_signal': 0, 'volatility': 0.02, 'atr_ratio': 0.02,
                    'price_momentum': 0, 'obv_trend': 0
                })
            
            # Support/Resistance score
            features['support_resistance_score'] = self.calculate_sr_score(ticker, market_data)
            
            # Sector momentum
            features['sector_momentum'] = self.get_sector_momentum(ticker)
            
            # Time features
            now = datetime.now()
            features['time_of_day'] = now.hour + now.minute / 60.0
            features['day_of_week'] = now.weekday()
            
            # Recent accuracy for this ticker
            features['recent_accuracy'] = self.get_recent_accuracy(ticker)
            
            return features
            
        except Exception as e:
            log(f"❌ Meta feature extraction error for {ticker}: {e}")
            return None
    
    def calculate_sr_score(self, ticker, market_data):
        """Calculate support/resistance score"""
        try:
            if market_data is None or market_data.empty:
                return 0.5
                
            current_price = market_data['close'].iloc[-1]
            return 1.0 if sr_analyzer.check_level_confirmation(ticker, current_price) else 0.0
            
        except:
            return 0.5
    
    def get_sector_momentum(self, ticker):
        """Get sector momentum score"""
        try:
            sector = self.get_ticker_sector(ticker)
            if sector == "Unknown":
                return 0.5
                
            # Calculate sector average performance
            sector_tickers = SECTOR_UNIVERSE.get(sector, [ticker])
            sector_returns = []
            
            for t in sector_tickers[:5]:  # Sample 5 tickers from sector
                try:
                    df = get_enhanced_data(t, limit=20)
                    if df is not None and not df.empty:
                        ret = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                        sector_returns.append(ret)
                except:
                    continue
            
            if sector_returns:
                avg_return = np.mean(sector_returns)
                return max(0, min(1, avg_return * 10 + 0.5))  # Normalize to 0-1
            
            return 0.5
            
        except:
            return 0.5
    
    def get_ticker_sector(self, ticker):
        """Get sector for ticker"""
        for sector, tickers in SECTOR_UNIVERSE.items():
            if ticker in tickers:
                return sector
        return "Unknown"
    
    def get_market_regime_numeric(self):
        """Get numeric market regime"""
        regime_map = {"bull": 1, "bear": -1, "neutral": 0}
        return regime_map.get(trading_state.regime_state, 0)
    
    def get_recent_accuracy(self, ticker):
        """Get recent prediction accuracy for ticker"""
        try:
            if ticker in trading_state.accuracy_tracker:
                recent_trades = trading_state.accuracy_tracker[ticker][-10:]  # Last 10 trades
                if recent_trades:
                    correct_predictions = sum(1 for trade in recent_trades if trade['correct'])
                    return correct_predictions / len(recent_trades)
            return 0.5  # Default neutral accuracy
            
        except:
            return 0.5
    
    def load_meta_model(self):
        """Load existing meta model"""
        try:
            if os.path.exists("models/meta/approval_model.pkl"):
                self.meta_model = joblib.load("models/meta/approval_model.pkl")
                log("✅ Meta approval model loaded")
                return True
            return False
        except Exception as e:
            log(f"❌ Failed to load meta model: {e}")
            return False
    
    def train_meta_model(self):
        """Train the meta approval model with auto-retraining"""
        try:
            # Load historical meta decisions
            if not os.path.exists("logs/meta_decisions.csv"):
                log("⚠️ No meta decision history found")
                return False
            
            df = pd.read_csv("logs/meta_decisions.csv")
            
            if len(df) < self.min_training_samples:
                log(f"⚠️ Not enough samples for meta model training: {len(df)}")
                return False
            
            # Prepare features and target
            X = df[self.feature_columns]
            y = df['outcome']  # 1 for profitable trades, 0 for losses
            
            # Train ensemble meta model
            self.meta_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Cross-validation with TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.meta_model, X, y, cv=tscv, scoring='accuracy')
            log(f"📊 Meta model CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Train on full dataset
            self.meta_model.fit(X, y)
            
            # Save model
            joblib.dump(self.meta_model, "models/meta/approval_model.pkl")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.meta_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            log("📈 Meta model feature importance:")
            for _, row in feature_importance.head(5).iterrows():
                log(f"   {row['feature']}: {row['importance']:.3f}")
            
            # Log performance
            log_to_google_sheets({
                'model_type': 'meta_model',
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(df)
            }, "ModelPerformance")
            
            self.last_retrain_date = datetime.now().date()
            return True
            
        except Exception as e:
            log(f"❌ Meta model training failed: {e}")
            return False
    
    def should_retrain_daily(self):
        """Check if daily auto-retraining is needed"""
        today = datetime.now().date()
        return (self.last_retrain_date is None or 
                self.last_retrain_date < today)
    
    def should_approve_trade(self, ticker, base_signal, market_data):
        """Main approval decision function"""
        try:
            # Extract features
            features = self.extract_meta_features(ticker, base_signal, market_data)
            if features is None:
                return False, "Feature extraction failed"
            
            # Load model if not loaded
            if self.meta_model is None:
                if not self.load_meta_model():
                    # If no model exists, use rule-based approval
                    return self.rule_based_approval(features)
            
            # Create feature vector
            feature_vector = [features.get(col, 0) for col in self.feature_columns]
            feature_df = pd.DataFrame([feature_vector], columns=self.feature_columns)
            
            # Get prediction probability
            approval_prob = self.meta_model.predict_proba(feature_df)[0][1]
            
            # Decision
            approved = approval_prob >= self.approval_threshold
            reason = f"Meta model probability: {approval_prob:.3f}"
            
            # Log decision
            self.log_meta_decision(ticker, features, approved, reason)
            
            return approved, reason
            
        except Exception as e:
            log(f"❌ Meta approval error for {ticker}: {e}")
            return False, f"Error: {e}"
    
    def rule_based_approval(self, features):
        """Fallback rule-based approval when no model exists"""
        try:
            score = 0
            reasons = []
            
            # Base signal strength
            if features['base_signal_strength'] > 0.7:
                score += 2
                reasons.append("Strong base signal")
            elif features['base_signal_strength'] > THRESHOLDS['SHORT_BUY_THRESHOLD']:
                score += 1
                reasons.append("Good base signal")
            
            # Sentiment
            if features['sentiment_score'] > 0.2:
                score += 1
                reasons.append("Positive sentiment")
            elif features['sentiment_score'] < THRESHOLDS['SENTIMENT_HOLD_OVERRIDE']:
                score -= 2
                reasons.append("Negative sentiment override")
            
            # Volume spike
            if features['volume_spike'] > THRESHOLDS['VOLUME_SPIKE_MIN']:
                score += 1
                reasons.append("Volume spike")
            
            # VWAP position
            if features['vwap_position'] > 0:
                score += 1
                reasons.append("Above VWAP")
            
            # Price momentum
            if features['price_momentum'] > THRESHOLDS['PRICE_MOMENTUM_MIN']:
                score += 1
                reasons.append("Good momentum")
            
            # Support/Resistance
            if features['support_resistance_score'] > 0.5:
                score += 1
                reasons.append("Good S/R position")
            
            # Market regime
            if features['market_regime'] > 0:
                score += 1
                reasons.append("Bull market")
            elif features['market_regime'] < 0:
                score -= 1
                reasons.append("Bear market")
            
            approved = score >= 4
            reason = f"Rule-based score: {score} ({', '.join(reasons)})"
            
            return approved, reason
            
        except Exception as e:
            return False, f"Rule-based error: {e}"
    
    def log_meta_decision(self, ticker, features, approved, reason):
        """Log meta model decision for future training"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'approved': approved,
                'reason': reason,
                **features
            }
            
            # Append to CSV
            df = pd.DataFrame([log_entry])
            csv_path = "logs/meta_decisions.csv"
            
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, index=False)
                
        except Exception as e:
            log(f"❌ Failed to log meta decision: {e}")
    
    def update_outcome(self, ticker, timestamp, profitable):
        """Update the outcome of a previous decision for training"""
        try:
            csv_path = "logs/meta_decisions.csv"
            if not os.path.exists(csv_path):
                return
            
            df = pd.read_csv(csv_path)
            
            # Find the decision to update
            mask = (df['ticker'] == ticker) & (df['timestamp'] == timestamp)
            if mask.any():
                df.loc[mask, 'outcome'] = 1 if profitable else 0
                df.to_csv(csv_path, index=False)
                log(f"✅ Updated outcome for {ticker}: {'Profitable' if profitable else 'Loss'}")
                
        except Exception as e:
            log(f"❌ Failed to update meta outcome: {e}")

meta_approval_system = MetaModelApprovalSystem()

# === DYNAMIC WATCHLIST OPTIMIZER WITH SECTOR LIMITS ===
class DynamicWatchlistOptimizer:
    def __init__(self):
        self.max_watchlist_size = THRESHOLDS['WATCHLIST_LIMIT']  # 20
        self.sector_limits = {sector: THRESHOLDS['MAX_PER_SECTOR_WATCHLIST'] 
                             for sector in SECTOR_UNIVERSE.keys()}  # 12 per sector
        self.scoring_weights = {
            'momentum': 0.25,
            'volume': 0.20,
            'volatility': 0.15,
            'sentiment': 0.15,
            'technical': 0.15,
            'sector_rotation': 0.10
        }
        
    def calculate_momentum_score(self, df):
        """Calculate momentum score for a ticker"""
        try:
            if df is None or df.empty or len(df) < 20:
                return 0
            
            # Multiple timeframe momentum
            returns_1d = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            returns_3d = (df['close'].iloc[-1] - df['close'].iloc[-60]) / df['close'].iloc[-60] if len(df) >= 60 else returns_1d
            returns_5d = (df['close'].iloc[-1] - df['close'].iloc[-100]) / df['close'].iloc[-100] if len(df) >= 100 else returns_1d
            
            # Weighted momentum score
            momentum_score = (
                returns_1d * 0.5 +
                returns_3d * 0.3 +
                returns_5d * 0.2
            )
            
            # Normalize to 0-1 scale
            return max(0, min(1, momentum_score * 5 + 0.5))
            
        except Exception as e:
            log(f"❌ Momentum calculation error: {e}")
            return 0
    
    def calculate_volume_score(self, df):
        """Calculate volume score for a ticker"""
        try:
            if df is None or df.empty or len(df) < 20:
                return 0
            
            current_volume = df['volume'].iloc[-1]
            avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
            avg_volume_5 = df['volume'].rolling(5).mean().iloc[-1]
            
            # Volume trend and spike
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            volume_trend = avg_volume_5 / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Combined volume score
            volume_score = (volume_ratio * 0.6 + volume_trend * 0.4) / 3  # Normalize
            
            return max(0, min(1, volume_score))
            
        except Exception as e:
            log(f"❌ Volume calculation error: {e}")
            return 0
    
    def calculate_volatility_score(self, df):
        """Calculate volatility score (higher volatility = higher opportunity)"""
        try:
            if df is None or df.empty or len(df) < 20:
                return 0
            
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Normalize volatility (target range 0.01-0.05 daily volatility)
            normalized_vol = max(0, min(1, (volatility - 0.005) / 0.045))
            
            return normalized_vol
            
        except Exception as e:
            log(f"❌ Volatility calculation error: {e}")
            return 0
    
    def calculate_technical_score(self, df):
        """Calculate technical analysis score"""
        try:
            if df is None or df.empty or len(df) < 50:
                return 0
            
            score = 0
            
            # RSI (prefer 30-70 range, avoid extremes)
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if 30 <= rsi <= 70:
                    score += 0.3
                elif 40 <= rsi <= 60:
                    score += 0.5
            
            # MACD signal
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
                if macd_diff > 0:
                    score += 0.3
            
            # Price vs moving averages
            if len(df) >= 20:
                ma_20 = df['close'].rolling(20).mean().iloc[-1]
                if df['close'].iloc[-1] > ma_20:
                    score += 0.2
            
            # Bollinger Bands position
            if 'bb_position' in df.columns:
                bb_position = df['bb_position'].iloc[-1]
                # Prefer middle range, avoid extremes
                if 0.2 <= bb_position <= 0.8:
                    score += 0.2
            
            return min(1, score)
            
        except Exception as e:
            log(f"❌ Technical score calculation error: {e}")
            return 0
    
    def calculate_sector_rotation_score(self, ticker):
        """Calculate sector rotation score"""
        try:
            sector = self.get_ticker_sector(ticker)
            if sector == "Unknown":
                return 0.5
            
            # Get sector performance
            sector_tickers = SECTOR_UNIVERSE.get(sector, [])
            if not sector_tickers:
                return 0.5
            
            sector_returns = []
            for t in random.sample(sector_tickers, min(5, len(sector_tickers))):
                try:
                    df = get_enhanced_data(t, limit=20)
                    if df is not None and not df.empty:
                        ret = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                        sector_returns.append(ret)
                except:
                    continue
            
            if not sector_returns:
                return 0.5
            
            avg_sector_return = np.mean(sector_returns)
            
            # Compare to market (SPY)
            try:
                spy_df = get_enhanced_data("SPY", limit=20)
                if spy_df is not None and not spy_df.empty:
                    spy_return = (spy_df['close'].iloc[-1] - spy_df['close'].iloc[0]) / spy_df['close'].iloc[0]
                    relative_performance = avg_sector_return - spy_return
                    
                    # Normalize to 0-1
                    return max(0, min(1, relative_performance * 10 + 0.5))
            except:
                pass
            
            return max(0, min(1, avg_sector_return * 10 + 0.5))
            
        except Exception as e:
            log(f"❌ Sector rotation score error: {e}")
            return 0.5
    
    def get_ticker_sector(self, ticker):
        """Get sector for ticker"""
        for sector, tickers in SECTOR_UNIVERSE.items():
            if ticker in tickers:
                return sector
        return "Unknown"
    
    def score_ticker(self, ticker):
        """Calculate comprehensive score for a ticker"""
        try:
            # Get market data
            df = get_enhanced_data(ticker, limit=100)
            if df is None or df.empty:
                return 0, "No data available"
            
            # Calculate individual scores
            momentum_score = self.calculate_momentum_score(df)
            volume_score = self.calculate_volume_score(df)
            volatility_score = self.calculate_volatility_score(df)
            technical_score = self.calculate_technical_score(df)
            sector_score = self.calculate_sector_rotation_score(ticker)
            
            # Get sentiment score
            sentiment_score = sentiment_analyzer.get_combined_sentiment(ticker)
            sentiment_normalized = max(0, min(1, sentiment_score + 0.5))  # Normalize -1,1 to 0,1
            
            # Calculate weighted final score
            final_score = (
                momentum_score * self.scoring_weights['momentum'] +
                volume_score * self.scoring_weights['volume'] +
                volatility_score * self.scoring_weights['volatility'] +
                sentiment_normalized * self.scoring_weights['sentiment'] +
                technical_score * self.scoring_weights['technical'] +
                sector_score * self.scoring_weights['sector_rotation']
            )
            
            score_breakdown = {
                'momentum': momentum_score,
                'volume': volume_score,
                'volatility': volatility_score,
                'sentiment': sentiment_normalized,
                'technical': technical_score,
                'sector': sector_score,
                'final': final_score
            }
            
            return final_score, score_breakdown
            
        except Exception as e:
            log(f"❌ Ticker scoring error for {ticker}: {e}")
            return 0, f"Error: {e}"
    
    def generate_dynamic_watchlist(self):
        """Generate optimized watchlist with sector concentration limits"""
        try:
            log("🎯 Generating dynamic watchlist...")
            
            candidates = []
            
            # Score all tickers
            all_tickers = FALLBACK_UNIVERSE.copy()
            random.shuffle(all_tickers)  # Randomize to avoid bias
            
            for ticker in all_tickers[:100]:  # Limit to first 100 to avoid timeout
                try:
                    score, breakdown = self.score_ticker(ticker)
                    
                    if score > 0.3:  # Minimum threshold
                        sector = self.get_ticker_sector(ticker)
                        
                        candidates.append({
                            'ticker': ticker,
                            'score': score,
                            'sector': sector,
                            'breakdown': breakdown
                        })
                        
                        log(f"📊 {ticker}: {score:.3f} ({sector})")
                    
                except Exception as e:
                    log(f"⚠️ Error scoring {ticker}: {e}")
                    continue
            
            # Sort by score
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # Apply sector limits (MAX_PER_SECTOR_WATCHLIST = 12)
            final_watchlist = []
            sector_counts = defaultdict(int)
            
            for candidate in candidates:
                ticker = candidate['ticker']
                sector = candidate['sector']
                
                # Check sector limit
                sector_limit = self.sector_limits.get(sector, THRESHOLDS['MAX_PER_SECTOR_WATCHLIST'])
                if sector_counts[sector] >= sector_limit:
                    continue
                
                # Check total limit
                if len(final_watchlist) >= self.max_watchlist_size:
                    break
                
                final_watchlist.append(candidate)
                sector_counts[sector] += 1
            
            # Log final watchlist
            log(f"✅ Generated watchlist with {len(final_watchlist)} tickers:")
            for item in final_watchlist[:10]:  # Show top 10
                log(f"   {item['ticker']}: {item['score']:.3f} ({item['sector']})")
            
            # Log sector distribution
            log("📈 Sector distribution:")
            for sector, count in sector_counts.items():
                log(f"   {sector}: {count}")
            
            return [item['ticker'] for item in final_watchlist]
            
        except Exception as e:
            log(f"❌ Watchlist generation failed: {e}")
            return FALLBACK_UNIVERSE[:20]  # Fallback to first 20 tickers

watchlist_optimizer = DynamicWatchlistOptimizer()

# === ENHANCED TRADING EXECUTOR WITH ATR-BASED STOPS ===
class EnhancedTradingExecutor:
    def __init__(self):
        self.cooldown_periods = {}  # Track cooldowns per ticker
        self.position_tracker = {}  # Track position details
        self.stop_loss_targets = {}  # Dynamic stop-loss tracking
        self.profit_targets = {}  # Dynamic profit targets
        self.trade_start_times = {}  # Track when positions were opened
        
        # Configuration
        self.max_positions = 10
        self.cooldown_duration = 300  # 5 minutes in seconds
        
    def check_cooldown(self, ticker):
        """Check if ticker is in cooldown period"""
        if ticker not in self.cooldown_periods:
            return True
            
        cooldown_end = self.cooldown_periods[ticker]
        if datetime.now().timestamp() > cooldown_end:
            del self.cooldown_periods[ticker]
            return True
            
        return False
    
    def set_cooldown(self, ticker):
        """Set cooldown period for ticker"""
        self.cooldown_periods[ticker] = datetime.now().timestamp() + self.cooldown_duration
        log(f"⏰ Cooldown set for {ticker}: {self.cooldown_duration/60:.1f} minutes")
    
    def calculate_atr_based_stops(self, ticker, entry_price, atr_value, confidence):
        """Calculate ATR-based dynamic stop-loss and take-profit"""
        try:
            # Dynamic stop-loss: ATR * 1.2 (or lower if confidence is low)
            confidence_multiplier = max(0.7, min(1.3, confidence))
            stop_multiplier = THRESHOLDS['ATR_STOP_MULTIPLIER'] / confidence_multiplier
            stop_loss = atr_value * stop_multiplier
            stop_price = entry_price - stop_loss
            
            # Dynamic take-profit: ATR * 2-3 based on confidence
            profit_multiplier = THRESHOLDS['ATR_PROFIT_MULTIPLIER'] * confidence_multiplier
            take_profit = atr_value * profit_multiplier
            target_price = entry_price + take_profit
            
            log(f"🎯 ATR-based targets for {ticker}:")
            log(f"   Stop-loss: ${stop_price:.2f} (ATR: {atr_value:.2f} * {stop_multiplier:.2f})")
            log(f"   Take-profit: ${target_price:.2f} (ATR: {atr_value:.2f} * {profit_multiplier:.2f})")
            
            return stop_price, target_price, stop_loss / entry_price, take_profit / entry_price
            
        except Exception as e:
            log(f"❌ ATR-based calculation failed: {e}")
            # Fallback to percentage-based
            stop_price = entry_price * 0.97  # 3% stop
            target_price = entry_price * 1.05  # 5% target
            return stop_price, target_price, 0.03, 0.05
    
    def check_profit_decay_exit(self, ticker, entry_price, current_price, entry_time):
        """Check if position should be exited due to profit decay"""
        try:
            if ticker not in self.trade_start_times:
                return False
            
            # Calculate time held
            time_held = (datetime.now() - entry_time).total_seconds() / 60  # minutes
            
            if time_held < 60:  # Less than 1 hour
                return False
            
            # Calculate current profit
            current_profit = (current_price - entry_price) / entry_price
            
            # Profit decay logic: exit when gain momentum fades
            if 0 < current_profit < 0.01 and time_held > 120:  # Less than 1% profit after 2 hours
                log(f"💸 Profit decay exit triggered for {ticker}: {current_profit:.2%} after {time_held:.0f} min")
                return True
            
            # Also check if profit is declining
            if ticker in self.position_tracker:
                tracking_data = self.position_tracker[ticker]
                if 'max_profit' not in tracking_data:
                    tracking_data['max_profit'] = current_profit
                else:
                    if current_profit > tracking_data['max_profit']:
                        tracking_data['max_profit'] = current_profit
                    else:
                        # Profit is declining - check if we should exit
                        profit_decline = tracking_data['max_profit'] - current_profit
                        if profit_decline > 0.015 and time_held > 90:  # 1.5% decline after 1.5 hours
                            log(f"📉 Profit decline exit for {ticker}: max {tracking_data['max_profit']:.2%} -> {current_profit:.2%}")
                            return True
            
            return False
            
        except Exception as e:
            log(f"❌ Profit decay check failed: {e}")
            return False
    
    def check_portfolio_risk_limits(self):
        """Check portfolio-level risk controls"""
        try:
            # Check drawdown limit
            if not check_drawdown_limit():
                return False, "Daily drawdown limit exceeded"
            
            # Check sector exposure limits
            current_positions = get_current_positions()
            account = api.get_account()
            portfolio_value = float(account.equity)
            
            sector_exposure = defaultdict(float)
            for ticker, position in current_positions.items():
                sector = self.get_ticker_sector(ticker)
                sector_exposure[sector] += position['market_value']
            
            # Check if any sector exceeds 30% allocation
            for sector, exposure in sector_exposure.items():
                sector_pct = exposure / portfolio_value
                if sector_pct > THRESHOLDS['MAX_PER_SECTOR_PORTFOLIO']:
                    return False, f"Sector {sector} exposure limit exceeded: {sector_pct:.1%}"
            
            return True, "Risk limits OK"
            
        except Exception as e:
            log(f"❌ Portfolio risk check failed: {e}")
            return True, "Risk check failed"
    
    def execute_buy_order(self, ticker, signal_strength, market_data):
        """Execute buy order with enhanced logic"""
        try:
            # Pre-execution checks
            if not self.check_cooldown(ticker):
                log(f"⏰ {ticker} in cooldown period, skipping")
                return False
            
            # Portfolio risk limits
            risk_ok, risk_msg = self.check_portfolio_risk_limits()
            if not risk_ok:
                log(f"🚫 {risk_msg}, skipping {ticker}")
                return False
            
            current_positions = get_current_positions()
            if len(current_positions) >= self.max_positions:
                log(f"🚫 Maximum positions reached ({self.max_positions}), skipping {ticker}")
                return False
            
            if ticker in current_positions:
                log(f"📌 Already holding {ticker}, skipping buy")
                return False
            
            # Check sector allocation
            if not self.check_sector_allocation(ticker):
                log(f"🚫 Sector allocation limit reached for {ticker}")
                return False
            
            # Get current price and ATR
            bar = api.get_latest_bar(ticker)
            current_price = bar.c
            
            atr_value = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else current_price * 0.02
            
            # Calculate position size using Dynamic Kelly Criterion
            position_size = calculate_dynamic_kelly_position_size(ticker, confidence=signal_strength)
            
            # Calculate ATR-based stop-loss and profit targets
            stop_price, target_price, stop_pct, target_pct = self.calculate_atr_based_stops(
                ticker, current_price, atr_value, signal_strength
            )
            
            # Execute the buy order
            order = api.submit_order(
                symbol=ticker,
                qty=position_size,
                side="buy",
                type="market",
                time_in_force="gtc"
            )
            
            # Track position details
            self.position_tracker[ticker] = {
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'stop_price': stop_price,
                'target_price': target_price,
                'quantity': position_size,
                'signal_strength': signal_strength,
                'atr_value': atr_value,
                'max_profit': 0.0
            }
            
            self.trade_start_times[ticker] = datetime.now()
            
            # Update sector allocation
            self.update_sector_allocation(ticker)
            
            # Log the trade
            trade_data = {
                'action': 'BUY',
                'ticker': ticker,
                'price': current_price,
                'quantity': position_size,
                'signal_strength': signal_strength,
                'stop_loss_pct': stop_pct,
                'profit_target_pct': target_pct,
                'atr_value': atr_value
            }
            
            log_to_google_sheets(trade_data, "Trades")
            
            # Send alerts
            alert_msg = f"✅ BUY {position_size} {ticker} @ ${current_price:.2f}\n"
            alert_msg += f"Signal: {signal_strength:.3f} | Stop: {stop_pct:.1%} | Target: {target_pct:.1%}"
            send_discord_alert(alert_msg)
            
            log(f"✅ Buy order executed: {position_size} shares of {ticker} @ ${current_price:.2f}")
            
            return True
            
        except Exception as e:
            log(f"❌ Buy order failed for {ticker}: {e}")
            send_discord_alert(f"❌ Buy order failed for {ticker}: {e}", urgent=True)
            self.set_cooldown(ticker)
            return False
    
    def execute_sell_order(self, ticker, reason="Manual"):
        """Execute sell order with enhanced logic"""
        try:
            current_positions = get_current_positions()
            if ticker not in current_positions:
                log(f"⚠️ No position found for {ticker}")
                return False
            
            position = current_positions[ticker]
            quantity = position['qty']
            current_price = position['current_price']
            entry_price = position['avg_entry_price']
            
            # Calculate profit/loss
            profit_loss = (current_price - entry_price) / entry_price
            
            # Execute sell order
            order = api.submit_order(
                symbol=ticker,
                qty=quantity,
                side="sell",
                type="market",
                time_in_force="gtc"
            )
            
            # Determine outcome
            if profit_loss > 0.005:  # More than 0.5% profit
                outcome = "profit"
            elif profit_loss < -0.015:  # More than 1.5% loss
                outcome = "loss"
            else:
                outcome = "breakeven"
            
            # Update tracking
            if ticker in self.position_tracker:
                del self.position_tracker[ticker]
            if ticker in self.trade_start_times:
                del self.trade_start_times[ticker]
            
            # Set cooldown
            self.set_cooldown(ticker)
            
            # Log the trade
            trade_data = {
                'action': 'SELL',
                'ticker': ticker,
                'price': current_price,
                'quantity': quantity,
                'profit_loss': profit_loss,
                'reason': reason,
                'outcome': outcome
            }
            
            log_to_google_sheets(trade_data, "Trades")
            
            # Update accuracy tracking
            self.update_accuracy_tracking(ticker, outcome == "profit")
            
            # Send alerts
            alert_msg = f"🔻 SELL {quantity} {ticker} @ ${current_price:.2f}\n"
            alert_msg += f"P&L: {profit_loss:.2%} | Reason: {reason}"
            
            if outcome == "profit":
                send_discord_alert(f"💰 {alert_msg}")
            elif outcome == "loss":
                send_discord_alert(f"📉 {alert_msg}")
            else:
                send_discord_alert(alert_msg)
            
            log(f"🔻 Sell order executed: {quantity} shares of {ticker} @ ${current_price:.2f} ({profit_loss:.2%})")
            
            return True
            
        except Exception as e:
            log(f"❌ Sell order failed for {ticker}: {e}")
            send_discord_alert(f"❌ Sell order failed for {ticker}: {e}", urgent=True)
            return False
    
    def check_exit_conditions(self):
        """Check all positions for exit conditions"""
        try:
            current_positions = get_current_positions()
            
            for ticker, position in current_positions.items():
                current_price = position['current_price']
                entry_price = position['avg_entry_price']
                profit_loss = (current_price - entry_price) / entry_price
                
                # Get position tracking data
                if ticker not in self.position_tracker:
                    continue
                
                tracking_data = self.position_tracker[ticker]
                stop_price = tracking_data['stop_price']
                target_price = tracking_data['target_price']
                entry_time = tracking_data['entry_time']
                
                # Check stop-loss
                if current_price <= stop_price:
                    log(f"🛑 Stop-loss triggered for {ticker}: ${current_price:.2f} <= ${stop_price:.2f}")
                    self.execute_sell_order(ticker, "Stop-loss")
                    continue
                
                # Check profit target
                if current_price >= target_price:
                    log(f"🎯 Profit target hit for {ticker}: ${current_price:.2f} >= ${target_price:.2f}")
                    self.execute_sell_order(ticker, "Profit target")
                    continue
                
                # Check profit decay
                if self.check_profit_decay_exit(ticker, entry_price, current_price, entry_time):
                    self.execute_sell_order(ticker, "Profit decay")
                    continue
                
                # Trailing stop logic
                if profit_loss > 0.03:  # If up more than 3%, implement trailing stop
                    atr_value = tracking_data.get('atr_value', current_price * 0.02)
                    trailing_stop = current_price - (atr_value * 1.5)  # 1.5x ATR trailing
                    
                    if trailing_stop > stop_price:
                        self.position_tracker[ticker]['stop_price'] = trailing_stop
                        log(f"📈 Trailing stop adjusted for {ticker}: ${trailing_stop:.2f}")
                
        except Exception as e:
            log(f"❌ Exit condition check failed: {e}")
    
    def check_sector_allocation(self, ticker):
        """Check if sector allocation allows new position"""
        try:
            sector = self.get_ticker_sector(ticker)
            current_positions = get_current_positions()
            
            # Count current positions in this sector
            sector_count = 0
            for pos_ticker in current_positions:
                if self.get_ticker_sector(pos_ticker) == sector:
                    sector_count += 1
            
            max_per_sector = 3  # Maximum 3 positions per sector
            return sector_count < max_per_sector
            
        except Exception as e:
            log(f"❌ Sector allocation check failed: {e}")
            return True
    
    def update_sector_allocation(self, ticker):
        """Update sector allocation tracking"""
        try:
            sector = self.get_ticker_sector(ticker)
            if sector not in trading_state.sector_allocations:
                trading_state.sector_allocations[sector] = 0
            trading_state.sector_allocations[sector] += 1
            
        except Exception as e:
            log(f"❌ Sector allocation update failed: {e}")
    
    def get_ticker_sector(self, ticker):
        """Get sector for ticker"""
        for sector, tickers in SECTOR_UNIVERSE.items():
            if ticker in tickers:
                return sector
        return "Unknown"
    
    def update_accuracy_tracking(self, ticker, was_profitable):
        """Update accuracy tracking for meta model training"""
        try:
            if ticker not in trading_state.accuracy_tracker:
                trading_state.accuracy_tracker[ticker] = []
            
            trading_state.accuracy_tracker[ticker].append({
                'timestamp': datetime.now().isoformat(),
                'correct': was_profitable
            })
            
            # Keep only last 50 trades per ticker
            if len(trading_state.accuracy_tracker[ticker]) > 50:
                trading_state.accuracy_tracker[ticker] = trading_state.accuracy_tracker[ticker][-50:]
                
        except Exception as e:
            log(f"❌ Accuracy tracking update failed: {e}")
    
    def liquidate_all_positions(self, reason="End of day"):
        """Liquidate all positions"""
        try:
            current_positions = get_current_positions()
            
            if not current_positions:
                log("📭 No positions to liquidate")
                return
            
            log(f"🔻 Liquidating {len(current_positions)} positions: {reason}")
            
            for ticker in current_positions:
                self.execute_sell_order(ticker, reason)
                time.sleep(1)  # Small delay between orders
            
            # Clear tracking data
            self.position_tracker.clear()
            self.trade_start_times.clear()
            
            send_discord_alert(f"🔻 All positions liquidated: {reason}")
            
        except Exception as e:
            log(f"❌ Liquidation failed: {e}")
            send_discord_alert(f"❌ Liquidation failed: {e}", urgent=True)

trading_executor = EnhancedTradingExecutor()

# === Q-LEARNING SYSTEM WITH REWARD SHAPING ===
class QLearningSystem:
    def __init__(self):
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        
    def calculate_reward(self, ticker, action, profit_loss, sentiment_score, volatility):
        """Calculate reward with sentiment + volatility factors"""
        try:
            base_reward = profit_loss * 100  # Scale profit/loss
            
            # Sentiment factor
            sentiment_factor = 1.0
            if sentiment_score > 0.2:
                sentiment_factor = 1.2  # Boost for positive sentiment
            elif sentiment_score < -0.2:
                sentiment_factor = 0.8  # Penalty for negative sentiment
            
            # Volatility factor
            volatility_factor = 1.0
            if volatility > 0.05:  # High volatility
                volatility_factor = 0.9  # Slight penalty
            elif volatility < 0.02:  # Low volatility
                volatility_factor = 1.1  # Slight boost
            
            # Action-specific rewards
            if action == 0:  # Buy
                if profit_loss > 0:
                    reward = base_reward * sentiment_factor * volatility_factor
                else:
                    reward = base_reward * 0.5  # Reduce penalty for losses
            elif action == 1:  # Hold
                reward = 0.1  # Small positive reward for holding
            else:  # Sell
                reward = base_reward * 0.8  # Slightly reduce sell rewards
            
            return reward
            
        except Exception as e:
            log(f"❌ Reward calculation failed: {e}")
            return 0
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        try:
            if len(self.memory) >= self.memory_size:
                self.memory.pop(0)
            
            self.memory.append((state, action, reward, next_state, done))
            
        except Exception as e:
            log(f"❌ Experience storage failed: {e}")
    
    def train_q_network(self):
        """Train Q-network with experience replay"""
        try:
            if len(self.memory) < self.batch_size:
                return
            
            # Sample random batch
            batch = random.sample(self.memory, self.batch_size)
            
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch])
            dones = torch.BoolTensor([e[4] for e in batch])
            
            current_q_values = q_net(states).gather(1, actions.unsqueeze(1))
            next_q_values = q_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            loss = q_criterion(current_q_values.squeeze(), target_q_values)
            
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()
            
            log(f"🧠 Q-network trained: loss={loss.item():.4f}")
            
        except Exception as e:
            log(f"❌ Q-network training failed: {e}")
    
    def get_q_action(self, state):
        """Get action from Q-network"""
        try:
            if random.random() < self.epsilon:
                return random.randint(0, 2)  # Random action (exploration)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_net(state_tensor)
                return q_values.argmax().item()
                
        except Exception as e:
            log(f"❌ Q-action selection failed: {e}")
            return 1  # Default to hold
    
    def save_q_network(self):
        """Save Q-network state"""
        try:
            torch.save({
                'model_state_dict': q_net.state_dict(),
                'optimizer_state_dict': q_optimizer.state_dict(),
            }, "models/q_learning/enhanced_q_net.pth")
            log("💾 Q-network saved")
            
        except Exception as e:
            log(f"❌ Q-network save failed: {e}")

q_learning_system = QLearningSystem()

# === ENHANCED TRADING BOT WITH ALL FEATURES ===
class EnhancedTradingBot:
    def __init__(self):
        self.loop_interval = 300  # 5 minutes (live training loop)
        self.watchlist_refresh_interval = 1800  # 30 minutes
        self.last_watchlist_update = 0
        self.current_watchlist = []
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'accuracy': 0.0
        }
        self.retraining_schedule = {
            'live_per_ticker': 300,  # Every 5 minutes
            'daily_full': 86400,  # 1x per day
            'weekly_deep': 604800  # 1x per week
        }
        self.last_daily_retrain = None
        self.last_weekly_retrain = None
        
    def initialize_bot(self):
        """Initialize the trading bot"""
        try:
            log("🚀 Initializing Enhanced AI Trading Bot...")
            
            # Set starting equity for drawdown tracking
            account = api.get_account()
            trading_state.starting_equity = float(account.equity)
            
            # Detect initial market regime
            detect_market_regime()
            
            # Load existing models
            meta_approval_system.load_meta_model()
            
            # Generate initial watchlist
            self.current_watchlist = watchlist_optimizer.generate_dynamic_watchlist()
            self.last_watchlist_update = time.time()
            
            # Send startup notification
            startup_msg = f"🚀 Enhanced AI Trading Bot Started!\n"
            startup_msg += f"💰 Starting Equity: ${trading_state.starting_equity:,.2f}\n"
            startup_msg += f"📊 Market Regime: {trading_state.regime_state}\n"
            startup_msg += f"🎯 Watchlist: {len(self.current_watchlist)} tickers"
            send_discord_alert(startup_msg)
            
            log("✅ Bot initialization complete")
            return True
            
        except Exception as e:
            log(f"❌ Bot initialization failed: {e}")
            send_discord_alert(f"❌ Bot initialization failed: {e}", urgent=True)
            return False
    
    def should_refresh_watchlist(self):
        """Check if watchlist should be refreshed"""
        return (time.time() - self.last_watchlist_update) > self.watchlist_refresh_interval
    
    def refresh_watchlist(self):
        """Refresh the dynamic watchlist"""
        try:
            log("🔄 Refreshing dynamic watchlist...")
            new_watchlist = watchlist_optimizer.generate_dynamic_watchlist()
            
            # Compare with current watchlist
            added = set(new_watchlist) - set(self.current_watchlist)
            removed = set(self.current_watchlist) - set(new_watchlist)
            
            if added or removed:
                log(f"📝 Watchlist updated: +{len(added)} -{len(removed)}")
                if added:
                    log(f"   Added: {', '.join(list(added)[:5])}")
                if removed:
                    log(f"   Removed: {', '.join(list(removed)[:5])}")
            
            self.current_watchlist = new_watchlist
            self.last_watchlist_update = time.time()
            
        except Exception as e:
            log(f"❌ Watchlist refresh failed: {e}")
    
    def process_ticker(self, ticker):
        """Process a single ticker for trading opportunities"""
        try:
            log(f"🔍 Processing {ticker}")
            
            # Get enhanced market data for both horizons
            short_data = get_enhanced_data(ticker, days_back=2)  # 2-day data, 5-minute candles
            medium_data = get_enhanced_data(ticker, days_back=15)  # 15-day data, daily bars
            
            if short_data is None or short_data.empty:
                log(f"⚠️ No short-term data available for {ticker}")
                return False
            
            if medium_data is None or medium_data.empty:
                log(f"⚠️ No medium-term data available for {ticker}")
                return False
            
            # Get sentiment score for filtering
            sentiment_score = sentiment_analyzer.get_combined_sentiment(ticker)
            
            # Check ALL REQUIRED FILTERS
            filter_passed, filter_reason = vwap_filter.passes_all_filters(ticker, sentiment_score)
            if not filter_passed:
                log(f"⚠️ {ticker} failed filters: {filter_reason}")
                return False
            
            current_price = short_data['close'].iloc[-1]
            if not sr_analyzer.check_level_confirmation(ticker, current_price):
                log(f"⚠️ {ticker} failed Support/Resistance confirmation")
                return False
            
            # Live training loop - train models for this ticker
            success = dual_predictor.train_models(ticker, short_data, medium_data)
            if not success:
                log(f"⚠️ Model training failed for {ticker}")
                return False
            
            # Get current features for prediction
            short_features = short_data.iloc[-1]
            medium_features = medium_data.iloc[-1]
            
            # Required feature columns
            feature_cols = [
                'returns', 'log_returns', 'momentum_5', 'momentum_10', 'momentum_20',
                'price_momentum', 'volume_ratio', 'rsi', 'macd', 'macd_signal', 
                'macd_histogram', 'obv', 'bb_position', 'price_vs_vwap', 
                'volatility', 'atr', 'support_resistance_ratio'
            ]
            
            # Ensure all required features are present
            available_short_features = [col for col in feature_cols if col in short_features.index]
            available_medium_features = [col for col in feature_cols if col in medium_features.index]
            
            if len(available_short_features) < 12 or len(available_medium_features) < 12:
                log(f"⚠️ Insufficient features for {ticker}")
                return False
            
            current_short_data = short_features[available_short_features].values
            current_medium_data = medium_features[available_medium_features].values
            
            # Make dual-horizon predictions - BOTH MUST AGREE
            short_prob, medium_prob, combined_prob, agreement = dual_predictor.predict(
                ticker, current_short_data, current_medium_data
            )
            
            if not agreement:
                log(f"⚠️ {ticker} - Models disagree: Short={short_prob:.3f}, Medium={medium_prob:.3f}")
                return False
            
            log(f"📊 {ticker} predictions - Short: {short_prob:.3f}, Medium: {medium_prob:.3f}, Combined: {combined_prob:.3f}")
            
            # Check confidence thresholds
            if combined_prob < THRESHOLDS['SHORT_SELL_AVOID_THRESHOLD']:
                log(f"⚠️ {ticker} below avoid threshold: {combined_prob:.3f}")
                return False
            
            if combined_prob < THRESHOLDS['SHORT_BUY_THRESHOLD']:
                log(f"⏸️ {ticker} below buy threshold: {combined_prob:.3f}")
                return False
            
            # Sentiment override check
            if not sentiment_analyzer.sentiment_override_check(ticker, combined_prob):
                log(f"😡 Sentiment override rejected {ticker}")
                return False
            
            # Meta model approval
            approved, reason = meta_approval_system.should_approve_trade(ticker, combined_prob, short_data)
            if not approved:
                log(f"🛑 Meta model rejected {ticker}: {reason}")
                return False
            
            log(f"✅ Meta model approved {ticker}: {reason}")
            
            # Q-Learning decision
            q_state = self.prepare_q_state(ticker, short_data, combined_prob, sentiment_score)
            q_action = q_learning_system.get_q_action(q_state)
            
            if q_action != 0:  # 0 = Buy, 1 = Hold, 2 = Sell
                log(f"🤖 Q-Learning suggests action {q_action} for {ticker}")
                return False
            
            # Execute trade if all conditions met
            success = trading_executor.execute_buy_order(ticker, combined_prob, short_data)
            if success:
                self.daily_stats['trades_executed'] += 1
                log(f"✅ Trade executed for {ticker}")
                
                # Store Q-Learning experience
                reward = 0.1  # Initial reward for taking action
                q_learning_system.store_experience(q_state, q_action, reward, q_state, False)
                
                return True
            
            return False
            
        except Exception as e:
            log(f"❌ Error processing {ticker}: {e}")
            return False
    
    def prepare_q_state(self, ticker, market_data, signal_strength, sentiment_score):
        """Prepare state vector for Q-Learning"""
        try:
            state = []
            
            # Price features
            state.append(signal_strength)
            state.append(sentiment_score)
            state.append(market_data['rsi'].iloc[-1] / 100.0)
            state.append(market_data['price_momentum'].iloc[-1])
            state.append(market_data['volume_ratio'].iloc[-1])
            state.append(market_data['volatility'].iloc[-1])
            state.append(market_data['price_vs_vwap'].iloc[-1])
            
            # Market regime
            regime_map = {"bull": 1, "bear": -1, "neutral": 0}
            state.append(regime_map.get(trading_state.regime_state, 0))
            
            # Time features
            now = datetime.now()
            state.append(now.hour / 24.0)
            state.append(now.weekday() / 7.0)
            
            # Portfolio features
            current_positions = get_current_positions()
            state.append(len(current_positions) / 10.0)  # Normalize by max positions
            state.append(trading_state.daily_drawdown)
            
            # Pad to fixed size (20 features)
            while len(state) < 20:
                state.append(0.0)
            
            return state[:20]  # Ensure exactly 20 features
            
        except Exception as e:
            log(f"❌ Q-state preparation failed: {e}")
            return [0.0] * 20
    
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            log("🔄 Starting trading cycle...")
            
            # Update market regime
            detect_market_regime()
            
            # Check exit conditions for existing positions
            trading_executor.check_exit_conditions()
            
            # Refresh watchlist if needed
            if self.should_refresh_watchlist():
                self.refresh_watchlist()
            
            # Process watchlist (limit to avoid timeout)
            processed_count = 0
            successful_trades = 0
            
            for ticker in self.current_watchlist[:15]:  # Process top 15 tickers
                try:
                    if self.process_ticker(ticker):
                        successful_trades += 1
                    processed_count += 1
                    
                    # Small delay between tickers to avoid rate limits
                    time.sleep(3)
                    
                except Exception as e:
                    log(f"❌ Error in trading cycle for {ticker}: {e}")
                    continue
            
            # Train Q-network
            q_learning_system.train_q_network()
            
            log(f"✅ Trading cycle complete: {processed_count} processed, {successful_trades} trades")
            
            # Update daily stats
            self.update_daily_stats()
            
        except Exception as e:
            log(f"❌ Trading cycle failed: {e}")
            send_discord_alert(f"❌ Trading cycle failed: {e}", urgent=True)
    
    def update_daily_stats(self):
        """Update daily trading statistics"""
        try:
            current_positions = get_current_positions()
            
            # Calculate current P&L
            total_unrealized_pl = sum(pos['unrealized_pl'] for pos in current_positions.values())
            
            # Update accuracy if we have completed trades
            if self.daily_stats['trades_executed'] > 0:
                self.daily_stats['accuracy'] = self.daily_stats['profitable_trades'] / self.daily_stats['trades_executed']
            
            # Log stats periodically
            if self.daily_stats['trades_executed'] % 3 == 0 and self.daily_stats['trades_executed'] > 0:
                stats_msg = f"📊 Daily Stats:\n"
                stats_msg += f"Trades: {self.daily_stats['trades_executed']}\n"
                stats_msg += f"Accuracy: {self.daily_stats['accuracy']:.1%}\n"
                stats_msg += f"Unrealized P&L: ${total_unrealized_pl:.2f}\n"
                stats_msg += f"Positions: {len(current_positions)}\n"
                stats_msg += f"Drawdown: {trading_state.daily_drawdown:.1%}\n"
                stats_msg += f"Regime: {trading_state.regime_state}"
                
                log(stats_msg)
                
        except Exception as e:
            log(f"❌ Stats update failed: {e}")
    
    def check_retraining_schedule(self):
        """Check if models need retraining"""
        try:
            now = datetime.now()
            
            # Daily full retrain
            if (self.last_daily_retrain is None or 
                (now - self.last_daily_retrain).total_seconds() > self.retraining_schedule['daily_full']):
                
                if meta_approval_system.should_retrain_daily():
                    log("🔄 Starting daily meta model retraining...")
                    meta_approval_system.train_meta_model()
                    self.last_daily_retrain = now
            
            # Weekly deep retrain
            if (self.last_weekly_retrain is None or 
                (now - self.last_weekly_retrain).total_seconds() > self.retraining_schedule['weekly_deep']):
                
                log("🔄 Starting weekly deep retraining...")
                # Clear model caches for fresh training
                dual_predictor.short_models.clear()
                dual_predictor.medium_models.clear()
                self.last_weekly_retrain = now
                
        except Exception as e:
            log(f"❌ Retraining schedule check failed: {e}")
    
    def end_of_day_cleanup(self):
        """Perform end-of-day cleanup and reporting"""
        try:
            log("🌅 Starting end-of-day cleanup...")
            
            # Auto-liquidation near close
            trading_executor.liquidate_all_positions("End of day")
            
            # Generate daily report
            self.generate_daily_report()
            
            # Meta model training attempt
            if meta_approval_system.should_retrain_daily():
                meta_approval_system.train_meta_model()
            
            # Save Q-network
            q_learning_system.save_q_network()
            
            # Reset daily state
            trading_state.reset_daily()
            self.daily_stats = {
                'trades_executed': 0,
                'profitable_trades': 0,
                'total_pnl': 0.0,
                'accuracy': 0.0
            }
            
            log("✅ End-of-day cleanup complete")
            
        except Exception as e:
            log(f"❌ End-of-day cleanup failed: {e}")
            send_discord_alert(f"❌ End-of-day cleanup failed: {e}", urgent=True)
    
    def generate_daily_report(self):
        """Generate and send daily trading report"""
        try:
            account = api.get_account()
            
            report = "📊 **Daily Trading Report**\n"
            report += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            report += f"💰 Account Value: ${float(account.equity):,.2f}\n"
            report += f"💵 Cash: ${float(account.cash):,.2f}\n"
            report += f"📈 Day P&L: ${float(account.todays_pl):,.2f}\n"
            report += f"📉 Max Drawdown: {trading_state.daily_drawdown:.1%}\n\n"
            
            report += f"🎯 Trades Executed: {self.daily_stats['trades_executed']}\n"
            report += f"✅ Accuracy: {self.daily_stats['accuracy']:.1%}\n"
            report += f"🌊 Market Regime: {trading_state.regime_state}\n"
            report += f"📋 Watchlist Size: {len(self.current_watchlist)}"
            
            # Send to Discord and log to sheets
            send_discord_alert(report)
            log_to_google_sheets({
                'account_value': float(account.equity),
                'cash': float(account.cash),
                'day_pl': float(account.todays_pl),
                'max_drawdown': trading_state.daily_drawdown,
                'trades_executed': self.daily_stats['trades_executed'],
                'accuracy': self.daily_stats['accuracy'],
                'market_regime': trading_state.regime_state
            }, "DailyReports")
            
        except Exception as e:
            log(f"❌ Daily report generation failed: {e}")
    
    def run_main_loop(self):
        """Main trading bot loop"""
        try:
            if not self.initialize_bot():
                return
            
            log("🔄 Starting main trading loop...")
            
            while True:
                try:
                    # Check if market is open
                    if not is_market_open():
                        log("⏳ Market closed, waiting...")
                        time.sleep(300)  # Check every 5 minutes
                        continue
                    
                    # Check if near market close
                    if is_near_market_close(30):
                        log("⏰ Near market close, performing cleanup...")
                        self.end_of_day_cleanup()
                        break
                    
                    # Check retraining schedule
                    self.check_retraining_schedule()
                    
                    # Run trading cycle (every 5 minutes - live training loop)
                    self.run_trading_cycle()
                    
                    # Wait for next cycle
                    log(f"⏸️ Waiting {self.loop_interval/60:.1f} minutes for next cycle...")
                    time.sleep(self.loop_interval)
                    
                except KeyboardInterrupt:
                    log("🛑 Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    log(f"❌ Error in main loop: {e}")
                    send_discord_alert(f"❌ Main loop error: {e}", urgent=True)
                    time.sleep(60)  # Wait 1 minute before retrying
                    continue
            
            log("🏁 Trading bot stopped")
            send_discord_alert("🏁 Trading bot stopped")
            
        except Exception as e:
            log(f"❌ Fatal error in main loop: {e}")
            send_discord_alert(f"❌ Fatal error: {e}", urgent=True)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        log("🚀 Starting Enhanced AI Trading Bot with ALL MAXIMUM PROFITABILITY FEATURES...")
        log("📋 Feature Checklist:")
        log("✅ Dual-Horizon Prediction (Short & Medium-term models must agree)")
        log("✅ VotingClassifier Ensemble (XGBoost + Random Forest + Logistic Regression)")
        log("✅ VWAP & Volume Spike Filtering (>1.2x volume, price > VWAP)")
        log("✅ Support/Resistance Level Confirmation")
        log("✅ Sentiment Override (FinBERT + VADER ensemble)")
        log("✅ Meta Model Approval System with Auto-Retraining")
        log("✅ Dynamic Watchlist Optimization (20 tickers, max 12 per sector)")
        log("✅ Cooldown Management (5-minute cooldowns)")
        log("✅ Kelly Criterion Position Sizing (confidence-adjusted)")
        log("✅ ATR-based Dynamic Stops & Profit Targets")
        log("✅ Profit Decay Exit Logic & Trailing Stops")
        log("✅ Sector Diversification (max 30% per sector)")
        log("✅ Volume Spike Confirmation (>1.2x recent volume)")
        log("✅ Live Accuracy Tracking & Model Performance")
        log("✅ Q-Learning with Reward Shaping")
        log("✅ Market Regime Detection (Bull/Bear/Neutral)")
        log("✅ End-of-Day Liquidation & Reporting")
        log("✅ Google Sheets Logging & Discord Alerts")
        log("✅ P&L Logging & Trade Outcome Tracking")
        log("✅ Live Training Loop (every 5 minutes)")
        log("✅ TimeSeriesSplit Cross-Validation")
        log("✅ Comprehensive Risk Management")
        
        bot = EnhancedTradingBot()
        bot.run_main_loop()
    except Exception as e:
        log(f"❌ Fatal startup error: {e}")
        send_discord_alert(f"❌ Fatal startup error: {e}", urgent=True)
