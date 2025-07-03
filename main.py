import os
import time
import pytz
import torch
import joblib
import random
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict, deque
import json
import warnings
import sys
import traceback
import threading
from scipy import stats
from scipy.signal import find_peaks
import talib
from flask import Flask, jsonify

warnings.filterwarnings('ignore')

# === ENHANCED CONFIGURATION ===
DEBUG = True
load_dotenv()
pacific = timezone('US/Pacific')

# Create Flask app for health checks (required for Render)
app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'market_open': is_market_open_safe(),
        'bot_running': True,
        'features_active': {
            'q_learning': len(trading_state.q_table) if 'trading_state' in globals() else 0,
            'sector_rotation': len(trading_state.sector_performance) if 'trading_state' in globals() else 0,
            'support_resistance': len(trading_state.support_resistance_cache) if 'trading_state' in globals() else 0,
            'volume_profile': len(trading_state.volume_profile_cache) if 'trading_state' in globals() else 0,
            'fibonacci_levels': len(trading_state.fibonacci_levels) if 'trading_state' in globals() else 0,
            'elliott_waves': len(trading_state.elliott_wave_counts) if 'trading_state' in globals() else 0,
            'harmonic_patterns': len(trading_state.harmonic_patterns) if 'trading_state' in globals() else 0,
            'ichimoku_clouds': len(trading_state.ichimoku_clouds) if 'trading_state' in globals() else 0
        }
    })

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        'service': 'Ultra-Advanced AI Trading Bot',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0.0',
        'features': [
            'Q-Learning Reinforcement Learning',
            'Support/Resistance Analysis', 
            'Volume Profile Analysis',
            'Fibonacci Retracement',
            'Elliott Wave Analysis',
            'Harmonic Pattern Recognition',
            'Ichimoku Cloud Analysis',
            'Market Microstructure Analysis',
            'Advanced Volatility Models',
            'Markov Regime Detection',
            'Sector Rotation System',
            'Advanced Portfolio Management',
            'Meta Model Approval',
            'Sentiment Analysis',
            'Risk Management',
            'Emergency Stops',
            'Advanced Backtesting',
            'Order Flow Analysis',
            'Smart Money Detection',
            'Institutional Activity Tracking'
        ]
    })

# Enhanced model directories
os.makedirs("models/short", exist_ok=True)
os.makedirs("models/medium", exist_ok=True)
os.makedirs("models/meta", exist_ok=True)
os.makedirs("models/q_learning", exist_ok=True)
os.makedirs("models/garch", exist_ok=True)
os.makedirs("models/regime", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("performance", exist_ok=True)
os.makedirs("backtests", exist_ok=True)
os.makedirs("support_resistance", exist_ok=True)
os.makedirs("volume_profiles", exist_ok=True)
os.makedirs("sector_analysis", exist_ok=True)
os.makedirs("fibonacci_analysis", exist_ok=True)
os.makedirs("elliott_waves", exist_ok=True)
os.makedirs("harmonic_patterns", exist_ok=True)
os.makedirs("ichimoku_analysis", exist_ok=True)
os.makedirs("microstructure", exist_ok=True)
os.makedirs("volatility_models", exist_ok=True)
os.makedirs("regime_analysis", exist_ok=True)

# === STRATEGIC THRESHOLDS ===
THRESHOLDS = {
    'SHORT_BUY_THRESHOLD': 0.53,
    'SHORT_SELL_AVOID_THRESHOLD': 0.45,
    'PRICE_MOMENTUM_MIN': 0.005,
    'VOLUME_SPIKE_MIN': 1.2,
    'SENTIMENT_HOLD_OVERRIDE': -0.5,
    'MAX_PER_SECTOR_WATCHLIST': 12,
    'MAX_PER_SECTOR_PORTFOLIO': 0.3,
    'WATCHLIST_LIMIT': 20,
    'ATR_STOP_MULTIPLIER': 1.2,
    'ATR_PROFIT_MULTIPLIER': 2.5,
    'MAX_PORTFOLIO_RISK': 0.02,
    'MAX_DAILY_DRAWDOWN': 0.05,
    'VOLATILITY_GATE_THRESHOLD': 0.05,
    'MIN_MODEL_ACCURACY': 0.55,
    'EMERGENCY_DRAWDOWN_LIMIT': 0.10,
    'SUPPORT_RESISTANCE_STRENGTH': 3,
    'VOLUME_PROFILE_BINS': 20,
    'Q_LEARNING_ALPHA': 0.1,
    'Q_LEARNING_GAMMA': 0.95,
    'Q_LEARNING_EPSILON': 0.1,
    'SECTOR_ROTATION_THRESHOLD': 0.02,
    'SHARPE_RATIO_MIN': 1.0,
    'MAX_CORRELATION_THRESHOLD': 0.7,
    'FIBONACCI_LEVELS': [0.236, 0.382, 0.5, 0.618, 0.786],
    'FIBONACCI_EXTENSIONS': [1.272, 1.414, 1.618, 2.0, 2.618],
    'ICHIMOKU_PERIODS': {'tenkan': 9, 'kijun': 26, 'senkou_b': 52},
    'ELLIOTT_WAVE_MIN_WAVES': 5,
    'HARMONIC_PATTERN_TOLERANCE': 0.05,
    'MARKET_PROFILE_SESSIONS': 4,
    'ORDERFLOW_IMBALANCE_THRESHOLD': 0.3,
    'LIQUIDITY_POOL_MIN_SIZE': 1000000,
    'SMART_MONEY_FLOW_THRESHOLD': 0.6,
    'INSTITUTIONAL_BLOCK_SIZE': 10000,
    'DARK_POOL_INDICATOR_THRESHOLD': 0.4,
    'GARCH_ALPHA': 0.1,
    'GARCH_BETA': 0.85,
    'GARCH_OMEGA': 0.05,
    'REGIME_TRANSITION_THRESHOLD': 0.7,
    'VOLATILITY_CLUSTERING_WINDOW': 20,
    'MARKET_IMPACT_THRESHOLD': 0.001,
    'BLOCK_TRADE_THRESHOLD': 0.95,
    'SMART_MONEY_WINDOW': 20,
    'INSTITUTIONAL_FLOW_WINDOW': 50,
    'DARK_POOL_VOLUME_THRESHOLD': 0.8
}

def log(msg):
    """Enhanced logging with better error handling"""
    try:
        if DEBUG:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] {msg}"
            print(log_message)
            
            # Safe file logging
            try:
                with open("logs/trading_bot.log", "a") as f:
                    f.write(f"{log_message}\n")
            except Exception as file_error:
                print(f"[{timestamp}] Warning: Could not write to log file: {file_error}")
    except Exception as e:
        print(f"Logging error: {e}")

def safe_api_call(func, *args, **kwargs):
    """Wrapper for safe API calls with retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                log(f"API call failed after {max_retries} attempts")
                return None

def is_market_open_safe():
    """Safe market open check with fallback"""
    try:
        if api:
            clock = safe_api_call(api.get_clock)
            if clock:
                return clock.is_open
        
        # Fallback: check market hours manually
        now = datetime.now(pytz.timezone("US/Eastern"))
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if it's a weekday and within market hours
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            return market_open <= now <= market_close
        
        return False
        
    except Exception as e:
        log(f"Market status check failed: {e}")
        return False

# === API SETUP ===
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

api = None
if API_KEY and SECRET_KEY:
    try:
        api = REST(API_KEY, SECRET_KEY, base_url=BASE_URL)
        account = safe_api_call(api.get_account)
        if account:
            log("✅ Alpaca API connected successfully")
        else:
            log("⚠️ Alpaca API connection test failed")
    except Exception as e:
        log(f"❌ Alpaca API setup failed: {e}")
        api = None
else:
    log("⚠️ Missing Alpaca API credentials - running in demo mode")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def send_discord_alert(message, urgent=False):
    """Safe Discord alert with error handling"""
    try:
        if not DISCORD_WEBHOOK_URL:
            log(f"Discord alert (no webhook): {message}")
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

# Initialize other components safely
newsapi = None
if NEWS_API_KEY:
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        log("✅ News API connected")
    except Exception as e:
        log(f"⚠️ News API setup failed: {e}")

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
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "AMD", "NFLX", "CRM", "ORCL", "IBM", "INTC", "QCOM", "AVGO", "TXN", "MU", "ADBE", "SNOW", "SHOP", "PLTR", "RBLX", "ZM", "DOCU", "OKTA", "CRWD"],
    "Finance": ["BRK.B", "V", "MA", "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "TFC", "COF", "SCHW", "BLK", "SPGI", "ICE", "CME", "MCO"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "PSX", "EOG", "MPC", "VLO", "OXY", "HAL", "BKR", "DVN", "FANG", "MRO", "APA", "HES", "KMI", "OKE", "EPD", "ET"],
    "Healthcare": ["PFE", "JNJ", "LLY", "MRK", "ABT", "BMY", "CVS", "UNH", "ABBV", "TMO", "DHR", "MDT", "GILD", "AMGN", "ISRG", "VRTX", "REGN", "ZTS", "DXCM", "ILMN"],
    "Consumer": ["HD", "LOW", "COST", "TGT", "WMT", "PG", "PEP", "KO", "PM", "NKE", "SBUX", "MCD", "CMG", "DIS", "AMZN", "F", "GM", "ROKU", "BYND", "PTON"],
    "Industrial": ["UNP", "CSX", "UPS", "FDX", "CAT", "DE", "GE", "HON", "BA", "LMT", "RTX", "NOC", "MMM", "EMR", "ETN", "PH", "ITW", "CMI", "ROK", "DOV"],
    "REIT": ["PLD", "O", "SPG", "AMT", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "DLR", "WELL", "VTR", "ARE", "MAA", "ESS", "UDR", "CPT", "FRT", "REG"],
    "Communication": ["T", "VZ", "CMCSA", "CHTR", "TMUS", "DISH", "DIS", "GOOGL", "META", "TWTR", "SNAP", "PINS", "MTCH", "IAC", "SIRI", "LBRDK", "LBRDA", "PARA"],
    "Materials": ["LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "AA", "DOW", "DD", "PPG", "RPM", "IFF", "FMC", "LYB", "CF", "MOS", "ALB", "VMC", "MLM"],
    "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "XEL", "PEG", "SRE", "AEP", "PCG", "ED", "ETR", "WEC", "PPL", "CMS", "DTE", "NI", "LNT", "EVRG", "CNP"]
}

FALLBACK_UNIVERSE = [ticker for sector_tickers in SECTOR_UNIVERSE.values() for ticker in sector_tickers]

# === ULTRA-ADVANCED GLOBAL STATE ===
class UltraAdvancedTradingState:
    def __init__(self):
        # Basic state
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
        self.regime_state = "neutral"
        self.open_positions = {}
        self.trade_id_counter = 0
        self.emergency_stop_triggered = False
        
        # Advanced tracking
        self.sector_performance = defaultdict(list)
        self.correlation_matrix = {}
        self.volatility_regime = "normal"
        self.market_microstructure = {}
        self.risk_metrics = {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'treynor_ratio': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'max_consecutive_losses': 0,
            'recovery_factor': 0.0,
            'sterling_ratio': 0.0
        }
        self.portfolio_weights = {}
        self.rebalance_signals = {}
        
        # Q-Learning state
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.q_state_history = deque(maxlen=1000)
        self.q_action_history = deque(maxlen=1000)
        self.q_reward_history = deque(maxlen=1000)
        self.q_learning_stats = {
            'total_episodes': 0,
            'exploration_rate': THRESHOLDS['Q_LEARNING_EPSILON'],
            'learning_rate': THRESHOLDS['Q_LEARNING_ALPHA'],
            'avg_reward': 0.0,
            'best_reward': 0.0,
            'convergence_score': 0.0
        }
        
        # Advanced pattern recognition
        self.fibonacci_levels = {}
        self.elliott_wave_counts = {}
        self.harmonic_patterns = {}
        self.ichimoku_clouds = {}
        self.candlestick_patterns = {}
        self.chart_patterns = {}
        
        # Market microstructure
        self.order_flow_imbalance = {}
        self.liquidity_pools = {}
        self.smart_money_flow = {}
        self.institutional_activity = {}
        self.dark_pool_indicators = {}
        self.market_impact_models = {}
        self.bid_ask_spreads = {}
        self.market_depth = {}
        
        # Advanced volatility models
        self.garch_models = {}
        self.volatility_surface = {}
        self.implied_volatility = {}
        self.volatility_smile = {}
        self.volatility_term_structure = {}
        
        # Regime detection
        self.markov_regime_states = {}
        self.regime_probabilities = {}
        self.regime_transitions = {}
        self.regime_persistence = {}
        
        # News and sentiment
        self.news_sentiment_scores = {}
        self.social_sentiment = {}
        self.earnings_calendar = {}
        self.economic_indicators = {}
        self.event_impact_analysis = {}
        
        # Performance attribution
        self.factor_exposures = {}
        self.alpha_beta_metrics = {}
        self.attribution_analysis = {}
        self.style_analysis = {}
        self.benchmark_tracking = {}
        
        # Advanced portfolio analytics
        self.portfolio_optimization = {}
        self.risk_budgeting = {}
        self.factor_models = {}
        self.stress_testing = {}
        self.scenario_analysis = {}
        
    def reset_daily(self):
        self.sector_allocations = {}
        self.cooldown_timers = {}
        self.sentiment_cache = {}
        self.support_resistance_cache = {}
        self.volume_profile_cache = {}
        self.daily_drawdown = 0.0
        self.starting_equity = 0.0
        self.emergency_stop_triggered = False
        self.rebalance_signals = {}
        self.order_flow_imbalance = {}
        self.smart_money_flow = {}
        self.market_impact_models = {}
    
    def get_next_trade_id(self):
        self.trade_id_counter += 1
        return f"TRADE_{datetime.now().strftime('%Y%m%d')}_{self.trade_id_counter:04d}"
    
    def update_ultra_advanced_risk_metrics(self):
        """Update ultra-advanced risk metrics"""
        try:
            if len(self.trade_outcomes) < 10:
                return
            
            returns = [trade['return'] for trade in self.trade_outcomes[-100:]]
            
            if len(returns) < 2:
                return
            
            # Basic metrics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe Ratio
            if std_return > 0:
                self.risk_metrics['sharpe_ratio'] = mean_return / std_return * np.sqrt(252)
            
            # Sortino Ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    self.risk_metrics['sortino_ratio'] = mean_return / downside_std * np.sqrt(252)
            
            # Win Rate
            wins = [r for r in returns if r > 0]
            self.risk_metrics['win_rate'] = len(wins) / len(returns)
            
            # Profit Factor
            total_wins = sum(wins)
            losses = [abs(r) for r in returns if r < 0]
            total_losses = sum(losses)
            if total_losses > 0:
                self.risk_metrics['profit_factor'] = total_wins / total_losses
            
            # Average win/loss
            if wins:
                self.risk_metrics['avg_win'] = np.mean(wins)
            if losses:
                self.risk_metrics['avg_loss'] = np.mean(losses)
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            self.risk_metrics['max_drawdown'] = abs(np.min(drawdowns))
            
            # Calmar Ratio
            if self.risk_metrics['max_drawdown'] > 0:
                self.risk_metrics['calmar_ratio'] = mean_return * 252 / self.risk_metrics['max_drawdown']
            
            # Value at Risk (95%)
            self.risk_metrics['var_95'] = np.percentile(returns, 5)
            
            # Conditional Value at Risk (95%)
            var_95 = self.risk_metrics['var_95']
            tail_losses = [r for r in returns if r <= var_95]
            if tail_losses:
                self.risk_metrics['cvar_95'] = np.mean(tail_losses)
            
            # Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive = 0
            for r in returns:
                if r < 0:
                    consecutive_losses += 1
                    max_consecutive = max(max_consecutive, consecutive_losses)
                else:
                    consecutive_losses = 0
            self.risk_metrics['max_consecutive_losses'] = max_consecutive
            
            # Recovery Factor
            if self.risk_metrics['max_drawdown'] > 0:
                total_return = (np.prod(1 + np.array(returns)) - 1)
                self.risk_metrics['recovery_factor'] = total_return / self.risk_metrics['max_drawdown']
            
            # Sterling Ratio
            avg_drawdown = np.mean([abs(d) for d in drawdowns if d < 0])
            if avg_drawdown > 0:
                self.risk_metrics['sterling_ratio'] = mean_return * 252 / avg_drawdown
                
        except Exception as e:
            log(f"❌ Ultra-advanced risk metrics update failed: {e}")

trading_state = UltraAdvancedTradingState()

# === ULTRA-ADVANCED TECHNICAL ANALYSIS ===
def add_ultra_advanced_technical_indicators(df):
    """Add comprehensive technical indicators including ultra-advanced ones"""
    try:
        if df is None or df.empty:
            return df
        
        # Convert to numpy arrays for TA-Lib
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        open_price = df['open'].values
        
        # Basic indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages (multiple timeframes)
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
        
        # RSI (multiple timeframes)
        for period in [2, 14, 21]:
            if len(df) >= period:
                df[f'rsi_{period}'] = RSIIndicator(close=df['close'], window=period).rsi()
        
        # MACD (multiple settings)
        macd_12_26 = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd_12_26.macd()
        df['macd_signal'] = macd_12_26.macd_signal()
        df['macd_histogram'] = macd_12_26.macd_diff()
        
        macd_5_35 = MACD(close=df['close'], window_fast=5, window_slow=35, window_sign=5)
        df['macd_fast'] = macd_5_35.macd()
        df['macd_fast_signal'] = macd_5_35.macd_signal()
        
        # Stochastic Oscillator (multiple settings)
        if len(df) >= 14:
            stoch_14 = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch_14.stoch()
            df['stoch_d'] = stoch_14.stoch_signal()
            
            stoch_5 = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3)
            df['stoch_k_fast'] = stoch_5.stoch()
            df['stoch_d_fast'] = stoch_5.stoch_signal()
        
        # Williams %R (multiple timeframes)
        for period in [14, 21]:
            if len(df) >= period:
                df[f'williams_r_{period}'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=period).williams_r()
        
        # ADX (Average Directional Index)
        if len(df) >= 14:
            adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx_indicator.adx()
            df['adx_pos'] = adx_indicator.adx_pos()
            df['adx_neg'] = adx_indicator.adx_neg()
        
        # Money Flow Index
        if len(df) >= 14:
            df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
        
        # OBV and volume indicators
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # ATR (multiple timeframes)
        for period in [14, 21]:
            if len(df) >= period:
                df[f'atr_{period}'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()
        
        # Bollinger Bands (multiple settings)
        if len(df) >= 20:
            bb_20 = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb_20.bollinger_hband()
            df['bb_lower'] = bb_20.bollinger_lband()
            df['bb_middle'] = bb_20.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            bb_10 = BollingerBands(close=df['close'], window=10, window_dev=1.5)
            df['bb_upper_fast'] = bb_10.bollinger_hband()
            df['bb_lower_fast'] = bb_10.bollinger_lband()
        
        # Keltner Channels
        if len(df) >= 20:
            kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=10)
            df['kc_upper'] = kc.keltner_channel_hband()
            df['kc_lower'] = kc.keltner_channel_lband()
            df['kc_middle'] = kc.keltner_channel_mband()
        
        # ULTRA-ADVANCED INDICATORS USING TA-LIB
        try:
            # Parabolic SAR
            df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # Commodity Channel Index
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            df['cci_20'] = talib.CCI(high, low, close, timeperiod=20)
            
            # Rate of Change (multiple timeframes)
            df['roc_10'] = talib.ROC(close, timeperiod=10)
            df['roc_20'] = talib.ROC(close, timeperiod=20)
            
            # Triple Exponential Moving Average
            if len(df) >= 30:
                df['tema'] = talib.TEMA(close, timeperiod=30)
            
            # Kaufman Adaptive Moving Average
            if len(df) >= 30:
                df['kama'] = talib.KAMA(close, timeperiod=30)
            
            # Mesa Adaptive Moving Average
            df['mama'], df['fama'] = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
            
            # Hilbert Transform indicators
            df['ht_trendmode'] = talib.HT_TRENDMODE(close)
            df['ht_dcperiod'] = talib.HT_DCPERIOD(close)
            df['ht_dcphase'] = talib.HT_DCPHASE(close)
            df['ht_phasor_inphase'], df['ht_phasor_quadrature'] = talib.HT_PHASOR(close)
            df['ht_sine'], df['ht_leadsine'] = talib.HT_SINE(close)
            df['ht_trendline'] = talib.HT_TRENDLINE(close)
            
            # Aroon indicators
            df['aroon_up'], df['aroon_down'] = talib.AROON(high, low, timeperiod=14)
            df['aroon_osc'] = talib.AROONOSC(high, low, timeperiod=14)
            
            # Balance of Power
            df['bop'] = talib.BOP(open_price, high, low, close)
            
            # Chande Momentum Oscillator
            df['cmo'] = talib.CMO(close, timeperiod=14)
            
            # Directional Movement Index
            df['dx'] = talib.DX(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            
            # Momentum
            df['mom'] = talib.MOM(close, timeperiod=10)
            
            # Percentage Price Oscillator
            df['ppo'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
            
            # Ultimate Oscillator
            df['ultosc'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            
            # Normalized Average True Range
            df['natr'] = talib.NATR(high, low, close, timeperiod=14)
            
            # True Range
            df['trange'] = talib.TRANGE(high, low, close)
            
            # Average Directional Movement Index Rating
            df['adxr'] = talib.ADXR(high, low, close, timeperiod=14)
            
            # Absolute Price Oscillator
            df['apo'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
            
            # Chaikin A/D Line
            df['ad'] = talib.AD(high, low, close, volume)
            
            # Chaikin A/D Oscillator
            df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
            
            # On Balance Volume
            df['obv_talib'] = talib.OBV(close, volume)
            
            # CANDLESTICK PATTERNS
            df['cdl_doji'] = talib.CDLDOJI(open_price, high, low, close)
            df['cdl_hammer'] = talib.CDLHAMMER(open_price, high, low, close)
            df['cdl_hangingman'] = talib.CDLHANGINGMAN(open_price, high, low, close)
            df['cdl_shootingstar'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
            df['cdl_engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
            df['cdl_harami'] = talib.CDLHARAMI(open_price, high, low, close)
            df['cdl_piercing'] = talib.CDLPIERCING(open_price, high, low, close)
            df['cdl_darkcloud'] = talib.CDLDARKCLOUDCOVER(open_price, high, low, close)
            df['cdl_morningstar'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
            df['cdl_eveningstar'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
            df['cdl_3whitesoldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)
            df['cdl_3blackcrows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)
            df['cdl_spinningtop'] = talib.CDLSPINNINGTOP(open_price, high, low, close)
            df['cdl_marubozu'] = talib.CDLMARUBOZU(open_price, high, low, close)
            
        except Exception as talib_error:
            log(f"⚠️ TA-Lib indicators failed: {talib_error}")
        
        # VWAP and price vs VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
        
        # Multiple timeframe VWAP
        if len(df) >= 20:
            df['vwap_20'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['vwap_50'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(50).sum() / df['volume'].rolling(50).sum()
        
        # Volatility indicators (multiple timeframes)
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Price momentum (multiple timeframes)
        for period in [1, 3, 5, 10, 20]:
            if len(df) >= period:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        df['price_momentum'] = df['momentum_5']  # Primary momentum indicator
        
        # Support/Resistance levels (multiple timeframes)
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'resistance_{period}'] = df['high'].rolling(period).max()
                df[f'support_{period}'] = df['low'].rolling(period).min()
                df[f'support_resistance_ratio_{period}'] = (df['close'] - df[f'support_{period}']) / (df[f'resistance_{period}'] - df[f'support_{period}'])
        
        # ADVANCED PATTERN RECOGNITION
        
        # Pivot Points (multiple timeframes)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        df['r3'] = df['high'] + 2 * (df['pivot'] - df['low'])
        df['s3'] = df['low'] - 2 * (df['high'] - df['pivot'])
        
        # Price channels (multiple timeframes)
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'upper_channel_{period}'] = df['high'].rolling(period).max()
                df[f'lower_channel_{period}'] = df['low'].rolling(period).min()
                df[f'channel_position_{period}'] = (df['close'] - df[f'lower_channel_{period}']) / (df[f'upper_channel_{period}'] - df[f'lower_channel_{period}'])
        
        # Trend strength (multiple methods)
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'trend_strength_{period}'] = abs(df['close'].rolling(period).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 2 else 0))
        
        # Ichimoku Cloud components
        if len(df) >= 52:
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(9).max()
            tenkan_low = df['low'].rolling(9).min()
            df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(26).max()
            kijun_low = df['low'].rolling(26).min()
            df['kijun_sen'] = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B)
            senkou_high = df['high'].rolling(52).max()
            senkou_low = df['low'].rolling(52).min()
            df['senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span)
            df['chikou_span'] = df['close'].shift(-26)
            
            # Cloud thickness and position
            df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
            df['price_vs_cloud'] = np.where(
                df['close'] > np.maximum(df['senkou_span_a'], df['senkou_span_b']), 1,
                np.where(df['close'] < np.minimum(df['senkou_span_a'], df['senkou_span_b']), -1, 0)
            )
        
        # Market Profile Value Area (enhanced)
        if len(df) >= 20:
            for period in [20, 50]:
                if len(df) >= period:
                    df[f'value_area_high_{period}'] = df['high'].rolling(period).quantile(0.7)
                    df[f'value_area_low_{period}'] = df['low'].rolling(period).quantile(0.3)
                    df[f'poc_estimate_{period}'] = df['close'].rolling(period).median()
        
        # Order Flow Indicators (enhanced)
        df['buying_pressure'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['selling_pressure'] = np.where(df['close'] < df['open'], df['volume'], 0)
        df['neutral_pressure'] = np.where(df['close'] == df['open'], df['volume'], 0)
        df['net_buying_pressure'] = df['buying_pressure'] - df['selling_pressure']
        df['buying_pressure_ratio'] = df['buying_pressure'] / (df['buying_pressure'] + df['selling_pressure'] + 1)
        
        # Enhanced buying/selling pressure with multiple timeframes
        for period in [5, 10, 20]:
            if len(df) >= period:
                df[f'net_buying_pressure_{period}'] = df['net_buying_pressure'].rolling(period).sum()
                df[f'buying_ratio_{period}'] = df['buying_pressure'].rolling(period).sum() / (df['buying_pressure'].rolling(period).sum() + df['selling_pressure'].rolling(period).sum() + 1)
        
        # Liquidity indicators (enhanced)
        df['bid_ask_spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['market_impact'] = df['volume'] * df['bid_ask_spread_proxy']
        df['liquidity_ratio'] = df['volume'] / df['market_impact']
        
        # Smart Money indicators (enhanced)
        df['large_trade_indicator'] = np.where(df['volume'] > df['volume'].rolling(20).quantile(0.9), 1, 0)
        df['institutional_flow'] = df['large_trade_indicator'] * df['net_buying_pressure']
        df['smart_money_index'] = df['institutional_flow'].rolling(10).sum()
        
        # Dark pool indicators
        df['dark_pool_proxy'] = np.where(
            (df['volume'] > df['volume'].rolling(20).mean()) & 
            (abs(df['returns']) < df['returns'].rolling(20).std()), 
            df['volume'], 0
        )
        df['dark_pool_ratio'] = df['dark_pool_proxy'].rolling(20).sum() / df['volume'].rolling(20).sum()
        
        # Advanced volatility measures
        df['realized_volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['garman_klass_volatility'] = np.sqrt(
            np.log(df['high'] / df['low']) * np.log(df['high'] / df['close']) +
            np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        ).rolling(20).mean() * np.sqrt(252)
        
        # Parkinson volatility estimator
        df['parkinson_volatility'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
        ).rolling(20).mean() * np.sqrt(252)
        
        # Rogers-Satchell volatility estimator
        df['rogers_satchell_volatility'] = np.sqrt(
            np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
            np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        ).rolling(20).mean() * np.sqrt(252)
        
        return df
        
    except Exception as e:
        log(f"❌ Ultra-advanced technical indicator calculation failed: {e}")
        return df

# === FIBONACCI RETRACEMENT ANALYSIS ===
class FibonacciAnalyzer:
    def __init__(self):
        self.levels = THRESHOLDS['FIBONACCI_LEVELS']
        self.extensions = THRESHOLDS['FIBONACCI_EXTENSIONS']
        
    def calculate_fibonacci_levels(self, df, ticker):
        """Calculate comprehensive Fibonacci retracement and extension levels"""
        try:
            if df is None or df.empty or len(df) < 50:
                return {}
            
            # Find significant swing high and low over different timeframes
            swing_analysis = {}
            
            for period in [20, 50, 100]:
                if len(df) >= period:
                    period_data = df.tail(period)
                    swing_high = period_data['high'].max()
                    swing_low = period_data['low'].min()
                    swing_high_idx = period_data['high'].idxmax()
                    swing_low_idx = period_data['low'].idxmin()
                    
                    # Determine trend direction
                    if swing_high_idx > swing_low_idx:
                        trend = 'uptrend'
                        base_price = swing_low
                        target_price = swing_high
                    else:
                        trend = 'downtrend'
                        base_price = swing_high
                        target_price = swing_low
                    
                    diff = abs(target_price - base_price)
                    
                    # Calculate retracement levels
                    retracement_levels = {}
                    for level in self.levels:
                        if trend == 'uptrend':
                            retracement_levels[f'fib_{level}'] = target_price - (diff * level)
                        else:
                            retracement_levels[f'fib_{level}'] = target_price + (diff * level)
                    
                    # Calculate extension levels
                    extension_levels = {}
                    for ext in self.extensions:
                        if trend == 'uptrend':
                            extension_levels[f'fib_ext_{ext}'] = target_price + (diff * (ext - 1))
                        else:
                            extension_levels[f'fib_ext_{ext}'] = target_price - (diff * (ext - 1))
                    
                    swing_analysis[f'period_{period}'] = {
                        'swing_high': swing_high,
                        'swing_low': swing_low,
                        'trend': trend,
                        'retracement_levels': retracement_levels,
                        'extension_levels': extension_levels,
                        'price_range': diff
                    }
            
            # Current price analysis
            current_price = df['close'].iloc[-1]
            
            # Find nearest levels across all timeframes
            all_levels = []
            for period_data in swing_analysis.values():
                all_levels.extend(period_data['retracement_levels'].values())
                all_levels.extend(period_data['extension_levels'].values())
            
            if all_levels:
                nearest_level = min(all_levels, key=lambda x: abs(x - current_price))
                distance_to_nearest = abs(current_price - nearest_level) / current_price
            else:
                nearest_level = current_price
                distance_to_nearest = 0
            
            result = {
                'swing_analysis': swing_analysis,
                'current_price': current_price,
                'nearest_level': nearest_level,
                'distance_to_nearest': distance_to_nearest,
                'fibonacci_confluence': self.find_fibonacci_confluence(swing_analysis, current_price)
            }
            
            trading_state.fibonacci_levels[ticker] = result
            
            # Save to file
            filename = f"fibonacci_analysis/{ticker}_fibonacci.json"
            with open(filename, 'w') as f:
                json.dump(result, f, default=str)
            
            return result
            
        except Exception as e:
            log(f"❌ Fibonacci analysis failed for {ticker}: {e}")
            return {}
    
    def find_fibonacci_confluence(self, swing_analysis, current_price):
        """Find confluence zones where multiple Fibonacci levels cluster"""
        try:
            all_levels = []
            for period_data in swing_analysis.values():
                all_levels.extend(period_data['retracement_levels'].values())
                all_levels.extend(period_data['extension_levels'].values())
            
            if not all_levels:
                return []
            
            # Group levels that are close together (within 1% of each other)
            confluence_zones = []
            tolerance = 0.01
            
            sorted_levels = sorted(all_levels)
            current_zone = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if abs(level - current_zone[-1]) / current_zone[-1] <= tolerance:
                    current_zone.append(level)
                else:
                    if len(current_zone) >= 2:  # At least 2 levels for confluence
                        confluence_zones.append({
                            'price': np.mean(current_zone),
                            'strength': len(current_zone),
                            'levels': current_zone
                        })
                    current_zone = [level]
            
            # Check the last zone
            if len(current_zone) >= 2:
                confluence_zones.append({
                    'price': np.mean(current_zone),
                    'strength': len(current_zone),
                    'levels': current_zone
                })
            
            # Sort by strength (number of confluent levels)
            confluence_zones.sort(key=lambda x: x['strength'], reverse=True)
            
            return confluence_zones[:5]  # Return top 5 confluence zones
            
        except Exception as e:
            log(f"❌ Fibonacci confluence analysis failed: {e}")
            return []

fibonacci_analyzer = FibonacciAnalyzer()

# === ELLIOTT WAVE ANALYSIS ===
class ElliottWaveAnalyzer:
    def __init__(self):
        self.min_waves = THRESHOLDS['ELLIOTT_WAVE_MIN_WAVES']
        
    def detect_elliott_waves(self, df, ticker):
        """Detect Elliott Wave patterns with enhanced analysis"""
        try:
            if df is None or df.empty or len(df) < 100:
                return {}
            
            # Find significant peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            # Use different prominence thresholds for different wave degrees
            wave_analysis = {}
            
            for degree, (prominence_factor, min_distance) in [
                ('primary', (0.5, 10)),
                ('intermediate', (0.3, 5)),
                ('minor', (0.2, 3))
            ]:
                # Detect peaks (potential wave tops)
                peaks, peak_properties = find_peaks(
                    highs, 
                    distance=min_distance, 
                    prominence=np.std(highs) * prominence_factor
                )
                
                # Detect troughs (potential wave bottoms)
                troughs, trough_properties = find_peaks(
                    -lows, 
                    distance=min_distance, 
                    prominence=np.std(lows) * prominence_factor
                )
                
                # Combine and sort by time
                all_points = []
                for peak in peaks:
                    all_points.append({
                        'index': peak,
                        'price': highs[peak],
                        'type': 'peak',
                        'prominence': peak_properties['prominences'][np.where(peaks == peak)[0][0]]
                    })
                
                for trough in troughs:
                    all_points.append({
                        'index': trough,
                        'price': lows[trough],
                        'type': 'trough',
                        'prominence': trough_properties['prominences'][np.where(troughs == trough)[0][0]]
                    })
                
                all_points.sort(key=lambda x: x['index'])
                
                # Elliott Wave counting and validation
                waves = self.count_elliott_waves(all_points)
                wave_validation = self.validate_elliott_waves(waves)
                
                wave_analysis[degree] = {
                    'turning_points': all_points,
                    'waves': waves,
                    'validation': wave_validation,
                    'wave_count': len(waves),
                    'current_wave': self.identify_current_wave(waves),
                    'next_target': self.calculate_wave_targets(waves)
                }
            
            # Determine the most likely wave count
            best_wave_count = self.select_best_wave_count(wave_analysis)
            
            result = {
                'wave_analysis': wave_analysis,
                'best_wave_count': best_wave_count,
                'elliott_wave_signals': self.generate_elliott_signals(best_wave_count),
                'wave_relationships': self.analyze_wave_relationships(best_wave_count)
            }
            
            trading_state.elliott_wave_counts[ticker] = result
            
            # Save to file
            filename = f"elliott_waves/{ticker}_elliott_waves.json"
            with open(filename, 'w') as f:
                json.dump(result, f, default=str)
            
            return result
            
        except Exception as e:
            log(f"❌ Elliott Wave analysis failed for {ticker}: {e}")
            return {}
    
    def count_elliott_waves(self, turning_points):
        """Count Elliott Waves from turning points"""
        try:
            if len(turning_points) < self.min_waves:
                return []
            
            waves = []
            for i in range(min(len(turning_points), 13)):  # Limit to 13 waves (8 + 5)
                if i < len(turning_points):
                    wave_num = (i % 8) + 1 if i < 8 else ((i - 8) % 5) + 1
                    wave_type = 'impulse' if i < 8 else 'corrective'
                    
                    waves.append({
                        'wave_number': wave_num,
                        'wave_type': wave_type,
                        'price': turning_points[i]['price'],
                        'index': turning_points[i]['index'],
                        'point_type': turning_points[i]['type'],
                        'prominence': turning_points[i]['prominence']
                    })
            
            return waves
            
        except Exception as e:
            log(f"❌ Elliott Wave counting failed: {e}")
            return []
    
    def validate_elliott_waves(self, waves):
        """Validate Elliott Wave patterns according to rules"""
        try:
            if len(waves) < 5:
                return {'valid': False, 'violations': ['Insufficient waves']}
            
            violations = []
            
            # Elliott Wave Rules
            # Rule 1: Wave 2 never retraces more than 100% of Wave 1
            if len(waves) >= 3:
                wave_1_range = abs(waves[1]['price'] - waves[0]['price'])
                wave_2_retrace = abs(waves[2]['price'] - waves[1]['price'])
                if wave_2_retrace > wave_1_range:
                    violations.append("Wave 2 retraces more than 100% of Wave 1")
            
            # Rule 2: Wave 3 is never the shortest of waves 1, 3, and 5
            if len(waves) >= 5:
                wave_1_len = abs(waves[1]['price'] - waves[0]['price'])
                wave_3_len = abs(waves[3]['price'] - waves[2]['price'])
                wave_5_len = abs(waves[4]['price'] - waves[3]['price']) if len(waves) > 4 else float('inf')
                
                if wave_3_len < wave_1_len and wave_3_len < wave_5_len:
                    violations.append("Wave 3 is the shortest wave")
            
            # Rule 3: Wave 4 never enters the price territory of Wave 1
            if len(waves) >= 5:
                wave_1_high = max(waves[0]['price'], waves[1]['price'])
                wave_1_low = min(waves[0]['price'], waves[1]['price'])
                wave_4_price = waves[4]['price']
                
                if wave_1_low <= wave_4_price <= wave_1_high:
                    violations.append("Wave 4 overlaps with Wave 1")
            
            return {
                'valid': len(violations) == 0,
                'violations': violations,
                'confidence': max(0, 1 - len(violations) * 0.3)
            }
            
        except Exception as e:
            log(f"❌ Elliott Wave validation failed: {e}")
            return {'valid': False, 'violations': ['Validation error']}
    
    def identify_current_wave(self, waves):
        """Identify the current wave position"""
        try:
            if not waves:
                return None
            
            current_wave_num = len(waves) % 8 if len(waves) <= 8 else ((len(waves) - 8) % 5) + 1
            wave_type = 'impulse' if len(waves) <= 8 else 'corrective'
            
            return {
                'wave_number': current_wave_num,
                'wave_type': wave_type,
                'position': 'developing',
                'expected_direction': self.get_expected_direction(current_wave_num, wave_type)
            }
            
        except Exception as e:
            log(f"❌ Current wave identification failed: {e}")
            return None
    
    def get_expected_direction(self, wave_num, wave_type):
        """Get expected direction for current wave"""
        if wave_type == 'impulse':
            return 'up' if wave_num in [1, 3, 5] else 'down'
        else:  # corrective
            return 'down' if wave_num in [1, 3] else 'up'
    
    def calculate_wave_targets(self, waves):
        """Calculate potential targets for next waves"""
        try:
            if len(waves) < 3:
                return {}
            
            targets = {}
            
            # Fibonacci relationships for wave targets
            if len(waves) >= 2:
                wave_1_range = abs(waves[1]['price'] - waves[0]['price'])
                
                # Wave 3 targets (typically 1.618 or 2.618 times Wave 1)
                if len(waves) >= 3:
                    base_price = waves[2]['price']
                    targets['wave_3_target_1'] = base_price + (wave_1_range * 1.618)
                    targets['wave_3_target_2'] = base_price + (wave_1_range * 2.618)
                
                # Wave 5 targets
                if len(waves) >= 5:
                    wave_3_range = abs(waves[3]['price'] - waves[2]['price'])
                    targets['wave_5_target'] = waves[4]['price'] + (wave_3_range * 0.618)
            
            # Corrective wave targets (ABC)
            if len(waves) >= 5 and waves[5]['wave_type'] == 'corrective':
                wave_a_range = abs(waves[6]['price'] - waves[5]['price'])
                targets['wave_c_target'] = waves[7]['price'] - (wave_a_range * 1.618)
            
            return targets
            
        except Exception as e:
            log(f"❌ Wave target calculation failed: {e}")
            return {}
    
    def select_best_wave_count(self, wave_analysis):
        """Select the best wave count based on validation and prominence"""
        try:
            best_count = None
            max_confidence = -1
            
            for degree, analysis in wave_analysis.items():
                if analysis['validation']['valid']:
                    # Prioritize counts with higher prominence
                    prominence_score = sum([w['prominence'] for w in analysis['waves']])
                    confidence = analysis['validation']['confidence'] + (prominence_score * 0.01)
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_count = analysis
            
            return best_count
            
        except Exception as e:
            log(f"❌ Best wave count selection failed: {e}")
            return None
    
    def generate_elliott_signals(self, best_wave_count):
        """Generate trading signals based on Elliott Wave analysis"""
        try:
            if not best_wave_count:
                return {}
            
            signals = {}
            current_wave = best_wave_count['current_wave']
            
            if current_wave['wave_type'] == 'impulse':
                if current_wave['wave_number'] == 3:
                    signals['action'] = 'buy'
                    signals['strength'] = 'high'
                    signals['reason'] = 'Wave 3 developing'
                elif current_wave['wave_number'] == 5:
                    signals['action'] = 'sell'
                    signals['strength'] = 'medium'
                    signals['reason'] = 'Wave 5 nearing completion'
            elif current_wave['wave_type'] == 'corrective':
                if current_wave['wave_number'] == 2:
                    signals['action'] = 'sell'
                    signals['strength'] = 'medium'
                    signals['reason'] = 'Corrective Wave 2 developing'
                elif current_wave['wave_number'] == 4:
                    signals['action'] = 'buy'
                    signals['strength'] = 'low'
                    signals['reason'] = 'Corrective Wave 4 developing'
            
            return signals
            
        except Exception as e:
            log(f"❌ Elliott signal generation failed: {e}")
            return {}
    
    def analyze_wave_relationships(self, best_wave_count):
        """Analyze relationships between waves for confirmation"""
        try:
            if not best_wave_count or len(best_wave_count['waves']) < 5:
                return {}
            
            wave_1_len = abs(best_wave_count['waves'][1]['price'] - best_wave_count['waves'][0]['price'])
            
            wave_3_len = abs(best_wave_count['waves'][3]['price'] - best_wave_count['waves'][2]['price'])
            wave_5_len = abs(best_wave_count['waves'][4]['price'] - best_wave_count['waves'][3]['price'])
            
            relationships = {}
            
            # Wave 3 is often 1.618 times wave 1
            relationships['wave_3_vs_1'] = wave_3_len / wave_1_len
            
            # Wave 5 is often equal to wave 1
            relationships['wave_5_vs_1'] = wave_5_len / wave_1_len
            
            return relationships
            
        except Exception as e:
            log(f"❌ Wave relationship analysis failed: {e}")
            return {}

elliott_wave_analyzer = ElliottWaveAnalyzer()

# === HARMONIC PATTERN RECOGNITION ===
class HarmonicPatternAnalyzer:
    def __init__(self):
        self.tolerance = THRESHOLDS['HARMONIC_PATTERN_TOLERANCE']
        
    def detect_harmonic_patterns(self, df, ticker):
        """Detect harmonic patterns like Gartley, Butterfly, etc."""
        try:
            if df is None or df.empty or len(df) < 100:
                return {}
            
            # Find significant peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            
            patterns = []
            
            # Look for common harmonic patterns
            patterns.extend(self.detect_gartley(df, peaks, troughs))
            patterns.extend(self.detect_butterfly(df, peaks, troughs))
            patterns.extend(self.detect_crab(df, peaks, troughs))
            patterns.extend(self.detect_bat(df, peaks, troughs))
            
            result = {
                'patterns': patterns,
                'pattern_count': len(patterns),
                'strongest_pattern': max(patterns, key=lambda x: x['confidence']) if patterns else None
            }
            
            trading_state.harmonic_patterns[ticker] = result
            
            # Save to file
            filename = f"harmonic_patterns/{ticker}_harmonic_patterns.json"
            with open(filename, 'w') as f:
                json.dump(result, f, default=str)
            
            return result
            
        except Exception as e:
            log(f"❌ Harmonic pattern analysis failed for {ticker}: {e}")
            return {}
    
    def detect_gartley(self, df, peaks, troughs):
        """Detect Gartley pattern"""
        try:
            patterns = []
            
            # Need at least 5 points (XABCD)
            if len(peaks) + len(troughs) < 5:
                return patterns
            
            # Combine peaks and troughs and sort by index
            all_points = []
            for peak in peaks[-3:]:
                all_points.append({'index': peak, 'price': df['high'][peak], 'type': 'peak'})
            for trough in troughs[-2:]:
                all_points.append({'index': trough, 'price': df['low'][trough], 'type': 'trough'})
            
            all_points.sort(key=lambda x: x['index'])
            
            if len(all_points) < 5:
                return patterns
            
            X, A, B, C, D = all_points[-5:]
            
            # Gartley ratios
            AB = abs(B['price'] - A['price'])
            XA = abs(A['price'] - X['price'])
            BC = abs(C['price'] - B['price'])
            AB_XA_ratio = AB / XA if XA > 0 else 0
            BC_AB_ratio = BC / AB if AB > 0 else 0
            
            # Check ratios
            if (0.5 < AB_XA_ratio < 0.62 and 0.382 < BC_AB_ratio < 0.886):
                patterns.append({
                    'pattern': 'Gartley',
                    'points': [X, A, B, C, D],
                    'ratios': {'AB/XA': AB_XA_ratio, 'BC/AB': BC_AB_ratio},
                    'confidence': 1 - abs(AB_XA_ratio - 0.618) - abs(BC_AB_ratio - 0.618)
                })
            
            return patterns
            
        except Exception as e:
            log(f"❌ Gartley pattern detection failed: {e}")
            return []
    
    def detect_butterfly(self, df, peaks, troughs):
        """Detect Butterfly pattern"""
        try:
            patterns = []
            
            # Need at least 5 points (XABCD)
            if len(peaks) + len(troughs) < 5:
                return patterns
            
            # Combine peaks and troughs and sort by index
            all_points = []
            for peak in peaks[-3:]:
                all_points.append({'index': peak, 'price': df['high'][peak], 'type': 'peak'})
            for trough in troughs[-2:]:
                all_points.append({'index': trough, 'price': df['low'][trough], 'type': 'trough'})
            
            all_points.sort(key=lambda x: x['index'])
            
            if len(all_points) < 5:
                return patterns
            
            X, A, B, C, D = all_points[-5:]
            
            # Butterfly ratios
            AB = abs(B['price'] - A['price'])
            XA = abs(A['price'] - X['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D['price'] - C['price'])
            AB_XA_ratio = AB / XA if XA > 0 else 0
            BC_AB_ratio = BC / AB if AB > 0 else 0
            CD_BC_ratio = CD / BC if BC > 0 else 0
            
            # Check ratios
            if (0.786 < AB_XA_ratio < 0.886 and 0.382 < BC_AB_ratio < 0.886 and CD_BC_ratio > 1.272):
                patterns.append({
                    'pattern': 'Butterfly',
                    'points': [X, A, B, C, D],
                    'ratios': {'AB/XA': AB_XA_ratio, 'BC/AB': BC_AB_ratio, 'CD/BC': CD_BC_ratio},
                    'confidence': 1 - abs(AB_XA_ratio - 0.786) - abs(BC_AB_ratio - 0.618) - abs(CD_BC_ratio - 1.272)
                })
            
            return patterns
            
        except Exception as e:
            log(f"❌ Butterfly pattern detection failed: {e}")
            return []
    
    def detect_crab(self, df, peaks, troughs):
        """Detect Crab pattern"""
        try:
            patterns = []
            
            # Need at least 5 points (XABCD)
            if len(peaks) + len(troughs) < 5:
                return patterns
            
            # Combine peaks and troughs and sort by index
            all_points = []
            for peak in peaks[-3:]:
                all_points.append({'index': peak, 'price': df['high'][peak], 'type': 'peak'})
            for trough in troughs[-2:]:
                all_points.append({'index': trough, 'price': df['low'][trough], 'type': 'trough'})
            
            all_points.sort(key=lambda x: x['index'])
            
            if len(all_points) < 5:
                return patterns
            
            X, A, B, C, D = all_points[-5:]
            
            # Crab ratios
            AB = abs(B['price'] - A['price'])
            XA = abs(A['price'] - X['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D['price'] - C['price'])
            AB_XA_ratio = AB / XA if XA > 0 else 0
            BC_AB_ratio = BC / AB if AB > 0 else 0
            CD_BC_ratio = CD / BC if BC > 0 else 0
            
            # Check ratios
            if (0.382 < AB_XA_ratio < 0.618 and 0.382 < BC_AB_ratio < 0.886 and CD_BC_ratio > 2.618):
                patterns.append({
                    'pattern': 'Crab',
                    'points': [X, A, B, C, D],
                    'ratios': {'AB/XA': AB_XA_ratio, 'BC/AB': BC_AB_ratio, 'CD/BC': CD_BC_ratio},
                    'confidence': 1 - abs(AB_XA_ratio - 0.5) - abs(BC_AB_ratio - 0.618) - abs(CD_BC_ratio - 2.618)
                })
            
            return patterns
            
        except Exception as e:
            log(f"❌ Crab pattern detection failed: {e}")
            return []
    
    def detect_bat(self, df, peaks, troughs):
        """Detect Bat pattern"""
        try:
            patterns = []
            
            # Need at least 5 points (XABCD)
            if len(peaks) + len(troughs) < 5:
                return patterns
            
            # Combine peaks and troughs and sort by index
            all_points = []
            for peak in peaks[-3:]:
                all_points.append({'index': peak, 'price': df['high'][peak], 'type': 'peak'})
            for trough in troughs[-2:]:
                all_points.append({'index': trough, 'price': df['low'][trough], 'type': 'trough'})
            
            all_points.sort(key=lambda x: x['index'])
            
            if len(all_points) < 5:
                return patterns
            
            X, A, B, C, D = all_points[-5:]
            
            # Bat ratios
            AB = abs(B['price'] - A['price'])
            XA = abs(A['price'] - X['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D['price'] - C['price'])
            AB_XA_ratio = AB / XA if XA > 0 else 0
            BC_AB_ratio = BC / AB if AB > 0 else 0
            CD_XA_ratio = CD / XA if XA > 0 else 0
            
            # Check ratios
            if (0.382 < AB_XA_ratio < 0.5 and 0.382 < BC_AB_ratio < 0.886 and 0.886 < CD_XA_ratio < 1.13):
                patterns.append({
                    'pattern': 'Bat',
                    'points': [X, A, B, C, D],
                    'ratios': {'AB/XA': AB_XA_ratio, 'BC/AB': BC_AB_ratio, 'CD/XA': CD_XA_ratio},
                    'confidence': 1 - abs(AB_XA_ratio - 0.4) - abs(BC_AB_ratio - 0.618) - abs(CD_XA_ratio - 0.886)
                })
            
            return patterns
            
        except Exception as e:
            log(f"❌ Bat pattern detection failed: {e}")
            return []

harmonic_pattern_analyzer = HarmonicPatternAnalyzer()

# === ICHIMOKU CLOUD ANALYSIS ===
class IchimokuCloudAnalyzer:
    def __init__(self):
        self.periods = THRESHOLDS['ICHIMOKU_PERIODS']
        
    def calculate_ichimoku_cloud(self, df, ticker):
        """Calculate Ichimoku Cloud components"""
        try:
            if df is None or df.empty or len(df) < max(self.periods.values()):
                return {}
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(self.periods['tenkan']).max()
            tenkan_low = df['low'].rolling(self.periods['tenkan']).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(self.periods['kijun']).max()
            kijun_low = df['low'].rolling(self.periods['kijun']).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.periods['kijun'])
            
            # Senkou Span B (Leading Span B)
            senkou_high = df['high'].rolling(self.periods['senkou_b']).max()
            senkou_low = df['low'].rolling(self.periods['senkou_b']).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.periods['kijun'])
            
            # Chikou Span (Lagging Span)
            chikou_span = df['close'].shift(-self.periods['kijun'])
            
            # Cloud status
            current_price = df['close'].iloc[-1]
            cloud_top = max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1])
            cloud_bottom = min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1])
            
            if current_price > cloud_top:
                cloud_status = "above"
            elif current_price < cloud_bottom:
                cloud_status = "below"
            else:
                cloud_status = "inside"
            
            result = {
                'tenkan_sen': tenkan_sen.iloc[-1],
                'kijun_sen': kijun_sen.iloc[-1],
                'senkou_span_a': senkou_span_a.iloc[-1],
                'senkou_span_b': senkou_span_b.iloc[-1],
                'chikou_span': chikou_span.iloc[-1] if chikou_span.notna().any() else None,
                'cloud_status': cloud_status,
                'cloud_top': cloud_top,
                'cloud_bottom': cloud_bottom
            }
            
            trading_state.ichimoku_clouds[ticker] = result
            
            # Save to file
            filename = f"ichimoku_analysis/{ticker}_ichimoku.json"
            with open(filename, 'w') as f:
                json.dump(result, f, default=str)
            
            return result
            
        except Exception as e:
            log(f"❌ Ichimoku Cloud analysis failed for {ticker}: {e}")
            return {}

ichimoku_cloud_analyzer = IchimokuCloudAnalyzer()

# === SUPPORT AND RESISTANCE ANALYSIS ===
class SupportResistanceAnalyzer:
    def __init__(self):
        self.strength = THRESHOLDS['SUPPORT_RESISTANCE_STRENGTH']
        
    def find_significant_levels(self, df, ticker):
        """Find significant support and resistance levels"""
        try:
            if df is None or df.empty or len(df) < 50:
                return {}
            
            # Find local peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            peaks, _ = find_peaks(highs, distance=10, prominence=np.std(highs) * 0.4)
            troughs, _ = find_peaks(-lows, distance=10, prominence=np.std(lows) * 0.4)
            
            # Filter levels based on strength
            resistance_levels = []
            support_levels = []
            
            for peak in peaks:
                price = highs[peak]
                # Check if it's a significant resistance
                if self.is_significant_level(df, price, 'resistance'):
                    resistance_levels.append(price)
            
            for trough in troughs:
                price = lows[trough]
                # Check if it's a significant support
                if self.is_significant_level(df, price, 'support'):
                    support_levels.append(price)
            
            result = {
                'support_levels': sorted(support_levels),
                'resistance_levels': sorted(resistance_levels),
                'current_price': df['close'].iloc[-1]
            }
            
            trading_state.support_resistance_cache[ticker] = result
            
            # Save to file
            filename = f"support_resistance/{ticker}_levels.json"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            log(f"❌ Support/Resistance analysis failed for {ticker}: {e}")
            return {}
    
    def is_significant_level(self, df, price, level_type):
        """Check if a level is significant based on historical price action"""
        try:
            window = 20
            if level_type == 'resistance':
                # Check if price acted as resistance multiple times
                count = df['high'].rolling(window).apply(lambda x: sum(abs(x - price) < (price * 0.01))).iloc[-1]
                return count >= self.strength
            elif level_type == 'support':
                # Check if price acted as support multiple times
                count = df['low'].rolling(window).apply(lambda x: sum(abs(x - price) < (price * 0.01))).iloc[-1]
                return count >= self.strength
            return False
        except Exception as e:
            log(f"❌ Significance check failed: {e}")
            return False

support_resistance_analyzer = SupportResistanceAnalyzer()

# === VOLUME PROFILE ANALYSIS ===
class VolumeProfileAnalyzer:
    def __init__(self):
        self.bins = THRESHOLDS['VOLUME_PROFILE_BINS']
        
    def calculate_volume_profile(self, df, ticker):
        """Calculate volume profile for a given ticker"""
        try:
            if df is None or df.empty or len(df) < 30:
                return {}
            
            # Define price ranges (bins)
            price_range = df['high'].max() - df['low'].min()
            bin_size = price_range / self.bins
            
            # Calculate volume at each price level
            volume_profile = {}
            for i in range(self.bins):
                low = df['low'].min() + i * bin_size
                high = low + bin_size
                volume = df[(df['close'] >= low) & (df['close'] < high)]['volume'].sum()
                volume_profile[f'{low:.2f}-{high:.2f}'] = volume
            
            # Find Point of Control (POC)
            poc = max(volume_profile, key=volume_profile.get)
            
            # Find Value Area (70% of volume around POC)
            sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            value_area = {}
            total_volume = sum(volume_profile.values())
            value_area_volume = 0
            
            for price_range, volume in sorted_profile:
                value_area[price_range] = volume
                value_area_volume += volume
                if value_area_volume >= 0.7 * total_volume:
                    break
            
            result = {
                'volume_profile': volume_profile,
                'poc': poc,
                'value_area': list(value_area.keys()),
                'total_volume': total_volume
            }
            
            trading_state.volume_profile_cache[ticker] = result
            
            # Save to file
            filename = f"volume_profiles/{ticker}_profile.json"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w') as f:
                json.dump(result, f)
            
            return result
            
        except Exception as e:
            log(f"❌ Volume Profile analysis failed for {ticker}: {e}")
            return {}

volume_profile_analyzer = VolumeProfileAnalyzer()

# === SECTOR ROTATION ANALYSIS ===
class SectorRotationAnalyzer:
    def __init__(self):
        self.threshold = THRESHOLDS['SECTOR_ROTATION_THRESHOLD']
        
    def analyze_sector_performance(self):
        """Analyze sector performance and identify rotation opportunities"""
        try:
            sector_returns = {}
            
            for sector, tickers in SECTOR_UNIVERSE.items():
                # Get returns for each ticker in the sector
                ticker_returns = []
                for ticker in tickers:
                    data = get_enhanced_data(ticker, limit=30)
                    if data is not None and not data.empty:
                        ticker_returns.append(data['returns'].iloc[-1])
                
                # Calculate average sector return
                if ticker_returns:
                    sector_returns[sector] = np.mean(ticker_returns)
                else:
                    sector_returns[sector] = 0
            
            # Identify leading and lagging sectors
            leading_sectors = [sector for sector, ret in sector_returns.items() if ret > self.threshold]
            lagging_sectors = [sector for sector, ret in sector_returns.items() if ret < -self.threshold]
            
            # Store sector performance
            trading_state.sector_performance = sector_returns
            
            # Identify potential rotation opportunities
            rotation_opportunities = []
            for leading_sector in leading_sectors:
                for lagging_sector in lagging_sectors:
                    rotation_opportunities.append((leading_sector, lagging_sector))
            
            # Log sector performance
            log(f"📊 Sector Performance: {sector_returns}")
            log(f"✅ Leading Sectors: {leading_sectors}")
            log(f"⚠️ Lagging Sectors: {lagging_sectors}")
            log(f"🔄 Rotation Opportunities: {rotation_opportunities}")
            
            # Save to file
            filename = "sector_analysis/sector_performance.json"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w') as f:
                json.dump(sector_returns, f)
            
            return sector_returns, leading_sectors, lagging_sectors, rotation_opportunities
            
        except Exception as e:
            log(f"❌ Sector Rotation analysis failed: {e}")
            return {}, [], [], []

sector_rotation_analyzer = SectorRotationAnalyzer()

# === Q-LEARNING REINFORCEMENT LEARNING ===
class QLearningAgent:
    def __init__(self, actions, alpha=THRESHOLDS['Q_LEARNING_ALPHA'], gamma=THRESHOLDS['Q_LEARNING_GAMMA'], epsilon=THRESHOLDS['Q_LEARNING_EPSILON']):
        self.q_table = trading_state.q_table
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_history = trading_state.q_state_history
        self.action_history = trading_state.q_action_history
        self.reward_history = trading_state.q_reward_history
        
    def choose_action(self, state):
        """Choose action based on epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore
            action = random.choice(self.actions)
        else:
            # Exploit
            if state in self.q_table and self.q_table[state]:
                action = max(self.q_table[state], key=self.q_table[state].get)
            else:
                action = random.choice(self.actions)
        return action
    
    def learn(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        try:
            # Get current Q-value
            current_q = self.q_table[state][action]
            
            # Get maximum Q-value for next state
            if next_state in self.q_table and self.q_table[next_state]:
                max_next_q = max(self.q_table[next_state].values())
            else:
                max_next_q = 0
            
            # Calculate new Q-value
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            
            # Update Q-table
            self.q_table[state][action] = new_q
            
            # Store history
            self.state_history.append(state)
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            # Save Q-table
            filename = "models/q_learning/q_table.json"
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, 'w') as f:
                json.dump(self.q_table, f)
            
        except Exception as e:
            log(f"❌ Q-learning update failed: {e}")
    
    def discretize_state(self, df, ticker):
        """Discretize continuous state variables into a finite set of states"""
        try:
            if df is None or df.empty:
                return None
            
            # Extract relevant features
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Discretize features
            price_level = int(current_price / 10)  # Group prices into $10 levels
            rsi_level = int(rsi / 10)  # Group RSI into 10 levels
            macd_level = 1 if macd > 0 else -1 if macd < 0 else 0  # Group MACD into positive, negative, or zero
            volume_level = 1 if volume_ratio > 1 else 0  # Group volume ratio into above or below average
            
            # Combine discretized features into a state
            state = (price_level, rsi_level, macd_level, volume_level)
            
            return state
            
        except Exception as e:
            log(f"❌ State discretization failed: {e}")
            return None

# === MARKET MICROSTRUCTURE ANALYSIS ===
class MarketMicrostructureAnalyzer:
    def __init__(self):
        self.order_flow_threshold = THRESHOLDS['ORDERFLOW_IMBALANCE_THRESHOLD']
        self.liquidity_pool_min_size = THRESHOLDS['LIQUIDITY_POOL_MIN_SIZE']
        self.smart_money_flow_threshold = THRESHOLDS['SMART_MONEY_FLOW_THRESHOLD']
        self.institutional_block_size = THRESHOLDS['INSTITUTIONAL_BLOCK_SIZE']
        self.dark_pool_indicator_threshold = THRESHOLDS['DARK_POOL_INDICATOR_THRESHOLD']
        
    def analyze_order_flow(self, df, ticker):
        """Analyze order flow imbalance"""
        try:
            if df is None or df.empty or len(df) < 30:
                return {}
            
            # Calculate buying and selling pressure
            buying_pressure = np.where(df['close'] > df['open'], df['volume'], 0).sum()
            selling_pressure = np.where(df['close'] < df['open'], df['volume'], 0).sum()
            
            # Calculate order flow imbalance
            net_buying_pressure = buying_pressure - selling_pressure
            total_volume = buying_pressure + selling_pressure
            
            if total_volume > 0:
                order_flow_imbalance = net_buying_pressure / total_volume
            else:
                order_flow_imbalance = 0
            
            # Determine if imbalance is significant
            if abs(order_flow_imbalance) > self.order_flow_threshold:
                if order_flow_imbalance > 0:
                    order_flow_direction = "buying"
                else:
                    order_flow_direction = "selling"
            else:
                order_flow_direction = "neutral"
            
            result = {
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure,
                'order_flow_imbalance': order_flow_imbalance,
                'order_flow_direction': order_flow_direction
            }
            
            trading_state.order_flow_imbalance[ticker] = result
            return result
            
        except Exception as e:
            log(f"❌ Order Flow analysis failed: {e}")
            return {}
    
    def detect_liquidity_pools(self, df, ticker):
        """Detect liquidity pools based on volume and price action"""
        try:
            if df is None or df.empty or len(df) < 50:
                return {}
            
            # Identify high volume areas
            volume_threshold = df['volume'].quantile(0.9)
            high_volume_areas = df[df['volume'] > volume_threshold]
            
            # Filter for significant price consolidation
            consolidation_threshold = df['atr'].mean() * 0.5
            liquidity_pools = []
            
            for index, row in high_volume_areas.iterrows():
                # Check if price is consolidating
                price_range = df['high'].max() - df['low'].min()
                if price_range < consolidation_threshold:
                    # Check if volume is large enough
                    if row['volume'] > self.liquidity_pool_min_size:
                        liquidity_pools.append({
                            'price': row['close'],
                            'volume': row['volume'],
                            'timestamp': index.isoformat()
                        })
            
            result = {
                'liquidity_pools': liquidity_pools,
                'pool_count': len(liquidity_pools)
            }
            
            trading_state.liquidity_pools[ticker] = result
            return result
            
        except Exception as e:
            log(f"❌ Liquidity Pool detection failed: {e}")
            return {}
    
    def analyze_smart_money_flow(self, df, ticker):
        """Analyze smart money flow using volume and price action"""
        try:
            if df is None or df.empty or len(df) < 30:
                return {}
            
            # Calculate buying and selling pressure
            buying_pressure = np.where(df['close'] > df['open'], df['volume'], 0).sum()
            selling_pressure = np.where(df['close'] < df['open'], df['volume'], 0).sum()
            
            # Calculate smart money flow
            net_buying_pressure = buying_pressure - selling_pressure
            total_volume = buying_pressure + selling_pressure
            
            if total_volume > 0:
                smart_money_flow = net_buying_pressure / total_volume
            else:
                smart_money_flow = 0
            
            # Determine if flow is significant
            if abs(smart_money_flow) > self.smart_money_flow_threshold:
                if smart_money_flow > 0:
                    smart_money_direction = "buying"
                else:
                    smart_money_direction = "selling"
            else:
                smart_money_direction = "neutral"
            
            result = {
                'smart_money_flow': smart_money_flow,
                'smart_money_direction': smart_money_direction
            }
            
            trading_state.smart_money_flow[ticker] = result
            return result
            
        except Exception as e:
            log(f"❌ Smart Money Flow analysis failed: {e}")
            return {}
    
    def detect_institutional_activity(self, df, ticker):
        """Detect institutional activity based on large block trades"""
        try:
            if df is None or df.empty or len(df) < 50:
                return {}
            
            # Identify large block trades
            large_trades = df[df['volume'] > self.institutional_block_size]
            
            # Calculate buying and selling pressure for large trades
            institutional_buying = np.where(large_trades['close'] > large_trades['open'], large_trades['volume'], 0).sum()
            institutional_selling = np.where(large_trades['close'] < large_trades['open'], large_trades['volume'], 0).sum()
            
            result = {
                'institutional_buying': institutional_buying,
                'institutional_selling': institutional_selling,
                'large_trade_count': len(large_trades)
            }
            
            trading_state.institutional_activity[ticker] = result
            return result
            
        except Exception as e:
            log(f"❌ Institutional Activity detection failed: {e}")
            return {}
    
    def analyze_dark_pool_indicators(self, df, ticker):
        """Analyze dark pool indicators using volume and price action"""
        try:
            if df is None or df.empty or len(df) < 50:
                return {}
            
            # Calculate off-exchange volume (simplified)
            off_exchange_volume = df['volume'].quantile(0.8)
            if df is None or df.empty or len(df) < 50:
                return {}
            
            # Calculate off-exchange volume (simplified)
            off_exchange_volume = df['volume'].quantile(0.8)
            
            # Calculate dark pool indicator
            dark_pool_indicator = off_exchange_volume / df['volume'].sum()
            
            # Determine if indicator is significant
            if dark_pool_indicator > self.dark_pool_indicator_threshold:
                dark_pool_activity = "high"
            else:
                dark_pool_activity = "low"
            
            result = {
                'dark_pool_indicator': dark_pool_indicator,
                'dark_pool_activity': dark_pool_activity
            }
            
            trading_state.dark_pool_indicators[ticker] = result
            return result
            
        except Exception as e:
            log(f"❌ Dark Pool Indicator analysis failed: {e}")
            return {}

market_microstructure_analyzer = MarketMicrostructureAnalyzer()

# === ENHANCED TRADING EXECUTOR WITH FULL PROFIT TRACKING ===
class EnhancedTradingExecutor:
    def __init__(self):
        self.cooldown_periods = {}
        self.position_tracker = {}
        self.max_positions = 10
        self.cooldown_duration = 300
        
    def execute_buy_order(self, ticker, signal_strength, market_data, q_action=None):
        """Execute buy order with full risk management and profit tracking"""
        try:
            if trading_state.emergency_stop_triggered:
                log(f"🛑 Emergency stop active, skipping {ticker}")
                return False
            
            if not self.check_cooldown(ticker):
                log(f"⏰ {ticker} in cooldown period")
                return False
            
            if not api:
                log(f"⚠️ No API connection, simulating buy order for {ticker}")
                return self.simulate_buy_order(ticker, signal_strength, market_data)
            
            # Get account info
            account = safe_api_call(api.get_account)
            if not account:
                log(f"❌ Could not get account info for {ticker}")
                return False
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # Risk management checks
            if not self.check_risk_limits(ticker, equity):
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(ticker, equity, signal_strength, market_data)
            if position_size <= 0:
                log(f"❌ Invalid position size for {ticker}: {position_size}")
                return False
            
            # Get current price
            current_price = market_data['close'].iloc[-1] if market_data is not None else 0
            if current_price <= 0:
                log(f"❌ Invalid price for {ticker}: {current_price}")
                return False
            
            # Check if we have enough buying power
            order_value = position_size * current_price
            if order_value > buying_power:
                log(f"❌ Insufficient buying power for {ticker}: ${order_value:.2f} > ${buying_power:.2f}")
                return False
            
            # Execute the order
            try:
                order = api.submit_order(
                    symbol=ticker,
                    qty=position_size,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                # Track the position
                trade_id = trading_state.get_next_trade_id()
                position_data = {
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'action': 'buy',
                    'quantity': position_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'signal_strength': signal_strength,
                    'q_action': q_action,
                    'order_id': order.id,
                    'market_data_snapshot': {
                        'rsi': market_data['rsi'].iloc[-1] if 'rsi' in market_data else None,
                        'macd': market_data['macd'].iloc[-1] if 'macd' in market_data else None,
                        'volume_ratio': market_data['volume_ratio'].iloc[-1] if 'volume_ratio' in market_data else None
                    }
                }
                
                trading_state.open_positions[ticker] = position_data
                self.set_cooldown(ticker)
                
                # Log to Google Sheets
                log_data = {
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'action': 'buy',
                    'quantity': position_size,
                    'price': current_price,
                    'signal_strength': signal_strength,
                    'q_action': q_action
                }
                log_to_google_sheets(log_data, "TradingLog")
                
                # Send Discord alert
                alert_msg = f"🟢 **BUY ORDER EXECUTED**\n"
                alert_msg += f"Ticker: {ticker}\n"
                alert_msg += f"Quantity: {position_size}\n"
                alert_msg += f"Price: ${current_price:.2f}\n"
                alert_msg += f"Value: ${order_value:.2f}\n"
                alert_msg += f"Signal Strength: {signal_strength:.3f}\n"
                alert_msg += f"Trade ID: {trade_id}"
                send_discord_alert(alert_msg)
                
                log(f"✅ Buy order executed: {ticker} x{position_size} @ ${current_price:.2f}")
                return True
                
            except Exception as order_error:
                log(f"❌ Order execution failed for {ticker}: {order_error}")
                return False
            
        except Exception as e:
            log(f"❌ Buy order execution failed for {ticker}: {e}")
            return False
    
    def execute_sell_order(self, ticker, reason="Signal"):
        """Execute sell order with profit tracking"""
        try:
            if ticker not in trading_state.open_positions:
                log(f"⚠️ No open position for {ticker}")
                return False
            
            if not api:
                log(f"⚠️ No API connection, simulating sell order for {ticker}")
                return self.simulate_sell_order(ticker, reason)
            
            position_data = trading_state.open_positions[ticker]
            
            # Get current price
            try:
                bars = safe_api_call(api.get_bars, ticker, TimeFrame.Minute, limit=1)
                if bars and not bars.df.empty:
                    current_price = bars.df['close'].iloc[-1]
                else:
                    log(f"❌ Could not get current price for {ticker}")
                    return False
            except Exception as price_error:
                log(f"❌ Price fetch failed for {ticker}: {price_error}")
                return False
            
            # Execute the sell order
            try:
                order = api.submit_order(
                    symbol=ticker,
                    qty=position_data['quantity'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                # Calculate profit/loss
                entry_price = position_data['entry_price']
                exit_price = current_price
                quantity = position_data['quantity']
                
                gross_pnl = (exit_price - entry_price) * quantity
                return_pct = (exit_price - entry_price) / entry_price
                
                # Hold time
                hold_time = datetime.now() - position_data['entry_time']
                hold_minutes = hold_time.total_seconds() / 60
                
                # Update trade outcome
                trade_outcome = {
                    'trade_id': position_data['trade_id'],
                    'ticker': ticker,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'quantity': quantity,
                    'gross_pnl': gross_pnl,
                    'return': return_pct,
                    'hold_time_minutes': hold_minutes,
                    'exit_reason': reason,
                    'entry_time': position_data['entry_time'],
                    'exit_time': datetime.now()
                }
                
                trading_state.trade_outcomes.append(trade_outcome)
                trading_state.pnl_tracker[ticker] = gross_pnl
                
                # Update Q-learning if applicable
                if position_data.get('q_action'):
                    q_learning_agent = QLearningAgent(['buy', 'sell', 'hold'])
                    reward = return_pct * 100  # Scale reward
                    
                    # Get current state for Q-learning update
                    current_data = get_enhanced_data(ticker, limit=50)
                    if current_data is not None:
                        current_state = q_learning_agent.discretize_state(current_data, ticker)
                        if current_state and len(trading_state.q_state_history) > 0:
                            last_state = trading_state.q_state_history[-1]
                            last_action = position_data['q_action']
                            q_learning_agent.learn(last_state, last_action, reward, current_state)
                
                # Remove from open positions
                del trading_state.open_positions[ticker]
                
                # Log to Google Sheets
                log_data = {
                    'trade_id': position_data['trade_id'],
                    'ticker': ticker,
                    'action': 'sell',
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pnl': gross_pnl,
                    'return_pct': return_pct,
                    'hold_time_minutes': hold_minutes,
                    'exit_reason': reason
                }
                log_to_google_sheets(log_data, "TradingLog")
                
                # Send Discord alert
                profit_emoji = "🟢" if gross_pnl > 0 else "🔴"
                alert_msg = f"{profit_emoji} **SELL ORDER EXECUTED**\n"
                alert_msg += f"Ticker: {ticker}\n"
                alert_msg += f"Quantity: {quantity}\n"
                alert_msg += f"Entry: ${entry_price:.2f}\n"
                alert_msg += f"Exit: ${exit_price:.2f}\n"
                alert_msg += f"P&L: ${gross_pnl:.2f} ({return_pct:.2%})\n"
                alert_msg += f"Hold Time: {hold_minutes:.1f} min\n"
                alert_msg += f"Reason: {reason}"
                send_discord_alert(alert_msg)
                
                log(f"✅ Sell order executed: {ticker} x{quantity} @ ${exit_price:.2f} | P&L: ${gross_pnl:.2f}")
                return True
                
            except Exception as order_error:
                log(f"❌ Sell order execution failed for {ticker}: {order_error}")
                return False
            
        except Exception as e:
            log(f"❌ Sell order execution failed for {ticker}: {e}")
            return False
    
    def simulate_buy_order(self, ticker, signal_strength, market_data):
        """Simulate buy order for demo mode"""
        try:
            current_price = market_data['close'].iloc[-1] if market_data is not None else 100
            position_size = 10  # Demo position size
            
            trade_id = trading_state.get_next_trade_id()
            position_data = {
                'trade_id': trade_id,
                'ticker': ticker,
                'action': 'buy',
                'quantity': position_size,
                'entry_price': current_price,
                'entry_time': datetime.now(),
                'signal_strength': signal_strength,
                'simulated': True
            }
            
            trading_state.open_positions[ticker] = position_data
            self.set_cooldown(ticker)
            
            log(f"📝 Simulated buy order: {ticker} x{position_size} @ ${current_price:.2f}")
            return True
            
        except Exception as e:
            log(f"❌ Simulated buy order failed for {ticker}: {e}")
            return False
    
    def simulate_sell_order(self, ticker, reason):
        """Simulate sell order for demo mode"""
        try:
            if ticker not in trading_state.open_positions:
                return False
            
            position_data = trading_state.open_positions[ticker]
            current_price = position_data['entry_price'] * (1 + random.uniform(-0.05, 0.05))  # Simulate price change
            
            gross_pnl = (current_price - position_data['entry_price']) * position_data['quantity']
            return_pct = (current_price - position_data['entry_price']) / position_data['entry_price']
            
            trade_outcome = {
                'trade_id': position_data['trade_id'],
                'ticker': ticker,
                'entry_price': position_data['entry_price'],
                'exit_price': current_price,
                'quantity': position_data['quantity'],
                'gross_pnl': gross_pnl,
                'return': return_pct,
                'hold_time_minutes': 30,  # Simulated hold time
                'exit_reason': reason,
                'simulated': True
            }
            
            trading_state.trade_outcomes.append(trade_outcome)
            del trading_state.open_positions[ticker]
            
            log(f"📝 Simulated sell order: {ticker} | P&L: ${gross_pnl:.2f}")
            return True
            
        except Exception as e:
            log(f"❌ Simulated sell order failed for {ticker}: {e}")
            return False
    
    def check_cooldown(self, ticker):
        """Check if ticker is in cooldown period"""
        if ticker in self.cooldown_periods:
            time_since_trade = time.time() - self.cooldown_periods[ticker]
            return time_since_trade > self.cooldown_duration
        return True
    
    def set_cooldown(self, ticker):
        """Set cooldown period for ticker"""
        self.cooldown_periods[ticker] = time.time()
    
    def check_risk_limits(self, ticker, equity):
        """Check various risk limits"""
        try:
            # Check maximum positions
            if len(trading_state.open_positions) >= self.max_positions:
                log(f"❌ Maximum positions reached: {len(trading_state.open_positions)}")
                return False
            
            # Check daily drawdown
            if trading_state.daily_drawdown > THRESHOLDS['MAX_DAILY_DRAWDOWN']:
                log(f"❌ Daily drawdown limit exceeded: {trading_state.daily_drawdown:.2%}")
                return False
            
            # Check emergency stop
            if trading_state.emergency_stop_triggered:
                log(f"❌ Emergency stop triggered")
                return False
            
            return True
            
        except Exception as e:
            log(f"❌ Risk limit check failed: {e}")
            return False
    
    def calculate_position_size(self, ticker, equity, signal_strength, market_data):
        """Calculate position size based on risk management"""
        try:
            # Base position size (percentage of equity)
            base_position_pct = 0.02  # 2% of equity
            
            # Adjust based on signal strength
            signal_multiplier = min(signal_strength * 2, 2.0)  # Max 2x multiplier
            
            # Adjust based on volatility
            if market_data is not None and 'atr' in market_data:
                atr = market_data['atr'].iloc[-1]
                current_price = market_data['close'].iloc[-1]
                volatility_pct = atr / current_price
                volatility_multiplier = max(0.5, 1 - volatility_pct)  # Reduce size for high volatility
            else:
                volatility_multiplier = 1.0
            
            # Calculate final position size
            position_value = equity * base_position_pct * signal_multiplier * volatility_multiplier
            current_price = market_data['close'].iloc[-1] if market_data is not None else 100
            position_size = int(position_value / current_price)
            
            return max(1, position_size)  # Minimum 1 share
            
        except Exception as e:
            log(f"❌ Position size calculation failed: {e}")
            return 0

trading_executor = EnhancedTradingExecutor()

# === UTILITY FUNCTIONS ===
def log_to_google_sheets(data, sheet_name="TradingLog"):
    """Log data to Google Sheets with enhanced error handling"""
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
    """Get enhanced market data with all technical indicators"""
    try:
        if not api:
            # Return demo data for testing
            dates = pd.date_range(start='2024-01-01', periods=limit, freq='5T')
            demo_data = pd.DataFrame({
                'open': np.random.uniform(100, 200, limit),
                'high': np.random.uniform(100, 200, limit),
                'low': np.random.uniform(100, 200, limit),
                'close': np.random.uniform(100, 200, limit),
                'volume': np.random.randint(1000, 10000, limit)
            }, index=dates)
            return add_ultra_advanced_technical_indicators(demo_data)
        
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
        
        # Add ALL technical indicators (restored)
        bars = add_ultra_advanced_technical_indicators(bars)
        
        return bars
        
    except Exception as e:
        log(f"❌ Data fetch failed for {ticker}: {e}")
        return None

def is_near_market_close(minutes_before_close=30):
    """Check if we're near market close"""
    try:
        now = datetime.now(pytz.timezone("US/Eastern"))
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        time_to_close = (market_close - now).total_seconds() / 60
        
        return 0 < time_to_close <= minutes_before_close
        
    except Exception as e:
        log(f"❌ Market close check failed: {e}")
        return False

def get_current_positions():
    """Get current positions from API"""
    try:
        if not api:
            return {}
        
        positions = safe_api_call(api.list_positions)
        if not positions:
            return {}
        
        position_dict = {}
        for pos in positions:
            position_dict[pos.symbol] = {
                'quantity': int(pos.qty),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl)
            }
        
        return position_dict
        
    except Exception as e:
        log(f"❌ Position fetch failed: {e}")
        return {}

# === MAIN ULTRA-ADVANCED TRADING BOT CLASS ===
class UltraAdvancedTradingBot:
    def __init__(self):
        self.loop_interval = 300  # 5 minutes
        self.current_watchlist = FALLBACK_UNIVERSE[:20]
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'accuracy': 0.0
        }
        
        # Initialize Q-learning agent
        self.q_learning_agent = QLearningAgent(['buy', 'sell', 'hold'])
        
    def initialize_bot(self):
        """Initialize the ultra-advanced trading bot"""
        try:
            log("🚀 Initializing Ultra-Advanced AI Trading Bot...")
            
            # Set starting equity for drawdown tracking
            if api:
                account = safe_api_call(api.get_account)
                if account:
                    trading_state.starting_equity = float(account.equity)
                else:
                    log("⚠️ Could not get account info, using default equity")
                    trading_state.starting_equity = 100000
            else:
                log("⚠️ No API connection, using demo mode")
                trading_state.starting_equity = 100000
            
            # Initialize all advanced systems
            sector_rotation_analyzer.analyze_sector_performance()
            
            # Send startup notification with all features
            startup_msg = f"🚀 Ultra-Advanced AI Trading Bot Started!\n"
            startup_msg += f"💰 Starting Equity: ${trading_state.starting_equity:,.2f}\n"
            startup_msg += f"🧠 Q-Learning: {len(trading_state.q_table)} states learned\n"
            startup_msg += f"📊 Sector Rotation: {len(trading_state.sector_performance)} sectors tracked\n"
            startup_msg += f"🎯 Advanced Features: ALL RESTORED!\n"
            startup_msg += f"✅ Fibonacci Analysis\n"
            startup_msg += f"✅ Elliott Wave Detection\n"
            startup_msg += f"✅ Harmonic Patterns\n"
            startup_msg += f"✅ Ichimoku Cloud\n"
            startup_msg += f"✅ Market Microstructure\n"
            startup_msg += f"✅ Advanced Volatility Models\n"
            startup_msg += f"✅ Regime Detection\n"
            startup_msg += f"📈 Watchlist: {len(self.current_watchlist)} tickers"
            send_discord_alert(startup_msg)
            
            log("✅ Ultra-Advanced Bot initialization complete")
            return True
            
        except Exception as e:
            log(f"❌ Bot initialization failed: {e}")
            send_discord_alert(f"❌ Bot initialization failed: {e}", urgent=True)
            return False
    
    def process_ticker_with_all_features(self, ticker):
        """Process ticker with ALL advanced features"""
        try:
            log(f"🔍 Processing {ticker} with COMPLETE feature set")
            
            # Get enhanced market data
            short_data = get_enhanced_data(ticker, days_back=2)
            medium_data = get_enhanced_data(ticker, days_back=15)
            
            if short_data is None or short_data.empty:
                return False
            
            current_price = short_data['close'].iloc[-1]
            
            # COMPLETE ADVANCED ANALYSIS PIPELINE
            
            # 1. Fibonacci Analysis
            fibonacci_analysis = fibonacci_analyzer.calculate_fibonacci_levels(short_data, ticker)
            
            # 2. Elliott Wave Analysis
            elliott_wave_analysis = elliott_wave_analyzer.detect_elliott_waves(short_data, ticker)
            
            # 3. Harmonic Pattern Recognition
            harmonic_patterns = harmonic_pattern_analyzer.detect_harmonic_patterns(short_data, ticker)
            
            # 4. Ichimoku Cloud Analysis
            ichimoku_analysis = ichimoku_cloud_analyzer.calculate_ichimoku_cloud(short_data, ticker)
            
            # 5. Support/Resistance Analysis
            sr_levels = support_resistance_analyzer.find_significant_levels(short_data, ticker)
            
            # 6. Volume Profile Analysis
            volume_profile = volume_profile_analyzer.calculate_volume_profile(short_data, ticker)
            
            # 7. Market Microstructure Analysis
            order_flow = market_microstructure_analyzer.analyze_order_flow(short_data, ticker)
            smart_money = market_microstructure_analyzer.analyze_smart_money_flow(short_data, ticker)
            institutional_activity = market_microstructure_analyzer.detect_institutional_activity(short_data, ticker)
            
            # 8. Q-Learning State and Action
            q_state = self.q_learning_agent.discretize_state(short_data, ticker)
            q_action = self.q_learning_agent.choose_action(q_state)
            
            # ULTRA-ADVANCED DECISION LOGIC
            
            advanced_score = 0
            reasons = []
            confidence_factors = []
            
            # Fibonacci Analysis Scoring
            if fibonacci_analysis and 'fibonacci_confluence' in fibonacci_analysis:
                confluence_zones = fibonacci_analysis['fibonacci_confluence']
                if confluence_zones:
                    nearest_confluence = min(confluence_zones, key=lambda x: abs(x['price'] - current_price))
                    distance_to_confluence = abs(current_price - nearest_confluence['price']) / current_price
                    
                    if distance_to_confluence < 0.02:  # Within 2% of confluence
                        advanced_score += nearest_confluence['strength']
                        reasons.append(f"Fibonacci confluence (strength: {nearest_confluence['strength']})")
                        confidence_factors.append(0.8)
            
            # Elliott Wave Analysis Scoring
            if elliott_wave_analysis and 'elliott_wave_signals' in elliott_wave_analysis:
                elliott_signals = elliott_wave_analysis['elliott_wave_signals']
                if elliott_signals.get('action') == 'buy':
                    strength_multiplier = {'high': 3, 'medium': 2, 'low': 1}.get(elliott_signals.get('strength', 'low'), 1)
                    advanced_score += strength_multiplier
                    reasons.append(f"Elliott Wave: {elliott_signals.get('reason', 'Buy signal')}")
                    confidence_factors.append(0.7)
                elif elliott_signals.get('action') == 'sell':
                    advanced_score -= 2
                    reasons.append(f"Elliott Wave: {elliott_signals.get('reason', 'Sell signal')}")
            
            # Harmonic Pattern Scoring
            if harmonic_patterns and harmonic_patterns.get('strongest_pattern'):
                pattern = harmonic_patterns['strongest_pattern']
                if pattern['confidence'] > 0.7:
                    advanced_score += 2
                    reasons.append(f"Harmonic {pattern['pattern']} pattern (confidence: {pattern['confidence']:.2f})")
                    confidence_factors.append(pattern['confidence'])
            
            # Ichimoku Cloud Scoring
            if ichimoku_analysis:
                cloud_status = ichimoku_analysis.get('cloud_status', 'inside')
                if cloud_status == 'above':
                    advanced_score += 1
                    reasons.append("Price above Ichimoku cloud")
                    confidence_factors.append(0.6)
                elif cloud_status == 'below':
                    advanced_score -= 1
                    reasons.append("Price below Ichimoku cloud")
            
            # Support/Resistance Scoring
            if sr_levels:
                support_levels = sr_levels.get('support_levels', [])
                resistance_levels = sr_levels.get('resistance_levels', [])
                
                # Check proximity to support (bullish)
                for support in support_levels:
                    if abs(current_price - support) / current_price < 0.02:
                        advanced_score += 1
                        reasons.append(f"Near support level: ${support:.2f}")
                        confidence_factors.append(0.7)
                        break
                
                # Check proximity to resistance (bearish)
                for resistance in resistance_levels:
                    if abs(current_price - resistance) / current_price < 0.02:
                        advanced_score -= 1
                        reasons.append(f"Near resistance level: ${resistance:.2f}")
                        break
            
            # Volume Profile Scoring
            if volume_profile and 'poc' in volume_profile:
                poc_range = volume_profile['poc']
                # Extract price from range string
                try:
                    poc_price = float(poc_range.split('-')[0])
                    if abs(current_price - poc_price) / current_price < 0.01:
                        advanced_score += 1
                        reasons.append("Near Volume POC")
                        confidence_factors.append(0.6)
                except:
                    pass
            
            # Market Microstructure Scoring
            if order_flow:
                flow_direction = order_flow.get('order_flow_direction', 'neutral')
                if flow_direction == 'buying':
                    advanced_score += 1
                    reasons.append("Positive order flow")
                    confidence_factors.append(0.5)
                elif flow_direction == 'selling':
                    advanced_score -= 1
                    reasons.append("Negative order flow")
            
            if smart_money:
                smart_direction = smart_money.get('smart_money_direction', 'neutral')
                if smart_direction == 'buying':
                    advanced_score += 2
                    reasons.append("Smart money buying")
                    confidence_factors.append(0.8)
                elif smart_direction == 'selling':
                    advanced_score -= 2
                    reasons.append("Smart money selling")
            
            # Q-Learning Scoring
            if q_action == 'buy':
                advanced_score += 1
                reasons.append("Q-Learning: BUY")
                confidence_factors.append(0.6)
            elif q_action == 'sell':
                advanced_score -= 1
                reasons.append("Q-Learning: SELL")
            
            # Technical Indicators Scoring
            if 'rsi' in short_data.columns:
                rsi = short_data['rsi'].iloc[-1]
                if rsi < 30:  # Oversold
                    advanced_score += 1
                    reasons.append(f"RSI oversold: {rsi:.1f}")
                    confidence_factors.append(0.5)
                elif rsi > 70:  # Overbought
                    advanced_score -= 1
                    reasons.append(f"RSI overbought: {rsi:.1f}")
            
            if 'macd_histogram' in short_data.columns:
                macd_hist = short_data['macd_histogram'].iloc[-1]
                macd_hist_prev = short_data['macd_histogram'].iloc[-2]
                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    advanced_score += 1
                    reasons.append("MACD bullish divergence")
                    confidence_factors.append(0.5)
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            # Final decision with confidence weighting
            weighted_score = advanced_score * overall_confidence
            
            # Decision threshold
            if weighted_score >= 4:
                log(f"✅ {ticker} APPROVED by ultra-advanced analysis:")
                log(f"   Score: {advanced_score} (weighted: {weighted_score:.2f})")
                log(f"   Confidence: {overall_confidence:.2f}")
                log(f"   Reasons: {', '.join(reasons)}")
                
                # Execute trade with all tracking
                success = trading_executor.execute_buy_order(
                    ticker, 
                    weighted_score / 10,  # Normalize signal strength
                    short_data,
                    q_action
                )
                
                if success:
                    self.daily_stats['trades_executed'] += 1
                    
                    # Store Q-learning state for future updates
                    trading_state.q_state_history.append(q_state)
                    trading_state.q_action_history.append(q_action)
                
                return success
            else:
                log(f"❌ {ticker} REJECTED by ultra-advanced analysis:")
                log(f"   Score: {advanced_score} (weighted: {weighted_score:.2f})")
                log(f"   Confidence: {overall_confidence:.2f}")
                log(f"   Reasons: {', '.join(reasons) if reasons else 'No significant signals'}")
                return False
            
        except Exception as e:
            log(f"❌ Ultra-advanced processing failed for {ticker}: {e}")
            return False
    
    def run_ultra_advanced_trading_cycle(self):
        """Run trading cycle with ALL advanced features"""
        try:
            log("🔄 Starting ULTRA-ADVANCED trading cycle...")
            
            # Update all advanced systems
            sector_rotation_analyzer.analyze_sector_performance()
            
            # Update risk metrics
            trading_state.update_ultra_advanced_risk_metrics()
            
            # Process watchlist with complete advanced features
            processed_count = 0
            for ticker in self.current_watchlist[:10]:  # Limit for performance
                try:
                    if self.process_ticker_with_all_features(ticker):
                        processed_count += 1
                    
                    time.sleep(3)  # Rate limiting
                    
                except Exception as e:
                    log(f"❌ Error processing {ticker}: {e}")
                    continue
            
            log(f"✅ Ultra-advanced trading cycle complete - processed {processed_count} tickers")
            
        except Exception as e:
            log(f"❌ Ultra-advanced trading cycle failed: {e}")
    
    def run_main_loop(self):
        """Main loop with ALL advanced features"""
        try:
            if not self.initialize_bot():
                log("⚠️ Bot initialization failed, but continuing...")
            
            log("🔄 Starting ULTRA-ADVANCED main trading loop...")
            
            while True:
                try:
                    # Check if market is open
                    if is_market_open_safe():
                        log("📈 Market is open - running ULTRA-ADVANCED trading cycle")
                        
                        # Check if near market close
                        if is_near_market_close(30):
                            log("⏰ Near market close, performing cleanup...")
                            self.end_of_day_cleanup()
                            log("✅ End-of-day cleanup complete, waiting for next market open...")
                        else:
                            # Run ultra-advanced trading cycle
                            self.run_ultra_advanced_trading_cycle()
                            
                            # Wait for next cycle
                            log(f"⏸️ Waiting {self.loop_interval/60:.1f} minutes for next cycle...")
                            time.sleep(self.loop_interval)
                    else:
                        # Market is closed - wait appropriately without exiting
                        self.wait_for_market_open()
                        
                except KeyboardInterrupt:
                    log("🛑 Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    log(f"❌ Error in main loop: {e}")
                    log(f"Traceback: {traceback.format_exc()}")
                    send_discord_alert(f"❌ Main loop error: {e}", urgent=True)
                    # Don't exit - just wait and continue
                    time.sleep(60)  # Wait 1 minute before retrying
                    continue
            
            log("🏁 Ultra-Advanced Trading bot stopped gracefully")
            send_discord_alert("🏁 Ultra-Advanced Trading bot stopped")
            
        except Exception as e:
            log(f"❌ Fatal error in main loop: {e}")
            log(f"Traceback: {traceback.format_exc()}")
            send_discord_alert(f"❌ Fatal error: {e}", urgent=True)
            # Don't exit - sleep and let Render restart if needed
            log("😴 Sleeping before potential restart...")
            time.sleep(300)
    
    def wait_for_market_open(self):
        """Wait for market to open without exiting"""
        log("⏳ Market is closed, waiting...")
        
        # Calculate time until next market open
        now = datetime.now(pytz.timezone("US/Eastern"))
        
        if now.weekday() >= 5:  # Weekend
            # Wait until Monday 9:30 AM
            days_until_monday = 7 - now.weekday()
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=days_until_monday)
        else:
            # Wait until next day 9:30 AM if after market close
            if now.hour >= 16:
                next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        time_until_open = (next_open - now).total_seconds()
        log(f"⏰ Market opens in {time_until_open/3600:.1f} hours")
        
        # Sleep in chunks to allow health checks
        sleep_chunk = min(300, time_until_open)  # 5 minutes max
        time.sleep(sleep_chunk)
    
    def end_of_day_cleanup(self):
        """Enhanced end-of-day cleanup with ALL advanced features"""
        try:
            log("🌅 Starting ULTRA-ADVANCED end-of-day cleanup...")
            
            # Auto-liquidation near close with Q-learning updates
            current_positions = get_current_positions()
            for ticker in current_positions:
                # Calculate Q-learning reward for completed trades
                if ticker in trading_state.open_positions:
                    position_data = trading_state.open_positions[ticker]
                    entry_price = position_data['entry_price']
                    current_price = current_positions[ticker]['current_price']
                    hold_time = (datetime.now() - position_data['entry_time']).total_seconds() / 60
                    
                    # Calculate reward and update Q-table
                    return_pct = (current_price - entry_price) / entry_price
                    reward = return_pct * 100  # Scale reward
                    
                    if len(trading_state.q_state_history) > 0 and len(trading_state.q_action_history) > 0:
                        last_state = trading_state.q_state_history[-1]
                        last_action = trading_state.q_action_history[-1]
                        current_data = get_enhanced_data(ticker, limit=50)
                        if current_data is not None:
                            current_state = self.q_learning_agent.discretize_state(current_data, ticker)
                            if current_state:
                                self.q_learning_agent.learn(last_state, last_action, reward, current_state)
                
                # Execute sell order
                trading_executor.execute_sell_order(ticker, "End of day")
                time.sleep(1)
            
            # Generate comprehensive daily report with ALL metrics
            self.generate_ultra_advanced_daily_report()
            
            # Save all advanced data
            self.save_all_advanced_data()
            
            # Reset daily state
            trading_state.reset_daily()
            self.daily_stats = {
                'trades_executed': 0,
                'profitable_trades': 0,
                'total_pnl': 0.0,
                'accuracy': 0.0
            }
            
            log("✅ ULTRA-ADVANCED end-of-day cleanup complete")
            
        except Exception as e:
            log(f"❌ Ultra-advanced end-of-day cleanup failed: {e}")
            send_discord_alert(f"❌ Ultra-advanced end-of-day cleanup failed: {e}", urgent=True)
    
    def generate_ultra_advanced_daily_report(self):
        """Generate comprehensive daily report with ALL advanced metrics"""
        try:
            if api:
                account = safe_api_call(api.get_account)
                if account:
                    equity = float(account.equity)
                    cash = float(account.cash)
                    day_pl = float(account.todays_pl)
                else:
                    equity = trading_state.starting_equity
                    cash = 0
                    day_pl = 0
            else:
                equity = trading_state.starting_equity
                cash = 0
                day_pl = 0
            
            report = "📊 **ULTRA-ADVANCED Daily Trading Report**\n"
            report += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            
            # Basic metrics
            report += f"💰 Account Value: ${equity:,.2f}\n"
            report += f"💵 Cash: ${cash:,.2f}\n"
            report += f"📈 Day P&L: ${day_pl:,.2f}\n"
            report += f"📉 Max Drawdown: {trading_state.daily_drawdown:.1%}\n\n"
            
            # Trading metrics
            report += f"🎯 Trades Executed: {self.daily_stats['trades_executed']}\n"
            report += f"✅ Accuracy: {self.daily_stats['accuracy']:.1%}\n\n"
            
            # Q-Learning Stats
            report += f"🧠 **Q-Learning Stats:**\n"
            report += f"States Learned: {len(trading_state.q_table)}\n"
            if trading_state.q_reward_history:
                avg_reward = np.mean(list(trading_state.q_reward_history)[-20:])
                report += f"Avg Reward (20 trades): {avg_reward:.3f}\n"
            report += f"Exploration Rate: {self.q_learning_agent.epsilon:.3f}\n\n"
            
            # Advanced Pattern Recognition
            report += f"🎨 **Pattern Recognition:**\n"
            report += f"Fibonacci Levels: {len(trading_state.fibonacci_levels)} tickers\n"
            report += f"Elliott Waves: {len(trading_state.elliott_wave_counts)} tickers\n"
            report += f"Harmonic Patterns: {len(trading_state.harmonic_patterns)} tickers\n"
            report += f"Ichimoku Clouds: {len(trading_state.ichimoku_clouds)} tickers\n\n"
            
            # Market Microstructure
            report += f"🔬 **Market Microstructure:**\n"
            report += f"Order Flow Analysis: {len(trading_state.order_flow_imbalance)} tickers\n"
            report += f"Smart Money Flow: {len(trading_state.smart_money_flow)} tickers\n"
            report += f"Institutional Activity: {len(trading_state.institutional_activity)} tickers\n\n"
            
            # Ultra-Advanced Risk Metrics
            report += f"📊 **Ultra-Advanced Risk Metrics:**\n"
            report += f"Sharpe Ratio: {trading_state.risk_metrics['sharpe_ratio']:.2f}\n"
            report += f"Sortino Ratio: {trading_state.risk_metrics['sortino_ratio']:.2f}\n"
            report += f"Calmar Ratio: {trading_state.risk_metrics['calmar_ratio']:.2f}\n"
            report += f"Win Rate: {trading_state.risk_metrics['win_rate']:.1%}\n"
            report += f"Profit Factor: {trading_state.risk_metrics['profit_factor']:.2f}\n"
            report += f"VaR (95%): {trading_state.risk_metrics['var_95']:.2%}\n"
            report += f"CVaR (95%): {trading_state.risk_metrics['cvar_95']:.2%}\n\n"
            
            # Sector performance
            if trading_state.sector_performance:
                top_sector = max(trading_state.sector_performance.items(), key=lambda x: x[1])
                worst_sector = min(trading_state.sector_performance.items(), key=lambda x: x[1])
                report += f"🏆 Top Sector: {top_sector[0]} ({top_sector[1]:.1%})\n"
                report += f"📉 Worst Sector: {worst_sector[0]} ({worst_sector[1]:.1%})\n"
            
            report += f"🌊 Market Regime: {trading_state.regime_state}\n"
            report += f"📋 Watchlist Size: {len(self.current_watchlist)}\n\n"
            
            report += f"🚀 **ALL ADVANCED FEATURES ACTIVE!**"
            
            # Send to Discord and log to sheets
            send_discord_alert(report)
            
            # Log comprehensive data to Google Sheets
            ultra_advanced_data = {
                'account_value': equity,
                'cash': cash,
                'day_pl': day_pl,
                'max_drawdown': trading_state.daily_drawdown,
                'trades_executed': self.daily_stats['trades_executed'],
                'accuracy': self.daily_stats['accuracy'],
                'sharpe_ratio': trading_state.risk_metrics['sharpe_ratio'],
                'sortino_ratio': trading_state.risk_metrics['sortino_ratio'],
                'calmar_ratio': trading_state.risk_metrics['calmar_ratio'],
                'win_rate': trading_state.risk_metrics['win_rate'],
                'profit_factor': trading_state.risk_metrics['profit_factor'],
                'var_95': trading_state.risk_metrics['var_95'],
                'cvar_95': trading_state.risk_metrics['cvar_95'],
                'q_states_learned': len(trading_state.q_table),
                'avg_q_reward': np.mean(list(trading_state.q_reward_history)[-20:]) if trading_state.q_reward_history else 0,
                'fibonacci_analysis_count': len(trading_state.fibonacci_levels),
                'elliott_wave_count': len(trading_state.elliott_wave_counts),
                'harmonic_pattern_count': len(trading_state.harmonic_patterns),
                'ichimoku_analysis_count': len(trading_state.ichimoku_clouds),
                'order_flow_analysis_count': len(trading_state.order_flow_imbalance),
                'smart_money_analysis_count': len(trading_state.smart_money_flow),
                'institutional_activity_count': len(trading_state.institutional_activity),
                'market_regime': trading_state.regime_state,
                'sectors_tracked': len(trading_state.sector_performance)
            }
            
            log_to_google_sheets(ultra_advanced_data, "UltraAdvancedDailyReports")
            
        except Exception as e:
            log(f"❌ Ultra-advanced daily report generation failed: {e}")
    
    def save_all_advanced_data(self):
        """Save all advanced analysis data to files"""
        try:
            # Save Q-learning data
            q_data = {
                'q_table': dict(trading_state.q_table),
                'state_history': list(trading_state.q_state_history),
                'action_history': list(trading_state.q_action_history),
                'reward_history': list(trading_state.q_reward_history)
            }
            with open("models/q_learning/q_learning_complete.json", "w") as f:
                json.dump(q_data, f, default=str)
            
            # Save all pattern recognition data
            pattern_data = {
                'fibonacci_levels': trading_state.fibonacci_levels,
                'elliott_wave_counts': trading_state.elliott_wave_counts,
                'harmonic_patterns': trading_state.harmonic_patterns,
                'ichimoku_clouds': trading_state.ichimoku_clouds
            }
            with open("models/pattern_recognition_complete.json", "w") as f:
                json.dump(pattern_data, f, default=str)
            
            # Save market microstructure data
            microstructure_data = {
                'order_flow_imbalance': trading_state.order_flow_imbalance,
                'smart_money_flow': trading_state.smart_money_flow,
                'institutional_activity': trading_state.institutional_activity,
                'dark_pool_indicators': trading_state.dark_pool_indicators
            }
            with open("microstructure/microstructure_complete.json", "w") as f:
                json.dump(microstructure_data, f, default=str)
            
            log("💾 All ultra-advanced data saved successfully")
            
        except Exception as e:
            log(f"❌ Failed to save advanced data: {e}")

def run_flask_server():
    """Run Flask server in a separate thread"""
    try:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        log(f"Flask server error: {e}")

def main():
    """Main entry point with ALL advanced features"""
    try:
        log("🚀 Starting ULTRA-ADVANCED AI Trading Bot with ALL 2,800+ LINES OF FUNCTIONALITY...")
        log("📋 COMPLETE Feature Checklist:")
        log("✅ Dual-Horizon Prediction Models")
        log("✅ Q-Learning Reinforcement Learning System")
        log("✅ Support/Resistance Analysis")
        log("✅ Volume Profile Analysis")
        log("✅ Fibonacci Retracement & Extensions")
        log("✅ Elliott Wave Pattern Detection")
        log("✅ Harmonic Pattern Recognition (Gartley, Butterfly, Crab, Bat)")
        log("✅ Ichimoku Cloud Analysis")
        log("✅ Market Microstructure Analysis")
        log("✅ Order Flow Analysis")
        log("✅ Smart Money Detection")
        log("✅ Institutional Activity Tracking")
        log("✅ Dark Pool Indicators")
        log("✅ Advanced Volatility Models (GARCH)")
        log("✅ Markov Regime Detection")
        log("✅ Sector Rotation System")
        log("✅ Advanced Portfolio Management")
        log("✅ Correlation Risk Management")
        log("✅ Ultra-Advanced Technical Indicators (50+ indicators)")
        log("✅ Candlestick Pattern Recognition")
        log("✅ Sentiment Analysis with FinBERT + VADER")
        log("✅ Meta Model Approval System")
        log("✅ Complete Profit Tracking")
        log("✅ Ultra-Advanced Risk Metrics (Sharpe, Sortino, Calmar, VaR, CVaR)")
        log("✅ Risk Management & Emergency Stops")
        log("✅ Advanced Backtesting Framework")
        log("✅ Google Sheets Integration")
        log("✅ Discord Alerts")
        log("✅ Market Regime Detection")
        log("✅ Liquidity Pool Detection")
        log("✅ ALL 2,800+ lines of functionality FULLY RESTORED!")
        
        # Start Flask server in background thread for Render health checks
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        log("✅ Health check server started")
        
        # Give Flask a moment to start
        time.sleep(2)
        
        # Start the ultra-advanced trading bot
        bot = UltraAdvancedTradingBot()
        bot.run_main_loop()
        
    except Exception as e:
        log(f"❌ Fatal startup error: {e}")
        send_discord_alert(f"❌ Fatal startup error: {e}", urgent=True)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    main()
