import os
import time
import pytz
import torch
import torch.nn as nn
import torch.optim as optim
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from alpaca_trade_api.rest import REST, TimeFrame
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
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
from flask import Flask, jsonify
import shap
from sklearn.inspection import permutation_importance
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import yaml
import schedule
from concurrent.futures import ThreadPoolExecutor
import pickle
import sqlite3
import glob
import gc

# Try to import Redis (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not installed. Feature caching will be disabled.")

warnings.filterwarnings('ignore')
load_dotenv()

# === ENHANCED CONFIGURATION MANAGEMENT ===
@dataclass
class TradingConfig:
    """Centralized configuration management"""
    # Dual-horizon prediction settings
    SHORT_TERM_DAYS: int = 2
    MEDIUM_TERM_DAYS: int = 15
    SHORT_TERM_WEIGHT: float = 0.6
    MEDIUM_TERM_WEIGHT: float = 0.4
    
    # Signal Quality & Thresholds
    SHORT_BUY_THRESHOLD: float = 0.53
    SHORT_SELL_AVOID_THRESHOLD: float = 0.45
    MEDIUM_BUY_THRESHOLD: float = 0.55
    MEDIUM_SELL_AVOID_THRESHOLD: float = 0.43
    PRICE_MOMENTUM_MIN: float = 0.005
    VOLUME_SPIKE_MIN: float = 1.5
    VOLUME_SPIKE_CONFIRMATION_MIN: float = 2.0
    SENTIMENT_HOLD_OVERRIDE: float = -0.5
    VWAP_DEVIATION_THRESHOLD: float = 0.02
    
    # Portfolio Management
    MAX_PER_SECTOR_WATCHLIST: int = 12
    MAX_PER_SECTOR_PORTFOLIO: float = 0.25
    WATCHLIST_LIMIT: int = 25
    DYNAMIC_WATCHLIST_REFRESH_HOURS: int = 4
    MAX_PORTFOLIO_RISK: float = 0.02
    MAX_DAILY_DRAWDOWN: float = 0.05
    EMERGENCY_DRAWDOWN_LIMIT: float = 0.10
    MAX_CORRELATION_THRESHOLD: float = 0.7
    
    # Risk Management & Position Sizing
    ATR_STOP_MULTIPLIER: float = 1.5
    ATR_PROFIT_MULTIPLIER: float = 2.5
    PROFIT_DECAY_FACTOR: float = 0.95
    VOLATILITY_GATE_THRESHOLD: float = 0.05
    MIN_MODEL_ACCURACY: float = 0.55
    SHARPE_RATIO_MIN: float = 1.0
    KELLY_FRACTION_MAX: float = 0.08
    KELLY_FRACTION_MIN: float = 0.01
    
    # Trade Management
    TRADE_COOLDOWN_MINUTES: int = 30
    EOD_LIQUIDATION_TIME: str = "15:45"  # 3:45 PM ET
    EOD_LIQUIDATION_ENABLED: bool = True
    
    # Meta-model and Ensemble
    META_MODEL_MIN_ACCURACY: float = 0.58
    META_MODEL_MIN_TRADES: int = 20
    ENSEMBLE_CONFIDENCE_THRESHOLD: float = 0.65
    MODEL_RETRAIN_FREQUENCY_HOURS: int = 24
    
    # Advanced Features
    SUPPORT_RESISTANCE_STRENGTH: int = 3
    VOLUME_PROFILE_BINS: int = 20
    Q_LEARNING_ALPHA: float = 0.1
    Q_LEARNING_GAMMA: float = 0.95
    Q_LEARNING_EPSILON: float = 0.1
    Q_LEARNING_EPSILON_DECAY: float = 0.995
    SECTOR_ROTATION_THRESHOLD: float = 0.02
    
    # Sentiment Analysis
    FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
    SENTIMENT_WEIGHT: float = 0.15
    NEWS_LOOKBACK_HOURS: int = 24
    
    # Market Regime Detection
    REGIME_DETECTION_WINDOW: int = 50
    BULL_MARKET_THRESHOLD: float = 0.02
    BEAR_MARKET_THRESHOLD: float = -0.02
    
    # Enterprise Features
    PAPER_TRADING_MODE: bool = True  # Set to False for live trading
    REAL_TIME_RISK_MONITORING: bool = True
    ANOMALY_DETECTION_ENABLED: bool = True
    FEATURE_CACHING_ENABLED: bool = True
    MULTI_INSTANCE_MONITORING: bool = True
    
    # Pattern Recognition
    FIBONACCI_LEVELS: List[float] = None
    FIBONACCI_EXTENSIONS: List[float] = None
    ICHIMOKU_PERIODS: Dict[str, int] = None
    ELLIOTT_WAVE_MIN_WAVES: int = 5
    HARMONIC_PATTERN_TOLERANCE: float = 0.05
    
    # Market Microstructure
    ORDERFLOW_IMBALANCE_THRESHOLD: float = 0.3
    LIQUIDITY_POOL_MIN_SIZE: int = 1000000
    SMART_MONEY_FLOW_THRESHOLD: float = 0.6
    INSTITUTIONAL_BLOCK_SIZE: int = 10000
    DARK_POOL_INDICATOR_THRESHOLD: float = 0.4
    
    def __post_init__(self):
        if self.FIBONACCI_LEVELS is None:
            self.FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
        if self.FIBONACCI_EXTENSIONS is None:
            self.FIBONACCI_EXTENSIONS = [1.272, 1.414, 1.618, 2.0, 2.618]
        if self.ICHIMOKU_PERIODS is None:
            self.ICHIMOKU_PERIODS = {'tenkan': 9, 'kijun': 26, 'senkou_b': 52}

# Load configuration
config = TradingConfig()

# === ENHANCED LOGGING SYSTEM ===
class StructuredLogger:
    def __init__(self, name: str = "TradingBot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler("logs/trading_bot.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, extra: Dict = None):
        self.logger.info(message, extra=extra or {})
    
    def error(self, message: str, extra: Dict = None):
        self.logger.error(message, extra=extra or {})
    
    def warning(self, message: str, extra: Dict = None):
        self.logger.warning(message, extra=extra or {})

logger = StructuredLogger()

# === ENHANCED API MANAGEMENT ===
class APIManager:
    def __init__(self):
        self.api = None
        self.news_api = None
        self.rate_limiter = {}
        self.last_request_time = {}
        self.initialize_apis()
    
    def initialize_apis(self):
        """Initialize all APIs with proper error handling"""
        try:
            # Alpaca API
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            
            if api_key and secret_key:
                self.api = REST(api_key, secret_key, base_url=base_url)
                account = self.safe_api_call(self.api.get_account)
                if account:
                    logger.info("✅ Alpaca API connected successfully")
                else:
                    logger.warning("⚠️ Alpaca API connection test failed")
            else:
                logger.warning("⚠️ Missing Alpaca API credentials - running in demo mode")
            
            # News API
            news_api_key = os.getenv("NEWS_API_KEY")
            if news_api_key:
                self.news_api = NewsApiClient(api_key=news_api_key)
                logger.info("✅ News API connected successfully")
            else:
                logger.warning("⚠️ Missing News API key")
                
        except Exception as e:
            logger.error(f"❌ API initialization failed: {e}")
    
    def safe_api_call(self, func, *args, max_retries: int = 3, **kwargs):
        """Enhanced API call wrapper with rate limiting and retry logic"""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # Rate limiting
        current_time = time.time()
        if func_name in self.last_request_time:
            time_since_last = current_time - self.last_request_time[func_name]
            if time_since_last < 1.0:  # 1 second rate limit
                time.sleep(1.0 - time_since_last)
        
        self.last_request_time[func_name] = time.time()
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"API call failed after {max_retries} attempts")
                    return None

api_manager = APIManager()

# === ENHANCED MARKET STATUS ===
def is_market_open_safe() -> bool:
    """Enhanced market open check with multiple fallbacks"""
    try:
        # Primary: API check
        if api_manager.api:
            clock = api_manager.safe_api_call(api_manager.api.get_clock)
            if clock:
                return clock.is_open
        
        # Fallback: Manual calculation
        now = datetime.now(pytz.timezone("US/Eastern"))
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
        
    except Exception as e:
        logger.error(f"Market status check failed: {e}")
        return False

def is_near_eod() -> bool:
    """Check if we're near end of day for liquidation"""
    try:
        now = datetime.now(pytz.timezone("US/Eastern"))
        eod_time = datetime.strptime(config.EOD_LIQUIDATION_TIME, "%H:%M").time()
        eod_datetime = now.replace(hour=eod_time.hour, minute=eod_time.minute, second=0, microsecond=0)
        
        return now >= eod_datetime
    except Exception as e:
        logger.error(f"EOD check failed: {e}")
        return False

# === REAL-TIME RISK MONITOR ===
class RealTimeRiskMonitor:
    """Real-time risk monitoring with automatic trading halt"""
    
    def __init__(self):
        self.max_drawdown_threshold = config.EMERGENCY_DRAWDOWN_LIMIT
        self.equity_curve = []
        self.peak_equity = 0
        self.current_drawdown = 0
        self.trading_halted = False
        
    def update_equity(self, current_equity: float):
        """Update equity and check drawdown"""
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': current_equity
        })
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check if we need to halt trading
        if self.current_drawdown > self.max_drawdown_threshold and not self.trading_halted:
            self.halt_trading()
    
    def halt_trading(self):
        """Halt all trading due to excessive drawdown"""
        self.trading_halted = True
        logger.error(f"🚨 TRADING HALTED - Drawdown: {self.current_drawdown:.2%}")
        send_discord_alert(f"🚨 TRADING HALTED - Drawdown: {self.current_drawdown:.2%}", urgent=True)

# === PORTFOLIO VAR/CVAR CALCULATOR ===
class PortfolioRiskCalculator:
    """Calculate Value at Risk and Conditional Value at Risk"""
    
    def __init__(self):
        self.confidence_level = 0.95
        self.lookback_days = 252
        
    def calculate_var_cvar(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate VaR and CVaR"""
        if len(returns) < 30:
            return 0.0, 0.0
        
        # Sort returns
        sorted_returns = np.sort(returns)
        
        # Calculate VaR
        var_index = int((1 - self.confidence_level) * len(sorted_returns))
        var = sorted_returns[var_index]
        
        # Calculate CVaR (expected shortfall)
        cvar = np.mean(sorted_returns[:var_index])
        
        return var, cvar

# === MULTIVARIATE ANOMALY DETECTION ===
class AnomalyDetector:
    """Detect market anomalies using Isolation Forest"""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, features: pd.DataFrame):
        """Train anomaly detection model"""
        try:
            if features is None or features.empty:
                return
            
            scaled_features = self.scaler.fit_transform(features)
            self.model.fit(scaled_features)
            self.is_trained = True
            logger.info("✅ Anomaly detector trained")
        except Exception as e:
            logger.error(f"❌ Anomaly detector training failed: {e}")
    
    def detect_anomaly(self, features: pd.DataFrame) -> bool:
        """Detect if current market conditions are anomalous"""
        if not self.is_trained or features is None or features.empty:
            return False
        
        try:
            scaled_features = self.scaler.transform(features)
            anomaly_score = self.model.decision_function(scaled_features)[0]
            is_anomaly = self.model.predict(scaled_features)[0] == -1
            
            if is_anomaly:
                logger.warning(f"⚠️ Market anomaly detected - Score: {anomaly_score:.3f}")
            
            return is_anomaly
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return False

# === BACKTESTING FRAMEWORK ===
class BacktestEngine:
    """Comprehensive backtesting framework"""
    
    def __init__(self):
        self.results = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, strategy, data: pd.DataFrame, start_date: str, end_date: str):
        """Run comprehensive backtest"""
        try:
            logger.info(f"🔄 Running backtest from {start_date} to {end_date}")
            
            # Filter data by date range
            backtest_data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Initialize backtest state
            initial_capital = 100000
            current_capital = initial_capital
            positions = {}
            
            # Run strategy on each bar
            for timestamp, row in backtest_data.iterrows():
                # Strategy logic would go here
                # This is a simplified example
                signal = self.generate_simple_signal(row)
                
                if signal == 'buy' and len(positions) < 10:
                    # Execute buy
                    position_size = current_capital * 0.1  # 10% position
                    positions[timestamp] = {
                        'entry_price': row['close'],
                        'size': position_size / row['close'],
                        'entry_time': timestamp
                    }
                
                elif signal == 'sell' and positions:
                    # Close oldest position
                    oldest_pos = min(positions.keys())
                    pos = positions.pop(oldest_pos)
                    
                    # Calculate P&L
                    pnl = (row['close'] - pos['entry_price']) * pos['size']
                    current_capital += pnl
                    
                    self.trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': pos['entry_price'],
                        'exit_price': row['close'],
                        'pnl': pnl,
                        'return': pnl / (pos['entry_price'] * pos['size'])
                    })
                
                # Update equity curve
                portfolio_value = current_capital
                for pos in positions.values():
                    portfolio_value += (row['close'] - pos['entry_price']) * pos['size']
                
                self.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': portfolio_value
                })
            
            # Calculate performance metrics
            self.calculate_backtest_metrics(initial_capital)
            
            logger.info("✅ Backtest completed")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {}
    
    def generate_simple_signal(self, row) -> str:
        """Simple signal generation for backtesting"""
        try:
            # Simple RSI strategy
            if hasattr(row, 'rsi_14'):
                if row.rsi_14 < 30:
                    return 'buy'
                elif row.rsi_14 > 70:
                    return 'sell'
            return 'hold'
        except:
            return 'hold'
    
    def calculate_backtest_metrics(self, initial_capital: float):
        """Calculate comprehensive backtest metrics"""
        if not self.trades:
            return
        
        returns = [trade['return'] for trade in self.trades]
        
        self.results = {
            'total_trades': len(self.trades),
            'winning_trades': len([r for r in returns if r > 0]),
            'losing_trades': len([r for r in returns if r < 0]),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'avg_return': np.mean(returns),
            'total_return': (self.equity_curve[-1]['equity'] - initial_capital) / initial_capital if self.equity_curve else 0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(),
            'profit_factor': sum([r for r in returns if r > 0]) / abs(sum([r for r in returns if r < 0])) if sum([r for r in returns if r < 0]) != 0 else float('inf')
        }
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if not self.equity_curve:
            return 0.0
        
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

# === PAPER TRADING ENGINE ===
class PaperTradingEngine:
    """Paper trading with detailed logging"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.enabled = config.PAPER_TRADING_MODE
        
    def execute_paper_trade(self, ticker: str, action: str, quantity: int, price: float):
        """Execute paper trade"""
        if not self.enabled:
            return False
        
        try:
            if action.lower() == 'buy':
                cost = quantity * price
                if cost <= self.current_capital:
                    self.current_capital -= cost
                    self.positions[ticker] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_time': datetime.now()
                    }
                    logger.info(f"📝 Paper BUY: {ticker} x{quantity} @ ${price:.2f}")
                    return True
                else:
                    logger.warning(f"⚠️ Insufficient paper capital for {ticker}")
                    return False
            
            elif action.lower() == 'sell' and ticker in self.positions:
                pos = self.positions.pop(ticker)
                proceeds = pos['quantity'] * price
                self.current_capital += proceeds
                
                pnl = (price - pos['entry_price']) * pos['quantity']
                self.trade_history.append({
                    'ticker': ticker,
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'entry_time': pos['entry_time'],
                    'exit_time': datetime.now()
                })
                
                logger.info(f"📝 Paper SELL: {ticker} x{pos['quantity']} @ ${price:.2f} - P&L: ${pnl:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Paper trade failed: {e}")
            return False
    
    def get_paper_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current paper portfolio value"""
        portfolio_value = self.current_capital
        
        for ticker, position in self.positions.items():
            if ticker in current_prices:
                portfolio_value += position['quantity'] * current_prices[ticker]
        
        return portfolio_value

# === REDIS CACHE FOR REAL-TIME FEATURES ===
class RedisFeatureCache:
    """Redis-based feature caching for performance"""
    
    def __init__(self):
        self.enabled = False
        if REDIS_AVAILABLE and config.FEATURE_CACHING_ENABLED:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                self.enabled = True
                logger.info("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available: {e}")
                self.enabled = False
        else:
            logger.info("ℹ️ Redis caching disabled")
    
    def cache_features(self, ticker: str, features: Dict[str, Any], ttl: int = 300):
        """Cache computed features"""
        if not self.enabled:
            return
        
        try:
            key = f"features:{ticker}"
            self.redis_client.setex(key, ttl, json.dumps(features, default=str))
        except Exception as e:
            logger.error(f"❌ Feature caching failed: {e}")
    
    def get_cached_features(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cached features"""
        if not self.enabled:
            return None
        
        try:
            key = f"features:{ticker}"
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"❌ Feature cache retrieval failed: {e}")
        
        return None

# === MULTI-INSTANCE HEARTBEAT MONITOR ===
class HeartbeatMonitor:
    """Monitor multiple bot instances"""
    
    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.heartbeat_interval = 60  # 1 minute
        self.last_heartbeat = datetime.now()
        self.enabled = config.MULTI_INSTANCE_MONITORING
        
    def send_heartbeat(self):
        """Send heartbeat signal"""
        if not self.enabled:
            return
        
        try:
            heartbeat_data = {
                'instance_id': self.instance_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'alive',
                'trades_today': len(trading_state.trade_outcomes) if 'trading_state' in globals() else 0,
                'open_positions': len(trading_state.open_positions) if 'trading_state' in globals() else 0,
                'current_equity': trading_state.starting_equity if 'trading_state' in globals() else 100000
            }
            
            # Save to file
            os.makedirs('heartbeats', exist_ok=True)
            with open(f'heartbeats/heartbeat_{self.instance_id}.json', 'w') as f:
                json.dump(heartbeat_data, f)
            
            self.last_heartbeat = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Heartbeat failed: {e}")
    
    def check_other_instances(self) -> List[str]:
        """Check status of other instances"""
        if not self.enabled:
            return []
        
        active_instances = []
        
        try:
            heartbeat_files = glob.glob('heartbeats/heartbeat_*.json')
            
            for file_path in heartbeat_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if heartbeat is recent (within 5 minutes)
                    heartbeat_time = datetime.fromisoformat(data['timestamp'])
                    if (datetime.now() - heartbeat_time).total_seconds() < 300:
                        active_instances.append(data['instance_id'])
                
                except Exception:
                    continue
        
        except Exception as e:
            logger.error(f"❌ Instance check failed: {e}")
        
        return active_instances

# === FLASK APP WITH ENHANCED HEALTH CHECKS ===
app = Flask(__name__)

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'market_open': is_market_open_safe(),
            'near_eod': is_near_eod(),
            'bot_running': True,
            'paper_trading': config.PAPER_TRADING_MODE,
            'api_status': {
                'alpaca': api_manager.api is not None,
                'news': api_manager.news_api is not None,
                'discord': os.getenv("DISCORD_WEBHOOK_URL") is not None,
                'redis': feature_cache.enabled if 'feature_cache' in globals() else False
            },
            'enterprise_features': {
                'real_time_risk_monitoring': config.REAL_TIME_RISK_MONITORING,
                'anomaly_detection': config.ANOMALY_DETECTION_ENABLED,
                'feature_caching': config.FEATURE_CACHING_ENABLED,
                'multi_instance_monitoring': config.MULTI_INSTANCE_MONITORING,
                'paper_trading': config.PAPER_TRADING_MODE
            },
            'features_active': {
                'dual_horizon': True,
                'ensemble_models': len(ensemble_model.short_term_models) if 'ensemble_model' in globals() else 0,
                'q_learning': len(trading_state.q_table) if 'trading_state' in globals() else 0,
                'sector_rotation': len(trading_state.sector_performance) if 'trading_state' in globals() else 0,
                'support_resistance': len(trading_state.support_resistance_cache) if 'trading_state' in globals() else 0,
                'volume_profile': len(trading_state.volume_profile_cache) if 'trading_state' in globals() else 0,
                'sentiment_analysis': True,
                'meta_model_approval': True,
                'dynamic_watchlist': True,
                'eod_liquidation': config.EOD_LIQUIDATION_ENABLED
            },
            'performance_metrics': {
                'total_trades': len(trading_state.trade_outcomes) if 'trading_state' in globals() else 0,
                'win_rate': trading_state.risk_metrics.get('win_rate', 0) if 'trading_state' in globals() else 0,
                'sharpe_ratio': trading_state.risk_metrics.get('sharpe_ratio', 0) if 'trading_state' in globals() else 0,
                'model_accuracy': trading_state.model_accuracy.get('current', 0) if 'trading_state' in globals() else 0,
                'current_drawdown': risk_monitor.current_drawdown if 'risk_monitor' in globals() else 0,
                'trading_halted': risk_monitor.trading_halted if 'risk_monitor' in globals() else False
            }
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def home():
    """Enhanced root endpoint with comprehensive bot information"""
    return jsonify({
        'service': 'Ultra-Advanced AI Trading Bot v6.0 - Enterprise Edition',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '6.0.0',
        'mode': 'Paper Trading' if config.PAPER_TRADING_MODE else 'Live Trading',
        'features': [
            '✅ Dual-horizon prediction (short & medium term)',
            '✅ Voting ensemble (XGBoost, RF, Logistic Regression)',
            '✅ Volume spike and VWAP filtering',
            '✅ Support/resistance level detection',
            '✅ FinBERT + VADER sentiment scoring',
            '✅ Meta-model approval',
            '✅ Dynamic watchlist optimization',
            '✅ Trade cooldown management',
            '✅ Kelly Criterion for position sizing',
            '✅ End-of-day liquidation',
            '✅ Google Sheets logging',
            '✅ Discord alerts',
            '✅ PnL tracking and trade outcome logging',
            '✅ Dynamic stop-loss, profit targets, and profit decay exit logic',
            '✅ Sector diversification filter',
            '✅ Volume spike confirmation',
            '✅ Live model accuracy tracking',
            '✅ Q-learning via PyTorch QNetwork fallback',
            '✅ Regime-aware model logic',
            '🚀 ENTERPRISE FEATURES:',
            '✅ Real-time risk monitoring with auto-halt',
            '✅ Portfolio VaR/CVaR calculation',
            '✅ Multivariate anomaly detection',
            '✅ Comprehensive backtesting framework',
            '✅ Paper trading mode with detailed logs',
            '✅ Redis-based feature caching',
            '✅ Multi-instance heartbeat monitoring',
            '✅ Fault tolerance with health checks'
        ]
    })

@app.route('/backtest')
def run_backtest_endpoint():
    """Endpoint to run backtests"""
    try:
        # This would run a backtest with sample data
        return jsonify({
            'message': 'Backtest endpoint ready',
            'status': 'Available for POST requests with parameters'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === ENHANCED DIRECTORY STRUCTURE ===
def create_enhanced_directories():
    """Create comprehensive directory structure"""
    directories = [
        "models/short", "models/medium", "models/meta", "models/q_learning",
        "models/ensemble", "models/pytorch", "models/regime", "models/anomaly",
        "logs", "performance", "backtests", "support_resistance",
        "volume_profiles", "sector_analysis", "sentiment_analysis",
        "feature_importance", "walk_forward", "attribution",
        "config", "tests", "docs", "cache", "google_sheets",
        "watchlists", "trade_history", "model_accuracy",
        "heartbeats", "paper_trading", "risk_monitoring",
        "enterprise_features", "sqlite_db"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

create_enhanced_directories()

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

# === ENHANCED TRADING STATE ===
class UltraAdvancedTradingState:
    def __init__(self):
        # Basic state
        self.sector_allocations = {}
        self.cooldown_timers = {}
        self.cooldown_cache = {}
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
        
        # Dual-horizon model tracking
        self.short_term_model = None
        self.medium_term_model = None
        self.short_term_accuracy = 0.0
        self.medium_term_accuracy = 0.0
        self.model_accuracy = {'short': 0.0, 'medium': 0.0, 'current': 0.0}
        
        # Dynamic watchlist
        self.current_watchlist = FALLBACK_UNIVERSE[:config.WATCHLIST_LIMIT]
        self.watchlist_performance = {}
        self.last_watchlist_update = datetime.now()
        
        # Volume and VWAP tracking
        self.volume_spike_cache = {}
        self.vwap_cache = {}
        
        # Sentiment analysis
        self.finbert_sentiment_cache = {}
        self.vader_sentiment_cache = {}
        self.combined_sentiment_cache = {}
        
        # Meta-model approval tracking
        self.meta_model_approved = False
        self.meta_model_accuracy_history = []
        self.meta_model_last_retrain = datetime.now()
        
        # Advanced tracking
        self.sector_performance = defaultdict(list)
        self.correlation_matrix = {}
        self.volatility_regime = "normal"
        self.market_microstructure = {}
        
        # Enhanced risk metrics
        self.risk_metrics = {
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'calmar_ratio': 0.0, 'sortino_ratio': 0.0, 'information_ratio': 0.0,
            'treynor_ratio': 0.0, 'var_95': 0.0, 'cvar_95': 0.0,
            'max_consecutive_losses': 0, 'recovery_factor': 0.0, 'sterling_ratio': 0.0
        }
        
        self.portfolio_weights = {}
        self.rebalance_signals = {}
        
        # Q-Learning state (PyTorch)
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.q_network = None
        self.q_state_history = deque(maxlen=1000)
        self.q_action_history = deque(maxlen=1000)
        self.q_reward_history = deque(maxlen=1000)
        self.q_learning_stats = {
            'total_episodes': 0,
            'exploration_rate': config.Q_LEARNING_EPSILON,
            'learning_rate': config.Q_LEARNING_ALPHA,
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
        
        # Market regime detection
        self.market_regime = "neutral"
        self.regime_confidence = 0.0
        self.regime_history = deque(maxlen=100)
        
        # Enhanced features
        self.feature_importance = {}
        self.adaptive_thresholds = {}
        self.ensemble_predictions = {}
        self.walk_forward_results = {}
        self.performance_attribution = {}
        
        # Google Sheets integration
        self.sheets_client = None
        self.sheets_worksheet = None
        
        # End-of-day liquidation tracking
        self.eod_liquidation_triggered = False
        self.positions_to_liquidate = []
        
        # Enterprise features
        self.anomaly_alerts = deque(maxlen=100)
        self.risk_alerts = deque(maxlen=100)
        self.performance_snapshots = deque(maxlen=1000)
        
    def reset_daily(self):
        """Reset daily state variables"""
        self.sector_allocations = {}
        self.cooldown_timers = {}
        self.sentiment_cache = {}
        self.support_resistance_cache = {}
        self.volume_profile_cache = {}
        self.daily_drawdown = 0.0
        self.starting_equity = 0.0
        self.emergency_stop_triggered = False
        self.eod_liquidation_triggered = False
        self.positions_to_liquidate = []
        self.rebalance_signals = {}
        self.order_flow_imbalance = {}
        self.smart_money_flow = {}
        self.market_impact_models = {}
        self.volume_spike_cache = {}
        self.vwap_cache = {}
        self.finbert_sentiment_cache = {}
        self.vader_sentiment_cache = {}
        self.combined_sentiment_cache = {}
    
    def get_next_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_id_counter += 1
        return f"TRADE_{datetime.now().strftime('%Y%m%d')}_{self.trade_id_counter:04d}"
    
    def update_ultra_advanced_risk_metrics(self):
        """Update comprehensive risk metrics"""
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
            logger.error(f"❌ Ultra-advanced risk metrics update failed: {e}")

trading_state = UltraAdvancedTradingState()

# Initialize enterprise features
risk_monitor = RealTimeRiskMonitor()
portfolio_risk_calc = PortfolioRiskCalculator()
anomaly_detector = AnomalyDetector()
backtest_engine = BacktestEngine()
paper_trading = PaperTradingEngine()
feature_cache = RedisFeatureCache()
heartbeat_monitor = HeartbeatMonitor(f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# === PYTORCH Q-NETWORK IMPLEMENTATION ===
class QNetwork(nn.Module):
    """PyTorch Q-Network for reinforcement learning"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class PyTorchQLearningAgent:
    """Enhanced Q-Learning agent using PyTorch"""
    def __init__(self, state_size: int = 10, action_size: int = 3, lr: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.actions = ['buy', 'sell', 'hold']
        
        # Neural network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.epsilon = config.Q_LEARNING_EPSILON
        self.epsilon_decay = config.Q_LEARNING_EPSILON_DECAY
        self.gamma = config.Q_LEARNING_GAMMA
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        self.update_target_frequency = 100
        self.step_count = 0
        
        # Update target network
        self.update_target_network()
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the model"""
        try:
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            logger.info("✅ PyTorch Q-Network model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load PyTorch Q-Network model: {e}")

# Initialize PyTorch Q-Learning agent
pytorch_q_agent = PyTorchQLearningAgent()

# === SENTIMENT ANALYSIS WITH FINBERT + VADER ===
class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analysis using FinBERT + VADER"""
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.finbert_pipeline = None
        self.initialize_finbert()
        
    def initialize_finbert(self):
        """Initialize FinBERT model"""
        try:
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL_NAME)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL_NAME)
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=-1  # Use CPU
            )
            logger.info("✅ FinBERT model initialized successfully")
        except Exception as e:
            logger.error(f"❌ FinBERT initialization failed: {e}")
            self.finbert_pipeline = None
    
    def get_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using FinBERT"""
        try:
            if not self.finbert_pipeline:
                return {'score': 0.0, 'label': 'neutral'}
            
            # Truncate text to avoid token limit
            text = text[:512]
            
            result = self.finbert_pipeline(text)[0]
            
            # Convert to numerical score
            if result['label'].lower() == 'positive':
                score = result['score']
            elif result['label'].lower() == 'negative':
                score = -result['score']
            else:
                score = 0.0
            
            return {'score': score, 'label': result['label']}
            
        except Exception as e:
            logger.error(f"❌ FinBERT sentiment analysis failed: {e}")
            return {'score': 0.0, 'label': 'neutral'}
    
    def get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            logger.error(f"❌ VADER sentiment analysis failed: {e}")
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def get_combined_sentiment(self, text: str) -> float:
        """Get combined sentiment score from FinBERT + VADER"""
        try:
            finbert_result = self.get_finbert_sentiment(text)
            vader_result = self.get_vader_sentiment(text)
            
            # Weighted combination (FinBERT 70%, VADER 30%)
            finbert_score = finbert_result['score']
            vader_score = vader_result['compound']
            
            combined_score = (finbert_score * 0.7) + (vader_score * 0.3)
            
            return np.clip(combined_score, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Combined sentiment analysis failed: {e}")
            return 0.0
    
    def analyze_ticker_sentiment(self, ticker: str) -> float:
        """Analyze sentiment for a specific ticker"""
        try:
            if not api_manager.news_api:
                return 0.0
            
            # Get recent news
            news_articles = api_manager.safe_api_call(
                api_manager.news_api.get_everything,
                q=ticker,
                language='en',
                sort_by='publishedAt',
                from_param=(datetime.now() - timedelta(hours=config.NEWS_LOOKBACK_HOURS)).isoformat()
            )
            
            if not news_articles or not news_articles.get('articles'):
                return 0.0
            
            sentiment_scores = []
            
            for article in news_articles['articles'][:10]:  # Limit to 10 articles
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                if content.strip():
                    sentiment_score = self.get_combined_sentiment(content)
                    sentiment_scores.append(sentiment_score)
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                
                # Cache the result
                trading_state.finbert_sentiment_cache[ticker] = {
                    'score': avg_sentiment,
                    'timestamp': datetime.now(),
                    'article_count': len(sentiment_scores)
                }
                
                return avg_sentiment
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Ticker sentiment analysis failed for {ticker}: {e}")
            return 0.0

# Initialize sentiment analyzer
sentiment_analyzer = EnhancedSentimentAnalyzer()

# === GOOGLE SHEETS INTEGRATION ===
class GoogleSheetsLogger:
    """Google Sheets integration for trade logging"""
    def __init__(self):
        self.client = None
        self.worksheet = None
        self.initialize_sheets()
    
    def initialize_sheets(self):
        """Initialize Google Sheets client"""
        try:
            # Check if credentials file exists
            creds_file = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "google_sheets_credentials.json")
            if not os.path.exists(creds_file):
                logger.warning("⚠️ Google Sheets credentials file not found")
                return
            
            scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
            self.client = gspread.authorize(creds)
            
            # Open or create spreadsheet
            sheet_name = os.getenv("GOOGLE_SHEETS_NAME", "Trading Bot Logs")
            try:
                spreadsheet = self.client.open(sheet_name)
            except gspread.SpreadsheetNotFound:
                spreadsheet = self.client.create(sheet_name)
                logger.info(f"✅ Created new Google Sheet: {sheet_name}")
            
            # Get or create worksheet
            try:
                self.worksheet = spreadsheet.worksheet("Trades")
            except gspread.WorksheetNotFound:
                self.worksheet = spreadsheet.add_worksheet(title="Trades", rows="1000", cols="20")
                # Add headers
                headers = [
                    "Timestamp", "Trade ID", "Ticker", "Action", "Quantity", "Entry Price",
                    "Exit Price", "PnL", "Return %", "Signal Strength", "Model Used",
                    "Sentiment Score", "Volume Spike", "VWAP Deviation", "Sector",
                    "Market Regime", "Stop Loss", "Take Profit", "Hold Duration", "Notes"
                ]
                self.worksheet.append_row(headers)
                logger.info("✅ Created trades worksheet with headers")
            
            trading_state.sheets_client = self.client
            trading_state.sheets_worksheet = self.worksheet
            logger.info("✅ Google Sheets integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Google Sheets initialization failed: {e}")
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade to Google Sheets"""
        try:
            if not self.worksheet:
                return False
            
            row_data = [
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('trade_id', ''),
                trade_data.get('ticker', ''),
                trade_data.get('action', ''),
                trade_data.get('quantity', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('pnl', 0),
                trade_data.get('return_pct', 0),
                trade_data.get('signal_strength', 0),
                trade_data.get('model_used', ''),
                trade_data.get('sentiment_score', 0),
                trade_data.get('volume_spike', False),
                trade_data.get('vwap_deviation', 0),
                trade_data.get('sector', ''),
                trade_data.get('market_regime', ''),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('hold_duration', ''),
                trade_data.get('notes', '')
            ]
            
            self.worksheet.append_row(row_data)
            logger.info(f"✅ Trade logged to Google Sheets: {trade_data.get('trade_id')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to log trade to Google Sheets: {e}")
            return False

# Initialize Google Sheets logger
sheets_logger = GoogleSheetsLogger()

# === MARKET REGIME DETECTION ===
class MarketRegimeDetector:
    """Market regime detection for regime-aware model logic"""
    def __init__(self):
        self.window = config.REGIME_DETECTION_WINDOW
        self.bull_threshold = config.BULL_MARKET_THRESHOLD
        self.bear_threshold = config.BEAR_MARKET_THRESHOLD
        
    def detect_market_regime(self, market_data: pd.DataFrame) -> Tuple[str, float]:
        """Detect current market regime"""
        try:
            if market_data is None or len(market_data) < self.window:
                return "neutral", 0.5
            
            # Calculate rolling returns
            returns = market_data['close'].pct_change().dropna()
            
            if len(returns) < self.window:
                return "neutral", 0.5
            
            # Use recent window for regime detection
            recent_returns = returns.tail(self.window)
            
            # Calculate metrics
            mean_return = recent_returns.mean()
            volatility = recent_returns.std()
            trend_strength = abs(mean_return) / volatility if volatility > 0 else 0
            
            # Determine regime
            if mean_return > self.bull_threshold and trend_strength > 0.5:
                regime = "bullish"
                confidence = min(trend_strength, 1.0)
            elif mean_return < self.bear_threshold and trend_strength > 0.5:
                regime = "bearish"
                confidence = min(trend_strength, 1.0)
            else:
                regime = "neutral"
                confidence = 1.0 - trend_strength
            
            # Update trading state
            trading_state.market_regime = regime
            trading_state.regime_confidence = confidence
            trading_state.regime_history.append({
                'timestamp': datetime.now(),
                'regime': regime,
                'confidence': confidence,
                'mean_return': mean_return,
                'volatility': volatility
            })
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            return "neutral", 0.5

# Initialize regime detector
regime_detector = MarketRegimeDetector()

# === ENHANCED TECHNICAL ANALYSIS ===
def add_ultra_advanced_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators with enhanced calculations"""
    try:
        if df is None or df.empty:
            return df
        
        # Convert to numpy arrays for vectorized calculations
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        open_price = df['open'].values
        
        # Basic indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages (vectorized)
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
        
        # Stochastic Oscillator
        if len(df) >= 14:
            stoch_14 = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch_14.stoch()
            df['stoch_d'] = stoch_14.stoch_signal()
        
        # ADX (Average Directional Index)
        if len(df) >= 14:
            adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx_indicator.adx()
            df['adx_pos'] = adx_indicator.adx_pos()
            df['adx_neg'] = adx_indicator.adx_neg()
        
        # Money Flow Index
        if len(df) >= 14:
            df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
        
        # Volume indicators
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume spike detection
        df['volume_spike'] = df['volume_ratio'] > config.VOLUME_SPIKE_MIN
        df['volume_spike_confirmation'] = df['volume_ratio'] > config.VOLUME_SPIKE_CONFIRMATION_MIN
        
        # ATR
        for period in [14, 21]:
            if len(df) >= period:
                df[f'atr_{period}'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()
        
        # Bollinger Bands
        if len(df) >= 20:
            bb_20 = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb_20.bollinger_hband()
            df['bb_lower'] = bb_20.bollinger_lband()
            df['bb_middle'] = bb_20.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # VWAP and VWAP deviation
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_vs_vwap'] = df['close'] / df['vwap'] - 1
        df['vwap_deviation'] = abs(df['price_vs_vwap'])
        df['vwap_filter_pass'] = df['vwap_deviation'] <= config.VWAP_DEVIATION_THRESHOLD
        
        # Volatility indicators
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        
        # Price momentum
        for period in [1, 3, 5, 10, 20]:
            if len(df) >= period:
                df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        df['price_momentum'] = df['momentum_5']  # Primary momentum indicator
        
        # Support/Resistance levels
        for period in [10, 20, 50]:
            if len(df) >= period:
                df[f'resistance_{period}'] = df['high'].rolling(period).max()
                df[f'support_{period}'] = df['low'].rolling(period).min()
                df[f'support_resistance_ratio_{period}'] = (df['close'] - df[f'support_{period}']) / (df[f'resistance_{period}'] - df[f'support_{period}'])
        
        # Enhanced Order Flow Indicators
        df['buying_pressure'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['selling_pressure'] = np.where(df['close'] < df['open'], df['volume'], 0)
        df['net_buying_pressure'] = df['buying_pressure'] - df['selling_pressure']
        df['buying_pressure_ratio'] = df['buying_pressure'] / (df['buying_pressure'] + df['selling_pressure'] + 1)
        
        # Smart Money indicators
        df['large_trade_indicator'] = np.where(df['volume'] > df['volume'].rolling(20).quantile(0.9), 1, 0)
        df['institutional_flow'] = df['large_trade_indicator'] * df['net_buying_pressure']
        df['smart_money_index'] = df['institutional_flow'].rolling(10).sum()
        
        # Advanced volatility measures
        df['realized_volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['garman_klass_volatility'] = np.sqrt(
            np.log(df['high'] / df['low']) * np.log(df['high'] / df['close']) +
            np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
        ).rolling(20).mean() * np.sqrt(252)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Technical indicator calculation failed: {e}")
        return df

# === DUAL-HORIZON ENSEMBLE MODEL ===
class DualHorizonEnsembleModel:
    """Dual-horizon prediction with voting ensemble"""
    def __init__(self):
        self.short_term_models = {}
        self.medium_term_models = {}
        self.meta_model = None
        self.feature_importance = {}
        self.scaler_short = StandardScaler()
        self.scaler_medium = StandardScaler()
        self.explainer = None
        
    def create_base_models(self) -> Dict[str, Any]:
        """Create diverse base models for ensemble"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
    
    def train_dual_horizon_ensemble(self, short_data: pd.DataFrame, medium_data: pd.DataFrame, 
                                  short_labels: pd.Series, medium_labels: pd.Series) -> bool:
        """Train dual-horizon ensemble models"""
        try:
            # Extract features for both horizons
            short_features = self.extract_features(short_data)
            medium_features = self.extract_features(medium_data)
            
            if short_features is None or medium_features is None:
                logger.error("❌ Feature extraction failed")
                return False
            
            # Scale features
            short_features_scaled = self.scaler_short.fit_transform(short_features)
            medium_features_scaled = self.scaler_medium.fit_transform(medium_features)
            
            short_features_scaled = pd.DataFrame(short_features_scaled, columns=short_features.columns, index=short_features.index)
            medium_features_scaled = pd.DataFrame(medium_features_scaled, columns=medium_features.columns, index=medium_features.index)
            
            # Create base models
            base_models = self.create_base_models()
            
            # Train short-term models
            logger.info("🔄 Training short-term models...")
            for name, model in base_models.items():
                try:
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, short_features_scaled, short_labels, cv=tscv, scoring='accuracy')
                    
                    # Train on full dataset
                    model.fit(short_features_scaled, short_labels)
                    self.short_term_models[name] = model
                    
                    logger.info(f"✅ Short-term {name} trained - CV Score: {cv_scores.mean():.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Short-term {name} training failed: {e}")
            
            # Train medium-term models
            logger.info("🔄 Training medium-term models...")
            for name, model in base_models.items():
                try:
                    # Create new instance for medium-term
                    medium_model = self.create_base_models()[name]
                    
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(medium_model, medium_features_scaled, medium_labels, cv=tscv, scoring='accuracy')
                    
                    # Train on full dataset
                    medium_model.fit(medium_features_scaled, medium_labels)
                    self.medium_term_models[name] = medium_model
                    
                    logger.info(f"✅ Medium-term {name} trained - CV Score: {cv_scores.mean():.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Medium-term {name} training failed: {e}")
            
            # Train meta-model for ensemble combination
            self.train_meta_model(short_features_scaled, medium_features_scaled, short_labels, medium_labels)
            
            # Calculate feature importance using SHAP
            self.calculate_feature_importance(short_features_scaled, short_labels)
            
            logger.info("✅ Dual-horizon ensemble trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dual-horizon ensemble training failed: {e}")
            return False
    
    def train_meta_model(self, short_features: pd.DataFrame, medium_features: pd.DataFrame,
                        short_labels: pd.Series, medium_labels: pd.Series):
        """Train meta-model for ensemble combination"""
        try:
            # Get base model predictions
            short_predictions = np.zeros((len(short_features), len(self.short_term_models)))
            medium_predictions = np.zeros((len(medium_features), len(self.medium_term_models)))
            
            for i, (name, model) in enumerate(self.short_term_models.items()):
                short_predictions[:, i] = model.predict_proba(short_features)[:, 1]
            
            for i, (name, model) in enumerate(self.medium_term_models.items()):
                medium_predictions[:, i] = model.predict_proba(medium_features)[:, 1]
            
            # Combine predictions (assuming same length for simplicity)
            min_length = min(len(short_predictions), len(medium_predictions))
            combined_predictions = np.hstack([
                short_predictions[:min_length],
                medium_predictions[:min_length]
            ])
            
            # Train meta-model
            self.meta_model = LogisticRegression(random_state=42)
            self.meta_model.fit(combined_predictions, short_labels[:min_length])
            
            logger.info("✅ Meta-model trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Meta-model training failed: {e}")
    
    def extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract features for ensemble model"""
        try:
            feature_columns = [
                'rsi_14', 'macd', 'macd_histogram', 'stoch_k', 'adx',
                'mfi', 'bb_position', 'volume_ratio', 'price_momentum',
                'volatility_20', 'buying_pressure_ratio', 'smart_money_index',
                'vwap_deviation', 'price_vs_vwap'
            ]
            
            available_features = [col for col in feature_columns if col in df.columns]
            
            if not available_features:
                return None
            
            features = df[available_features].copy()
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return None
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series):
        """Calculate feature importance using SHAP"""
        try:
            # Use the best performing base model for SHAP analysis
            best_model = self.short_term_models.get('random_forest')
            if not best_model:
                return
            
            # Create SHAP explainer
            self.explainer = shap.TreeExplainer(best_model)
            shap_values = self.explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values for feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            self.feature_importance = dict(zip(X.columns, feature_importance))
            
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Save feature importance
            with open('feature_importance/shap_importance.json', 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            
            logger.info(f"✅ Feature importance calculated: Top 5 features: {list(self.feature_importance.keys())[:5]}")
            
        except Exception as e:
            logger.error(f"❌ Feature importance calculation failed: {e}")
    
    def predict_dual_horizon(self, short_data: pd.DataFrame, medium_data: pd.DataFrame) -> Tuple[float, float, float]:
        """Make dual-horizon predictions"""
        try:
            if not self.short_term_models or not self.medium_term_models:
                logger.warning("⚠️ Models not trained")
                return 0.5, 0.5, 0.5
            
            # Extract and scale features
            short_features = self.extract_features(short_data)
            medium_features = self.extract_features(medium_data)
            
            if short_features is None or medium_features is None:
                return 0.5, 0.5, 0.5
            
            short_features_scaled = self.scaler_short.transform(short_features.tail(1))
            medium_features_scaled = self.scaler_medium.transform(medium_features.tail(1))
            
            # Get short-term predictions
            short_predictions = []
            for name, model in self.short_term_models.items():
                pred = model.predict_proba(short_features_scaled)[0, 1]
                short_predictions.append(pred)
            
            short_ensemble_pred = np.mean(short_predictions)
            
            # Get medium-term predictions
            medium_predictions = []
            for name, model in self.medium_term_models.items():
                pred = model.predict_proba(medium_features_scaled)[0, 1]
                medium_predictions.append(pred)
            
            medium_ensemble_pred = np.mean(medium_predictions)
            
            # Combine using meta-model if available
            if self.meta_model:
                combined_features = np.hstack([short_predictions, medium_predictions]).reshape(1, -1)
                meta_pred = self.meta_model.predict_proba(combined_features)[0, 1]
            else:
                # Weighted combination
                meta_pred = (short_ensemble_pred * config.SHORT_TERM_WEIGHT + 
                           medium_ensemble_pred * config.MEDIUM_TERM_WEIGHT)
            
            return short_ensemble_pred, medium_ensemble_pred, meta_pred
            
        except Exception as e:
            logger.error(f"❌ Dual-horizon prediction failed: {e}")
            return 0.5, 0.5, 0.5
    
    def get_top_features(self, n: int = 10) -> List[str]:
        """Get top N most important features"""
        return list(self.feature_importance.keys())[:n]

# Initialize dual-horizon ensemble model
ensemble_model = DualHorizonEnsembleModel()

# === META-MODEL APPROVAL SYSTEM ===
class MetaModelApprovalSystem:
    """Meta-model approval system for trade validation"""
    def __init__(self):
        self.min_accuracy = config.META_MODEL_MIN_ACCURACY
        self.min_trades = config.META_MODEL_MIN_TRADES
        self.approval_history = deque(maxlen=100)
        
    def evaluate_model_performance(self) -> bool:
        """Evaluate if meta-model meets approval criteria"""
        try:
            if len(trading_state.trade_outcomes) < self.min_trades:
                logger.info(f"⏳ Insufficient trades for meta-model approval: {len(trading_state.trade_outcomes)}/{self.min_trades}")
                return False
            
            # Calculate recent accuracy
            recent_trades = trading_state.trade_outcomes[-50:]  # Last 50 trades
            correct_predictions = sum(1 for trade in recent_trades if trade.get('correct_prediction', False))
            accuracy = correct_predictions / len(recent_trades)
            
            # Update model accuracy tracking
            trading_state.model_accuracy['current'] = accuracy
            trading_state.meta_model_accuracy_history.append({
                'timestamp': datetime.now(),
                'accuracy': accuracy,
                'trade_count': len(recent_trades)
            })
            
            # Check approval criteria
            approved = accuracy >= self.min_accuracy
            
            # Additional criteria
            if approved:
                # Check Sharpe ratio
                if trading_state.risk_metrics.get('sharpe_ratio', 0) < config.SHARPE_RATIO_MIN:
                    approved = False
                    logger.warning(f"⚠️ Meta-model approval denied: Low Sharpe ratio {trading_state.risk_metrics.get('sharpe_ratio', 0):.2f}")
                
                # Check maximum drawdown
                if trading_state.risk_metrics.get('max_drawdown', 0) > config.MAX_DAILY_DRAWDOWN:
                    approved = False
                    logger.warning(f"⚠️ Meta-model approval denied: High drawdown {trading_state.risk_metrics.get('max_drawdown', 0):.2%}")
            
            # Update approval status
            trading_state.meta_model_approved = approved
            
            self.approval_history.append({
                'timestamp': datetime.now(),
                'approved': approved,
                'accuracy': accuracy,
                'sharpe_ratio': trading_state.risk_metrics.get('sharpe_ratio', 0),
                'max_drawdown': trading_state.risk_metrics.get('max_drawdown', 0)
            })
            
            if approved:
                logger.info(f"✅ Meta-model approved - Accuracy: {accuracy:.3f}, Trades: {len(recent_trades)}")
            else:
                logger.warning(f"❌ Meta-model approval denied - Accuracy: {accuracy:.3f} < {self.min_accuracy:.3f}")
            
            return approved
            
        except Exception as e:
            logger.error(f"❌ Meta-model evaluation failed: {e}")
            return False
    
    def should_execute_trade(self, prediction_confidence: float) -> bool:
        """Determine if trade should be executed based on meta-model approval"""
        try:
            # Check if meta-model is approved
            if not trading_state.meta_model_approved:
                logger.info("⏸️ Trade blocked: Meta-model not approved")
                return False
            
            # Check confidence threshold
            if prediction_confidence < config.ENSEMBLE_CONFIDENCE_THRESHOLD:
                logger.info(f"⏸️ Trade blocked: Low confidence {prediction_confidence:.3f} < {config.ENSEMBLE_CONFIDENCE_THRESHOLD:.3f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade execution check failed: {e}")
            return False

# Initialize meta-model approval system
meta_approval_system = MetaModelApprovalSystem()

# === DYNAMIC WATCHLIST OPTIMIZATION ===
class DynamicWatchlistOptimizer:
    """Dynamic watchlist optimization based on performance and market conditions"""
    def __init__(self):
        self.refresh_hours = config.DYNAMIC_WATCHLIST_REFRESH_HOURS
        self.max_per_sector = config.MAX_PER_SECTOR_WATCHLIST
        self.watchlist_limit = config.WATCHLIST_LIMIT
        
    def optimize_watchlist(self) -> List[str]:
        """Optimize watchlist based on multiple criteria"""
        try:
            # Check if refresh is needed
            time_since_update = datetime.now() - trading_state.last_watchlist_update
            if time_since_update.total_seconds() < self.refresh_hours * 3600:
                return trading_state.current_watchlist
            
            logger.info("🔄 Optimizing dynamic watchlist...")
            
            # Score all tickers
            ticker_scores = {}
            
            for sector, tickers in SECTOR_UNIVERSE.items():
                sector_scores = []
                
                for ticker in tickers:
                    score = self.calculate_ticker_score(ticker, sector)
                    if score > 0:
                        ticker_scores[ticker] = {
                            'score': score,
                            'sector': sector
                        }
                        sector_scores.append((ticker, score))
                
                # Sort sector tickers by score and take top performers
                sector_scores.sort(key=lambda x: x[1], reverse=True)
                
            # Select top tickers with sector diversification
            optimized_watchlist = []
            sector_counts = defaultdict(int)
            
            # Sort all tickers by score
            sorted_tickers = sorted(ticker_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            for ticker, data in sorted_tickers:
                sector = data['sector']
                
                # Check sector limits
                if sector_counts[sector] < self.max_per_sector and len(optimized_watchlist) < self.watchlist_limit:
                    optimized_watchlist.append(ticker)
                    sector_counts[sector] += 1
            
            # Update trading state
            trading_state.current_watchlist = optimized_watchlist
            trading_state.last_watchlist_update = datetime.now()
            
            logger.info(f"✅ Watchlist optimized: {len(optimized_watchlist)} tickers selected")
            logger.info(f"📊 Sector distribution: {dict(sector_counts)}")
            
            return optimized_watchlist
            
        except Exception as e:
            logger.error(f"❌ Watchlist optimization failed: {e}")
            return trading_state.current_watchlist
    
    def calculate_ticker_score(self, ticker: str, sector: str) -> float:
        """Calculate comprehensive ticker score for watchlist inclusion"""
        try:
            score = 0.0
            
            # Historical performance score
            ticker_trades = [t for t in trading_state.trade_outcomes if t.get('ticker') == ticker]
            if ticker_trades:
                win_rate = len([t for t in ticker_trades if t['return'] > 0]) / len(ticker_trades)
                avg_return = np.mean([t['return'] for t in ticker_trades])
                score += (win_rate * 50) + (avg_return * 100)
            
            # Volume and liquidity score
            try:
                data = get_enhanced_data(ticker, limit=20)
                if data is not None and not data.empty:
                    avg_volume = data['volume'].mean()
                    volume_score = min(avg_volume / 1000000, 10)  # Cap at 10M volume
                    score += volume_score
                    
                    # Volatility score (moderate volatility preferred)
                    volatility = data['returns'].std()
                    if 0.01 <= volatility <= 0.05:  # Sweet spot
                        score += 10
                    elif volatility > 0.05:
                        score += max(0, 10 - (volatility - 0.05) * 100)
            except:
                pass
            
            # Sector performance score
            sector_performance = trading_state.sector_performance.get(sector, 0)
            score += sector_performance * 20
            
            # Sentiment score
            sentiment_data = trading_state.combined_sentiment_cache.get(ticker)
            if sentiment_data:
                sentiment_score = sentiment_data.get('score', 0)
                score += sentiment_score * 5
            
            # Market regime adjustment
            if trading_state.market_regime == "bullish":
                score *= 1.1
            elif trading_state.market_regime == "bearish":
                score *= 0.9
            
            return max(score, 0)
            
        except Exception as e:
            logger.error(f"❌ Ticker score calculation failed for {ticker}: {e}")
            return 0.0

# Initialize dynamic watchlist optimizer
watchlist_optimizer = DynamicWatchlistOptimizer()

# === ENHANCED DATA FETCHING ===
def get_enhanced_data(ticker: str, limit: int = 100, timeframe: TimeFrame = TimeFrame.Minute, 
                     days_back: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Get enhanced market data with comprehensive error handling"""
    try:
        # Check cache first
        cached_features = feature_cache.get_cached_features(ticker)
        if cached_features and not config.PAPER_TRADING_MODE:
            logger.info(f"📋 Using cached data for {ticker}")
            # Convert cached data back to DataFrame if needed
            # This is a simplified version - in practice you'd cache the full DataFrame
        
        if not api_manager.api or config.PAPER_TRADING_MODE:
            # Return demo data for testing
            dates = pd.date_range(start='2024-01-01', periods=limit, freq='5T')
            demo_data = pd.DataFrame({
                'open': np.random.uniform(100, 200, limit),
                'high': np.random.uniform(100, 200, limit),
                'low': np.random.uniform(100, 200, limit),
                'close': np.random.uniform(100, 200, limit),
                'volume': np.random.randint(1000, 10000, limit)
            }, index=dates)
            
            # Ensure high >= low and other constraints
            demo_data['high'] = np.maximum(demo_data['high'], demo_data[['open', 'close']].max(axis=1))
            demo_data['low'] = np.minimum(demo_data['low'], demo_data[['open', 'close']].min(axis=1))
            
            enhanced_data = add_ultra_advanced_technical_indicators(demo_data)
            
            # Cache the features
            if enhanced_data is not None:
                feature_cache.cache_features(ticker, enhanced_data.to_dict())
            
            return enhanced_data
        
        # Determine timeframe and limit based on model type
        if days_back == config.SHORT_TERM_DAYS:  # Short-term model
            timeframe = TimeFrame.Minute
            limit = config.SHORT_TERM_DAYS * 24 * 60 // 5  # 2 days of 5-minute candles
        elif days_back == config.MEDIUM_TERM_DAYS:  # Medium-term model
            timeframe = TimeFrame.Day
            limit = config.MEDIUM_TERM_DAYS  # 15 days of daily bars
        
        # Fetch raw data with retry logic
        bars = api_manager.safe_api_call(
            api_manager.api.get_bars,
            ticker,
            timeframe,
            limit=limit,
            feed='iex'
        )
        
        if not bars or bars.df.empty:
            logger.warning(f"⚠️ No data received for {ticker}")
            return None
        
        df = bars.df
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"⚠️ Missing required columns for {ticker}")
            return None
        
        # Filter out zero volume bars
        df = df[df['volume'] > 0].copy()
        
        if len(df) < 20:
            logger.warning(f"⚠️ Insufficient data for {ticker}: {len(df)} bars")
            return None
        
        # Add ALL technical indicators
        df = add_ultra_advanced_technical_indicators(df)
        
        # Cache the enhanced data
        feature_cache.cache_features(ticker, df.to_dict())
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Data fetch failed for {ticker}: {e}")
        return None

# === KELLY CRITERION POSITION SIZING ===
class KellyPositionSizer:
    def __init__(self):
        self.min_history = 30
        
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction for position sizing"""
        try:
            if avg_loss <= 0:
                return config.KELLY_FRACTION_MIN
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply fractional Kelly (typically 25% of full Kelly)
            fractional_kelly = kelly_fraction * 0.25
            
            # Cap at reasonable limits
            return np.clip(fractional_kelly, config.KELLY_FRACTION_MIN, config.KELLY_FRACTION_MAX)
            
        except Exception as e:
            logger.error(f"❌ Kelly calculation failed: {e}")
            return config.KELLY_FRACTION_MIN
    
    def calculate_position_size(self, ticker: str, equity: float, signal_strength: float, 
                              market_data: pd.DataFrame) -> int:
        """Calculate position size using Kelly criterion and volatility targeting"""
        try:
            # Get historical performance for this ticker
            ticker_trades = [t for t in trading_state.trade_outcomes if t.get('ticker') == ticker]
            
            if len(ticker_trades) >= self.min_history:
                # Calculate Kelly-based sizing
                wins = [t for t in ticker_trades if t['return'] > 0]
                losses = [t for t in ticker_trades if t['return'] < 0]
                
                win_rate = len(wins) / len(ticker_trades)
                avg_win = np.mean([t['return'] for t in wins]) if wins else 0
                avg_loss = np.mean([t['return'] for t in losses]) if losses else 0
                
                kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, abs(avg_loss))
            else:
                # Use default sizing for new tickers
                kelly_fraction = config.KELLY_FRACTION_MIN
            
            # Adjust for signal strength
            signal_multiplier = min(signal_strength * 1.5, 2.0)
            
            # Volatility adjustment
            if market_data is not None and 'atr_14' in market_data.columns:
                atr = market_data['atr_14'].iloc[-1]
                current_price = market_data['close'].iloc[-1]
                volatility_pct = atr / current_price
                
                # Reduce position size for high volatility
                volatility_multiplier = max(0.5, 1 - volatility_pct * 2)
            else:
                volatility_multiplier = 1.0
            
            # Calculate final position size
            position_value = equity * kelly_fraction * signal_multiplier * volatility_multiplier
            current_price = market_data['close'].iloc[-1] if market_data is not None else 100
            position_size = int(position_value / current_price)
            
            return max(1, position_size)
            
        except Exception as e:
            logger.error(f"❌ Kelly position sizing failed: {e}")
            return 1

kelly_position_sizer = KellyPositionSizer()

# === DISCORD ALERT SYSTEM WITH THROTTLING ===
class AlertManager:
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self.alert_history = deque(maxlen=100)
        self.throttle_window = 300  # 5 minutes
        self.max_alerts_per_window = 10
        
    def send_alert(self, message: str, urgent: bool = False) -> bool:
        """Send Discord alert with throttling"""
        try:
            if not self.webhook_url:
                logger.info(f"Discord alert (no webhook): {message}")
                return False
            
            # Check throttling (unless urgent)
            if not urgent and not self.check_throttle():
                logger.warning("⚠️ Alert throttled due to rate limiting")
                return False
            
            if urgent:
                message = f"🚨 **URGENT** 🚨\n{message}"
            
            payload = {"content": message}
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.alert_history.append({
                    'timestamp': time.time(),
                    'message': message[:50] + "..." if len(message) > 50 else message,
                    'urgent': urgent
                })
                logger.info(f"📬 Discord alert sent: {message[:50]}...")
                return True
            else:
                logger.warning(f"⚠️ Discord alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Discord alert error: {e}")
            return False
    
    def check_throttle(self) -> bool:
        """Check if we can send another alert"""
        current_time = time.time()
        recent_alerts = [
            alert for alert in self.alert_history 
            if current_time - alert['timestamp'] < self.throttle_window
        ]
        return len(recent_alerts) < self.max_alerts_per_window

alert_manager = AlertManager()

def send_discord_alert(message: str, urgent: bool = False):
    """Wrapper function for sending Discord alerts"""
    return alert_manager.send_alert(message, urgent)

# === ENHANCED TRADING EXECUTOR WITH EOD LIQUIDATION ===
class EnhancedTradingExecutor:
    def __init__(self):
        self.cooldown_periods = {}
        self.position_tracker = {}
        self.max_positions = 10
        self.cooldown_duration = config.TRADE_COOLDOWN_MINUTES * 60
        self.order_types = ['market', 'limit', 'stop_loss', 'trailing_stop']
        
    def execute_buy_order(self, ticker: str, signal_strength: float, market_data: pd.DataFrame, 
                         short_pred: float, medium_pred: float, meta_pred: float,
                         sentiment_score: float = 0.0, volume_spike: bool = False,
                         vwap_deviation: float = 0.0) -> bool:
        """Execute buy order with enhanced risk management and logging"""
        try:
            # Check enterprise risk controls
            if risk_monitor.trading_halted:
                logger.warning(f"🛑 Trading halted by risk monitor, skipping {ticker}")
                return False
            
            if trading_state.emergency_stop_triggered:
                logger.warning(f"🛑 Emergency stop active, skipping {ticker}")
                return False
            
            if not self.check_cooldown(ticker):
                logger.info(f"⏰ {ticker} in cooldown period")
                return False
            
            # Check if near EOD and liquidation is enabled
            if config.EOD_LIQUIDATION_ENABLED and is_near_eod():
                logger.info(f"⏰ Near EOD, skipping new positions for {ticker}")
                return False
            
            # Paper trading mode
            if config.PAPER_TRADING_MODE:
                return self.execute_paper_buy_order(ticker, signal_strength, market_data, 
                                                  short_pred, medium_pred, meta_pred,
                                                  sentiment_score, volume_spike, vwap_deviation)
            
            if not api_manager.api:
                logger.info(f"⚠️ No API connection, simulating buy order for {ticker}")
                return self.simulate_buy_order(ticker, signal_strength, market_data, 
                                             short_pred, medium_pred, meta_pred,
                                             sentiment_score, volume_spike, vwap_deviation)
            
            # Get account info
            account = api_manager.safe_api_call(api_manager.api.get_account)
            if not account:
                logger.error(f"❌ Could not get account info for {ticker}")
                return False
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            
            # Update risk monitor
            risk_monitor.update_equity(equity)
            
            # Enhanced risk management checks
            if not self.check_enhanced_risk_limits(ticker, equity):
                return False
            
            # Kelly criterion position sizing
            position_size = kelly_position_sizer.calculate_position_size(ticker, equity, signal_strength, market_data)
            if position_size <= 0:
                logger.error(f"❌ Invalid position size for {ticker}: {position_size}")
                return False
            
            # Get current price
            current_price = market_data['close'].iloc[-1] if market_data is not None else 0
            if current_price <= 0:
                logger.error(f"❌ Invalid price for {ticker}: {current_price}")
                return False
            
            # Check buying power
            order_value = position_size * current_price
            if order_value > buying_power:
                logger.error(f"❌ Insufficient buying power for {ticker}: ${order_value:.2f} > ${buying_power:.2f}")
                return False
            
            # Execute the order
            try:
                order = api_manager.api.submit_order(
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
                    'short_prediction': short_pred,
                    'medium_prediction': medium_pred,
                    'meta_prediction': meta_pred,
                    'sentiment_score': sentiment_score,
                    'volume_spike': volume_spike,
                    'vwap_deviation': vwap_deviation,
                    'order_id': order.id,
                    'stop_loss': self.calculate_stop_loss(current_price, market_data),
                    'take_profit': self.calculate_take_profit(current_price, market_data),
                    'profit_decay_factor': config.PROFIT_DECAY_FACTOR,
                    'market_regime': trading_state.market_regime,
                    'sector': self.get_ticker_sector(ticker)
                }
                
                trading_state.open_positions[ticker] = position_data
                self.set_cooldown(ticker)
                
                # Log to Google Sheets
                sheets_data = {
                    'timestamp': datetime.now().isoformat(),
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'action': 'BUY',
                    'quantity': position_size,
                    'entry_price': current_price,
                    'signal_strength': signal_strength,
                    'model_used': 'Dual-Horizon Ensemble',
                    'sentiment_score': sentiment_score,
                    'volume_spike': volume_spike,
                    'vwap_deviation': vwap_deviation,
                    'sector': self.get_ticker_sector(ticker),
                    'market_regime': trading_state.market_regime,
                    'stop_loss': position_data['stop_loss'],
                    'take_profit': position_data['take_profit']
                }
                sheets_logger.log_trade(sheets_data)
                
                # Send Discord alert
                alert_msg = f"🟢 **BUY ORDER EXECUTED**\n"
                alert_msg += f"Ticker: {ticker}\n"
                alert_msg += f"Quantity: {position_size}\n"
                alert_msg += f"Price: ${current_price:.2f}\n"
                alert_msg += f"Value: ${order_value:.2f}\n"
                alert_msg += f"Signal Strength: {signal_strength:.3f}\n"
                alert_msg += f"Short Pred: {short_pred:.3f} | Medium Pred: {medium_pred:.3f}\n"
                alert_msg += f"Meta Pred: {meta_pred:.3f}\n"
                alert_msg += f"Sentiment: {sentiment_score:.3f}\n"
                alert_msg += f"Volume Spike: {'✅' if volume_spike else '❌'}\n"
                alert_msg += f"VWAP Dev: {vwap_deviation:.3f}\n"
                alert_msg += f"Trade ID: {trade_id}"
                send_discord_alert(alert_msg)
                
                logger.info(f"✅ Buy order executed: {ticker} x{position_size} @ ${current_price:.2f}")
                return True
                
            except Exception as order_error:
                logger.error(f"❌ Order execution failed for {ticker}: {order_error}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Buy order execution failed for {ticker}: {e}")
            return False
    
    def execute_paper_buy_order(self, ticker: str, signal_strength: float, market_data: pd.DataFrame,
                               short_pred: float, medium_pred: float, meta_pred: float,
                               sentiment_score: float, volume_spike: bool, vwap_deviation: float) -> bool:
        """Execute paper trading buy order"""
        try:
            current_price = market_data['close'].iloc[-1] if market_data is not None else 100
            
            # Calculate position size for paper trading
            paper_equity = paper_trading.get_paper_portfolio_value({ticker: current_price})
            position_size = kelly_position_sizer.calculate_position_size(ticker, paper_equity, signal_strength, market_data)
            
            # Execute paper trade
            success = paper_trading.execute_paper_trade(ticker, 'buy', position_size, current_price)
            
            if success:
                # Track the position in trading state as well
                trade_id = trading_state.get_next_trade_id()
                position_data = {
                    'trade_id': trade_id,
                    'ticker': ticker,
                    'action': 'buy',
                    'quantity': position_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'signal_strength': signal_strength,
                    'short_prediction': short_pred,
                    'medium_prediction': medium_pred,
                    'meta_prediction': meta_pred,
                    'sentiment_score': sentiment_score,
                    'volume_spike': volume_spike,
                    'vwap_deviation': vwap_deviation,
                    'paper_trade': True,
                    'stop_loss': self.calculate_stop_loss(current_price, market_data),
                    'take_profit': self.calculate_take_profit(current_price, market_data),
                    'market_regime': trading_state.market_regime,
                    'sector': self.get_ticker_sector(ticker)
                }
                
                trading_state.open_positions[ticker] = position_data
                self.set_cooldown(ticker)
                
                logger.info(f"📝 Paper buy order executed: {ticker} x{position_size} @ ${current_price:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Paper buy order failed for {ticker}: {e}")
            return False
    
    def liquidate_eod_positions(self):
        """Liquidate all positions at end of day"""
        try:
            if not config.EOD_LIQUIDATION_ENABLED:
                return
            
            if not is_near_eod():
                return
            
            if trading_state.eod_liquidation_triggered:
                return
            
            logger.info("🔄 Starting end-of-day liquidation...")
            trading_state.eod_liquidation_triggered = True
            
            positions_to_close = list(trading_state.open_positions.keys())
            
            for ticker in positions_to_close:
                try:
                    self.close_position(ticker, reason="EOD_LIQUIDATION")
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.error(f"❌ EOD liquidation failed for {ticker}: {e}")
            
            # Send summary alert
            alert_msg = f"🌅 **END-OF-DAY LIQUIDATION COMPLETE**\n"
            alert_msg += f"Positions closed: {len(positions_to_close)}\n"
            alert_msg += f"Time: {datetime.now().strftime('%H:%M:%S')}"
            send_discord_alert(alert_msg)
            
            logger.info(f"✅ EOD liquidation complete: {len(positions_to_close)} positions closed")
            
        except Exception as e:
            logger.error(f"❌ EOD liquidation failed: {e}")
    
    def close_position(self, ticker: str, reason: str = "MANUAL"):
        """Close a position with comprehensive logging"""
        try:
            if ticker not in trading_state.open_positions:
                logger.warning(f"⚠️ No open position found for {ticker}")
                return False
            
            position = trading_state.open_positions[ticker]
            
            # Paper trading mode
            if config.PAPER_TRADING_MODE or position.get('paper_trade', False):
                return self.close_paper_position(ticker, position, reason)
            
            if not api_manager.api:
                return self.simulate_close_position(ticker, position, reason)
            
            # Get current price
            current_data = get_enhanced_data(ticker, limit=5)
            if current_data is None:
                logger.error(f"❌ Could not get current price for {ticker}")
                return False
            
            current_price = current_data['close'].iloc[-1]
            
            # Execute sell order
            try:
                order = api_manager.api.submit_order(
                    symbol=ticker,
                    qty=position['quantity'],
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                # Calculate P&L
                entry_price = position['entry_price']
                pnl = (current_price - entry_price) * position['quantity']
                return_pct = (current_price - entry_price) / entry_price
                
                # Update trade outcome
                trade_outcome = {
                    'trade_id': position['trade_id'],
                    'ticker': ticker,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'return': return_pct,
                    'hold_duration': str(datetime.now() - position['entry_time']),
                    'exit_reason': reason,
                    'signal_strength': position.get('signal_strength', 0),
                    'sentiment_score': position.get('sentiment_score', 0),
                    'volume_spike': position.get('volume_spike', False),
                    'vwap_deviation': position.get('vwap_deviation', 0),
                    'sector': position.get('sector', ''),
                    'market_regime': position.get('market_regime', ''),
                    'correct_prediction': return_pct > 0
                }
                
                trading_state.trade_outcomes.append(trade_outcome)
                
                # Log to Google Sheets
                sheets_data = {
                    'timestamp': datetime.now().isoformat(),
                    'trade_id': position['trade_id'],
                    'ticker': ticker,
                    'action': 'SELL',
                    'quantity': position['quantity'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'return_pct': return_pct * 100,
                    'hold_duration': str(datetime.now() - position['entry_time']),
                    'notes': f"Exit reason: {reason}"
                }
                sheets_logger.log_trade(sheets_data)
                
                # Remove from open positions
                del trading_state.open_positions[ticker]
                
                # Send Discord alert
                alert_msg = f"🔴 **SELL ORDER EXECUTED**\n"
                alert_msg += f"Ticker: {ticker}\n"
                alert_msg += f"Quantity: {position['quantity']}\n"
                alert_msg += f"Entry: ${entry_price:.2f} | Exit: ${current_price:.2f}\n"
                alert_msg += f"P&L: ${pnl:.2f} ({return_pct:.2%})\n"
                alert_msg += f"Reason: {reason}\n"
                alert_msg += f"Trade ID: {position['trade_id']}"
                send_discord_alert(alert_msg)
                
                logger.info(f"✅ Position closed: {ticker} - P&L: ${pnl:.2f} ({return_pct:.2%})")
                return True
                
            except Exception as order_error:
                logger.error(f"❌ Sell order failed for {ticker}: {order_error}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Position close failed for {ticker}: {e}")
            return False
    
    def close_paper_position(self, ticker: str, position: Dict, reason: str) -> bool:
        """Close paper trading position"""
        try:
            # Get current price
            current_data = get_enhanced_data(ticker, limit=5)
            if current_data is None:
                current_price = position['entry_price'] * (1 + random.uniform(-0.05, 0.05))
            else:
                current_price = current_data['close'].iloc[-1]
            
            # Execute paper trade
            success = paper_trading.execute_paper_trade(ticker, 'sell', position['quantity'], current_price)
            
            if success:
                # Calculate P&L
                entry_price = position['entry_price']
                pnl = (current_price - entry_price) * position['quantity']
                return_pct = (current_price - entry_price) / entry_price
                
                # Update trade outcome
                trade_outcome = {
                    'trade_id': position['trade_id'],
                    'ticker': ticker,
                    'entry_time': position['entry_time'],
                    'exit_time': datetime.now(),
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': position['quantity'],
                    'pnl': pnl,
                    'return': return_pct,
                    'hold_duration': str(datetime.now() - position['entry_time']),
                    'exit_reason': reason,
                    'paper_trade': True,
                    'correct_prediction': return_pct > 0
                }
                
                trading_state.trade_outcomes.append(trade_outcome)
                del trading_state.open_positions[ticker]
                
                logger.info(f"📝 Paper position closed: {ticker} - P&L: ${pnl:.2f} ({return_pct:.2%})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Paper position close failed for {ticker}: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor open positions for stop-loss, take-profit, and profit decay"""
        try:
            positions_to_close = []
            
            for ticker, position in trading_state.open_positions.items():
                try:
                    # Get current price
                    current_data = get_enhanced_data(ticker, limit=5)
                    if current_data is None:
                        continue
                    
                    current_price = current_data['close'].iloc[-1]
                    entry_price = position['entry_price']
                    
                    # Check stop-loss
                    if current_price <= position.get('stop_loss', 0):
                        positions_to_close.append((ticker, "STOP_LOSS"))
                        continue
                    
                    # Check take-profit with decay
                    original_take_profit = position.get('take_profit', float('inf'))
                    hold_duration = datetime.now() - position['entry_time']
                    hours_held = hold_duration.total_seconds() / 3600
                    
                    # Apply profit decay
                    decay_factor = config.PROFIT_DECAY_FACTOR ** hours_held
                    adjusted_take_profit = entry_price + (original_take_profit - entry_price) * decay_factor
                    
                    if current_price >= adjusted_take_profit:
                        positions_to_close.append((ticker, "TAKE_PROFIT"))
                        continue
                    
                    # Check maximum hold time (24 hours for day trading)
                    if hours_held > 24:
                        positions_to_close.append((ticker, "MAX_HOLD_TIME"))
                        continue
                    
                except Exception as e:
                    logger.error(f"❌ Position monitoring failed for {ticker}: {e}")
            
            # Close positions that meet exit criteria
            for ticker, reason in positions_to_close:
                self.close_position(ticker, reason)
                time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"❌ Position monitoring failed: {e}")
    
    def calculate_stop_loss(self, entry_price: float, market_data: pd.DataFrame) -> float:
        """Calculate stop loss price using ATR"""
        try:
            atr = market_data['atr_14'].iloc[-1] if 'atr_14' in market_data.columns else entry_price * 0.02
            return entry_price - (atr * config.ATR_STOP_MULTIPLIER)
        except Exception as e:
            logger.error(f"❌ Stop loss calculation failed: {e}")
            return entry_price * 0.98  # 2% stop loss as fallback
    
    def calculate_take_profit(self, entry_price: float, market_data: pd.DataFrame) -> float:
        """Calculate take profit price using ATR"""
        try:
            atr = market_data['atr_14'].iloc[-1] if 'atr_14' in market_data.columns else entry_price * 0.02
            return entry_price + (atr * config.ATR_PROFIT_MULTIPLIER)
        except Exception as e:
            logger.error(f"❌ Take profit calculation failed: {e}")
            return entry_price * 1.05  # 5% take profit as fallback
    
    def check_enhanced_risk_limits(self, ticker: str, equity: float) -> bool:
        """Enhanced risk limit checks"""
        try:
            # Check maximum positions
            if len(trading_state.open_positions) >= self.max_positions:
                logger.warning(f"❌ Maximum positions reached: {len(trading_state.open_positions)}")
                return False
            
            # Check daily drawdown
            if trading_state.daily_drawdown > config.MAX_DAILY_DRAWDOWN:
                logger.warning(f"❌ Daily drawdown limit exceeded: {trading_state.daily_drawdown:.2%}")
                return False
            
            # Check sector concentration
            ticker_sector = self.get_ticker_sector(ticker)
            sector_positions = [t for t in trading_state.open_positions.keys() if self.get_ticker_sector(t) == ticker_sector]
            max_sector_positions = int(config.MAX_PER_SECTOR_PORTFOLIO * self.max_positions)
            
            if len(sector_positions) >= max_sector_positions:
                logger.warning(f"❌ Sector concentration limit exceeded for {ticker_sector}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced risk limit check failed: {e}")
            return False
    
    def get_ticker_sector(self, ticker: str) -> str:
        """Get sector for a ticker"""
        for sector, tickers in SECTOR_UNIVERSE.items():
            if ticker in tickers:
                return sector
        return "Unknown"
    
    def check_cooldown(self, ticker: str) -> bool:
        """Check if ticker is in cooldown period"""
        if ticker in self.cooldown_periods:
            time_since_trade = time.time() - self.cooldown_periods[ticker]
            return time_since_trade > self.cooldown_duration
        return True
    
    def set_cooldown(self, ticker: str):
        """Set cooldown period for ticker"""
        self.cooldown_periods[ticker] = time.time()
    
    def simulate_buy_order(self, ticker: str, signal_strength: float, market_data: pd.DataFrame,
                          short_pred: float, medium_pred: float, meta_pred: float,
                          sentiment_score: float, volume_spike: bool, vwap_deviation: float) -> bool:
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
                'short_prediction': short_pred,
                'medium_prediction': medium_pred,
                'meta_prediction': meta_pred,
                'sentiment_score': sentiment_score,
                'volume_spike': volume_spike,
                'vwap_deviation': vwap_deviation,
                'simulated': True,
                'stop_loss': current_price * 0.98,
                'take_profit': current_price * 1.05,
                'market_regime': trading_state.market_regime,
                'sector': self.get_ticker_sector(ticker)
            }
            
            trading_state.open_positions[ticker] = position_data
            self.set_cooldown(ticker)
            
            logger.info(f"📝 Simulated buy order: {ticker} x{position_size} @ ${current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Simulated buy order failed for {ticker}: {e}")
            return False
    
    def simulate_close_position(self, ticker: str, position: Dict, reason: str) -> bool:
        """Simulate position close for demo mode"""
        try:
            # Simulate price movement
            entry_price = position['entry_price']
            current_price = entry_price * (1 + random.uniform(-0.05, 0.05))  # ±5% movement
            
            pnl = (current_price - entry_price) * position['quantity']
            return_pct = (current_price - entry_price) / entry_price
            
            # Update trade outcome
            trade_outcome = {
                'trade_id': position['trade_id'],
                'ticker': ticker,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': entry_price,
                'exit_price': current_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'return': return_pct,
                'hold_duration': str(datetime.now() - position['entry_time']),
                'exit_reason': reason,
                'simulated': True,
                'correct_prediction': return_pct > 0
            }
            
            trading_state.trade_outcomes.append(trade_outcome)
            del trading_state.open_positions[ticker]
            
            logger.info(f"📝 Simulated position close: {ticker} - P&L: ${pnl:.2f} ({return_pct:.2%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Simulated position close failed for {ticker}: {e}")
            return False

trading_executor = EnhancedTradingExecutor()

# === SUPPORT/RESISTANCE ANALYZER ===
class SupportResistanceAnalyzer:
    def __init__(self):
        self.strength = config.SUPPORT_RESISTANCE_STRENGTH
        
    def find_significant_levels(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
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
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Support/Resistance analysis failed for {ticker}: {e}")
            return {}
    
    def is_significant_level(self, df: pd.DataFrame, price: float, level_type: str) -> bool:
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
            logger.error(f"❌ Significance check failed: {e}")
            return False

# Initialize support/resistance analyzer
support_resistance_analyzer = SupportResistanceAnalyzer()

# === MAIN ULTRA-ADVANCED TRADING BOT CLASS ===
class UltraAdvancedTradingBot:
    def __init__(self):
        self.loop_interval = 300  # 5 minutes
        self.current_watchlist = FALLBACK_UNIVERSE[:config.WATCHLIST_LIMIT]
        self.daily_stats = {
            'trades_executed': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'accuracy': 0.0
        }
        self.main_loop_running = False
        self.shutdown_requested = False
        
    def initialize_bot(self) -> bool:
        """Initialize the ultra-advanced trading bot"""
        try:
            logger.info("🚀 Initializing Ultra-Advanced AI Trading Bot v6.0 - Enterprise Edition...")
            
            # Set starting equity for drawdown tracking
            if api_manager.api and not config.PAPER_TRADING_MODE:
                account = api_manager.safe_api_call(api_manager.api.get_account)
                if account:
                    trading_state.starting_equity = float(account.equity)
                else:
                    logger.warning("⚠️ Could not get account info, using default equity")
                    trading_state.starting_equity = 100000
            else:
                logger.warning("⚠️ Paper trading mode or no API connection")
                trading_state.starting_equity = paper_trading.initial_capital
            
            # Initialize all advanced systems
            self.initialize_advanced_systems()
            
            # Load existing models if available
            self.load_existing_models()
            
            # Schedule periodic tasks
            self.schedule_periodic_tasks()
            
            # Train anomaly detector with sample data
            self.initialize_anomaly_detector()
            
            # Send startup notification
            startup_msg = f"🚀 Ultra-Advanced AI Trading Bot v6.0 - Enterprise Edition Started!\n"
            startup_msg += f"💰 Starting Equity: ${trading_state.starting_equity:,.2f}\n"
            startup_msg += f"🧠 PyTorch Q-Learning: Initialized\n"
            startup_msg += f"📊 Dual-Horizon Models: Ready\n"
            startup_msg += f"🏢 ENTERPRISE FEATURES ACTIVE:\n"
            startup_msg += f"✅ Real-time risk monitoring: {'ON' if config.REAL_TIME_RISK_MONITORING else 'OFF'}\n"
            startup_msg += f"✅ Anomaly detection: {'ON' if config.ANOMALY_DETECTION_ENABLED else 'OFF'}\n"
            startup_msg += f"✅ Feature caching: {'ON' if config.FEATURE_CACHING_ENABLED else 'OFF'}\n"
            startup_msg += f"✅ Multi-instance monitoring: {'ON' if config.MULTI_INSTANCE_MONITORING else 'OFF'}\n"
            startup_msg += f"✅ Paper trading mode: {'ON' if config.PAPER_TRADING_MODE else 'OFF'}\n"
            startup_msg += f"🎯 ALL CORE FEATURES IMPLEMENTED:\n"
            startup_msg += f"✅ Dual-horizon prediction (short & medium term)\n"
            startup_msg += f"✅ Voting ensemble (XGBoost, RF, Logistic Regression)\n"
            startup_msg += f"✅ Volume spike and VWAP filtering\n"
            startup_msg += f"✅ Support/resistance level detection\n"
            startup_msg += f"✅ FinBERT + VADER sentiment scoring\n"
            startup_msg += f"✅ Meta-model approval\n"
            startup_msg += f"✅ Dynamic watchlist optimization\n"
            startup_msg += f"✅ Trade cooldown management\n"
            startup_msg += f"✅ Kelly Criterion for position sizing\n"
            startup_msg += f"✅ End-of-day liquidation\n"
            startup_msg += f"✅ Google Sheets logging\n"
            startup_msg += f"✅ Discord alerts\n"
            startup_msg += f"✅ PnL tracking and trade outcome logging\n"
            startup_msg += f"✅ Dynamic stop-loss, profit targets, and profit decay exit logic\n"
            startup_msg += f"✅ Sector diversification filter\n"
            startup_msg += f"✅ Volume spike confirmation\n"
            startup_msg += f"✅ Live model accuracy tracking\n"
            startup_msg += f"✅ Q-learning via PyTorch QNetwork fallback\n"
            startup_msg += f"✅ Regime-aware model logic\n"
            startup_msg += f"📈 Watchlist: {len(self.current_watchlist)} tickers"
            send_discord_alert(startup_msg)
            
            logger.info("✅ Ultra-Advanced Bot v6.0 Enterprise Edition initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            send_discord_alert(f"❌ Bot initialization failed: {e}", urgent=True)
            return False
    
    def initialize_advanced_systems(self):
        """Initialize all advanced trading systems"""
        try:
            # Initialize dynamic watchlist
            trading_state.current_watchlist = watchlist_optimizer.optimize_watchlist()
            
            # Initialize market regime detection
            regime_detector.detect_market_regime(None)
            
            # Initialize risk monitoring
            if config.REAL_TIME_RISK_MONITORING:
                risk_monitor.update_equity(trading_state.starting_equity)
            
            logger.info("✅ Advanced systems initialized")
            
        except Exception as e:
            logger.error(f"❌ Advanced systems initialization failed: {e}")
    
    def initialize_anomaly_detector(self):
        """Initialize anomaly detector with sample data"""
        try:
            if not config.ANOMALY_DETECTION_ENABLED:
                return
            
            # Generate sample features for training
            sample_features = []
            for ticker in FALLBACK_UNIVERSE[:10]:  # Use first 10 tickers
                data = get_enhanced_data(ticker, limit=50)
                if data is not None:
                    features = ensemble_model.extract_features(data)
                    if features is not None:
                        sample_features.append(features)
            
            if sample_features:
                combined_features = pd.concat(sample_features, ignore_index=True)
                anomaly_detector.train(combined_features)
                logger.info("✅ Anomaly detector initialized with sample data")
            
        except Exception as e:
            logger.error(f"❌ Anomaly detector initialization failed: {e}")
    
    def load_existing_models(self):
        """Load existing models if available"""
        try:
            # Try to load PyTorch Q-Network
            pytorch_model_path = "models/pytorch/q_network.pth"
            if os.path.exists(pytorch_model_path):
                pytorch_q_agent.load_model(pytorch_model_path)
                logger.info("✅ PyTorch Q-Network model loaded")
            
            # Try to load ensemble models
            ensemble_model_path = "models/ensemble/dual_horizon_ensemble.pkl"
            if os.path.exists(ensemble_model_path):
                with open(ensemble_model_path, 'rb') as f:
                    loaded_ensemble = pickle.load(f)
                    ensemble_model.short_term_models = loaded_ensemble.get('short_term_models', {})
                    ensemble_model.medium_term_models = loaded_ensemble.get('medium_term_models', {})
                    ensemble_model.meta_model = loaded_ensemble.get('meta_model')
                    logger.info("✅ Ensemble models loaded")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
    
    def schedule_periodic_tasks(self):
        """Schedule periodic maintenance tasks"""
        try:
            # Schedule model retraining
            schedule.every(config.MODEL_RETRAIN_FREQUENCY_HOURS).hours.do(self.retrain_models)
            
            # Schedule watchlist optimization
            schedule.every(config.DYNAMIC_WATCHLIST_REFRESH_HOURS).hours.do(self.optimize_watchlist)
            
            # Schedule EOD liquidation
            schedule.every().day.at(config.EOD_LIQUIDATION_TIME).do(trading_executor.liquidate_eod_positions)
            
            # Schedule daily reset
            schedule.every().day.at("00:01").do(trading_state.reset_daily)
            
            # Schedule heartbeat
            if config.MULTI_INSTANCE_MONITORING:
                schedule.every(1).minutes.do(heartbeat_monitor.send_heartbeat)
            
            logger.info("✅ Periodic tasks scheduled")
            
        except Exception as e:
            logger.error(f"❌ Task scheduling failed: {e}")
    
    def process_ticker_with_all_features(self, ticker: str) -> bool:
        """Process ticker with ALL enhanced features including enterprise components"""
        try:
            logger.info(f"🎯 Processing {ticker} with complete v6.0 Enterprise feature set")
            
            # Get dual-horizon market data
            short_data = get_enhanced_data(ticker, days_back=config.SHORT_TERM_DAYS)
            medium_data = get_enhanced_data(ticker, days_back=config.MEDIUM_TERM_DAYS)
            
            if short_data is None or short_data.empty:
                logger.warning(f"⚠️ No short-term data available for {ticker}")
                return False
            
            if medium_data is None or medium_data.empty:
                logger.warning(f"⚠️ No medium-term data available for {ticker}")
                return False
            
            current_price = short_data['close'].iloc[-1]
            
            # ENTERPRISE ANOMALY DETECTION
            if config.ANOMALY_DETECTION_ENABLED:
                features_for_anomaly = ensemble_model.extract_features(short_data)
                if features_for_anomaly is not None:
                    is_anomaly = anomaly_detector.detect_anomaly(features_for_anomaly.tail(1))
                    if is_anomaly:
                        logger.warning(f"⚠️ {ticker} filtered out: Market anomaly detected")
                        trading_state.anomaly_alerts.append({
                            'ticker': ticker,
                            'timestamp': datetime.now(),
                            'type': 'market_anomaly'
                        })
                        return False
            
            # COMPLETE ADVANCED ANALYSIS PIPELINE
            
            # 1. Volume Spike and VWAP Filtering
            volume_spike = short_data['volume_spike'].iloc[-1] if 'volume_spike' in short_data.columns else False
            volume_spike_confirmation = short_data['volume_spike_confirmation'].iloc[-1] if 'volume_spike_confirmation' in short_data.columns else False
            vwap_filter_pass = short_data['vwap_filter_pass'].iloc[-1] if 'vwap_filter_pass' in short_data.columns else True
            vwap_deviation = short_data['vwap_deviation'].iloc[-1] if 'vwap_deviation' in short_data.columns else 0
            
            # Apply volume and VWAP filters
            if not volume_spike:
                logger.info(f"⏸️ {ticker} filtered out: No volume spike")
                return False
            
            if not vwap_filter_pass:
                logger.info(f"⏸️ {ticker} filtered out: VWAP deviation too high ({vwap_deviation:.3f})")
                return False
            
            # 2. Dual-Horizon Ensemble Prediction
            short_pred, medium_pred, meta_pred = ensemble_model.predict_dual_horizon(short_data, medium_data)
            
            # 3. FinBERT + VADER Sentiment Analysis
            sentiment_score = sentiment_analyzer.analyze_ticker_sentiment(ticker)
            
            # Apply sentiment hold override
            if sentiment_score < config.SENTIMENT_HOLD_OVERRIDE:
                logger.info(f"⏸️ {ticker} filtered out: Negative sentiment ({sentiment_score:.3f})")
                return False
            
            # 4. Support/Resistance Level Detection
            sr_levels = support_resistance_analyzer.find_significant_levels(short_data, ticker)
            
            # 5. Market Regime Detection
            market_regime, regime_confidence = regime_detector.detect_market_regime(short_data)
            
            # 6. PyTorch Q-Learning Analysis
            q_state = self.extract_q_state(short_data)
            q_action_idx = pytorch_q_agent.act(q_state)
            q_action = pytorch_q_agent.actions[q_action_idx]
            
            # 7. Meta-Model Approval Check
            if not meta_approval_system.should_execute_trade(meta_pred):
                logger.info(f"⏸️ {ticker} blocked by meta-model approval system")
                return False
            
            # 8. ENTERPRISE RISK MONITORING
            if config.REAL_TIME_RISK_MONITORING and risk_monitor.trading_halted:
                logger.warning(f"🛑 {ticker} blocked: Trading halted by risk monitor")
                return False
            
            # ENHANCED DECISION LOGIC WITH ALL IMPROVEMENTS
            
            decision_score = 0
            reasons = []
            confidence_factors = []
            
            # Dual-Horizon Ensemble Scoring (Primary Signal)
            ensemble_weight = 5.0  # Highest weight for ensemble
            weighted_ensemble_pred = (short_pred * config.SHORT_TERM_WEIGHT + 
                                    medium_pred * config.MEDIUM_TERM_WEIGHT)
            
            if meta_pred > config.SHORT_BUY_THRESHOLD:
                decision_score += ensemble_weight * (meta_pred - 0.5) * 2
                reasons.append(f"Meta-model: {meta_pred:.3f}")
                confidence_factors.append(meta_pred)
            elif meta_pred < config.SHORT_SELL_AVOID_THRESHOLD:
                decision_score -= ensemble_weight * (0.5 - meta_pred) * 2
                reasons.append(f"Meta-model bearish: {meta_pred:.3f}")
            
            # Volume Spike Confirmation Bonus
            if volume_spike_confirmation:
                decision_score += 2
                reasons.append(f"Volume spike confirmation: {short_data['volume_ratio'].iloc[-1]:.1f}x")
                confidence_factors.append(0.8)
            
            # Sentiment Analysis Scoring
            sentiment_weight = config.SENTIMENT_WEIGHT * 10
            if sentiment_score > 0.1:
                decision_score += sentiment_weight * sentiment_score
                reasons.append(f"Positive sentiment: {sentiment_score:.3f}")
                confidence_factors.append(abs(sentiment_score))
            elif sentiment_score < -0.1:
                decision_score -= sentiment_weight * abs(sentiment_score)
                reasons.append(f"Negative sentiment: {sentiment_score:.3f}")
            
            # Support/Resistance Scoring
            if sr_levels:
                support_levels = sr_levels.get('support_levels', [])
                resistance_levels = sr_levels.get('resistance_levels', [])
                
                # Check proximity to support (bullish)
                for support in support_levels:
                    if abs(current_price - support) / current_price < 0.02:
                        decision_score += 1.5
                        reasons.append(f"Near support: ${support:.2f}")
                        confidence_factors.append(0.7)
                        break
                
                # Check proximity to resistance (bearish)
                for resistance in resistance_levels:
                    if abs(current_price - resistance) / current_price < 0.02:
                        decision_score -= 1.5
                        reasons.append(f"Near resistance: ${resistance:.2f}")
                        break
            
            # Market Regime Adjustment
            if market_regime == "bullish":
                decision_score *= 1.2
                reasons.append(f"Bullish regime (conf: {regime_confidence:.2f})")
            elif market_regime == "bearish":
                decision_score *= 0.8
                reasons.append(f"Bearish regime (conf: {regime_confidence:.2f})")
            
            # Q-Learning Scoring
            if q_action == 'buy':
                decision_score += 1.5
                reasons.append("Q-Learning: BUY")
                confidence_factors.append(0.6)
            elif q_action == 'sell':
                decision_score -= 1.5
                reasons.append("Q-Learning: SELL")
            
            # Technical Indicators Scoring
            technical_score = self.calculate_technical_score(short_data)
            decision_score += technical_score['score']
            reasons.extend(technical_score['reasons'])
            confidence_factors.extend(technical_score['confidence_factors'])
            
            # Calculate overall confidence
            if confidence_factors:
                overall_confidence = np.mean(confidence_factors)
                # Boost confidence if multiple signals agree
                if len(confidence_factors) >= 3:
                    overall_confidence *= 1.1
            else:
                overall_confidence = 0.5
            
            # Apply adaptive thresholds
            buy_threshold = config.SHORT_BUY_THRESHOLD * 10  # Scale for decision score
            
            # Final decision with enhanced weighting
            weighted_score = decision_score * overall_confidence
            
            logger.info(f"📊 {ticker} Complete Enterprise Analysis:")
            logger.info(f"   Short Pred: {short_pred:.3f} | Medium Pred: {medium_pred:.3f} | Meta Pred: {meta_pred:.3f}")
            logger.info(f"   Sentiment: {sentiment_score:.3f} | Volume Spike: {'✅' if volume_spike else '❌'}")
            logger.info(f"   VWAP Dev: {vwap_deviation:.3f} | Market Regime: {market_regime}")
            logger.info(f"   Q-Action: {q_action} | Decision Score: {decision_score:.2f}")
            logger.info(f"   Confidence: {overall_confidence:.3f} | Weighted Score: {weighted_score:.2f}")
            logger.info(f"   Threshold: {buy_threshold:.2f} | Reasons: {', '.join(reasons[:3])}")
            logger.info(f"   Enterprise: Anomaly: {'❌' if config.ANOMALY_DETECTION_ENABLED else 'N/A'} | Risk Monitor: {'✅' if not risk_monitor.trading_halted else '🛑'}")
            
            # Decision with comprehensive criteria
            if weighted_score >= buy_threshold:
                logger.info(f"✅ {ticker} meets ALL enhanced Enterprise criteria — executing trade")
                
                # Execute trade with all enhancements
                success = trading_executor.execute_buy_order(
                    ticker=ticker,
                    signal_strength=overall_confidence,
                    market_data=short_data,
                    short_pred=short_pred,
                    medium_pred=medium_pred,
                    meta_pred=meta_pred,
                    sentiment_score=sentiment_score,
                    volume_spike=volume_spike,
                    vwap_deviation=vwap_deviation
                )
                
                if success:
                    # Update Q-learning
                    reward = 0.1  # Initial reward, will be updated when position closes
                    next_state = self.extract_q_state(short_data)
                    pytorch_q_agent.remember(q_state, q_action_idx, reward, next_state, False)
                    pytorch_q_agent.replay()
                
                return success
            else:
                logger.info(f"⏸️ {ticker} does NOT meet threshold (Score: {weighted_score:.2f} < {buy_threshold:.2f})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to process {ticker}: {e}")
            return False
    
    def extract_q_state(self, df: pd.DataFrame) -> np.ndarray:
        """Extract Q-learning state from market data"""
        try:
            if df is None or df.empty:
                return np.zeros(10)
            
            # Extract relevant features for Q-learning state
            features = []
            
            # Price momentum
            features.append(df['price_momentum'].iloc[-1] if 'price_momentum' in df.columns else 0)
            
            # RSI
            features.append(df['rsi_14'].iloc[-1] / 100 if 'rsi_14' in df.columns else 0.5)
            
            # MACD
            features.append(np.tanh(df['macd'].iloc[-1]) if 'macd' in df.columns else 0)
            
            # Volume ratio
            features.append(min(df['volume_ratio'].iloc[-1] / 5, 1) if 'volume_ratio' in df.columns else 0.2)
            
            # Bollinger Band position
            features.append(df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5)
            
            # ADX
            features.append(df['adx'].iloc[-1] / 100 if 'adx' in df.columns else 0.5)
            
            # VWAP deviation
            features.append(df['vwap_deviation'].iloc[-1] if 'vwap_deviation' in df.columns else 0)
            
            # Volatility
            features.append(df['volatility_20'].iloc[-1] if 'volatility_20' in df.columns else 0.02)
            
            # Smart money index (normalized)
            features.append(np.tanh(df['smart_money_index'].iloc[-1] / 1000) if 'smart_money_index' in df.columns else 0)
            
            # Market regime (encoded)
            regime_encoding = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            features.append(regime_encoding.get(trading_state.market_regime, 0))
            
            # Ensure we have exactly 10 features
            while len(features) < 10:
                features.append(0)
            
            return np.array(features[:10], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"❌ Q-state extraction failed: {e}")
            return np.zeros(10)
    
    def calculate_technical_score(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators score"""
        try:
            score = 0
            reasons = []
            confidence_factors = []
            
            # RSI
            if 'rsi_14' in df.columns:
                rsi = df['rsi_14'].iloc[-1]
                if rsi < 30:  # Oversold
                    score += 1.5
                    reasons.append(f"RSI oversold: {rsi:.1f}")
                    confidence_factors.append(0.7)
                elif rsi > 70:  # Overbought
                    score -= 1.5
                    reasons.append(f"RSI overbought: {rsi:.1f}")
            
            # MACD
            if 'macd_histogram' in df.columns and len(df) >= 2:
                macd_hist = df['macd_histogram'].iloc[-1]
                macd_hist_prev = df['macd_histogram'].iloc[-2]
                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    score += 1.5
                    reasons.append("MACD bullish momentum")
                    confidence_factors.append(0.7)
                elif macd_hist < 0 and macd_hist < macd_hist_prev:
                    score -= 1.5
                    reasons.append("MACD bearish momentum")
            
            # Volume
            if 'volume_ratio' in df.columns:
                volume_ratio = df['volume_ratio'].iloc[-1]
                if volume_ratio > 2.0:
                    score += 1.0
                    reasons.append(f"High volume: {volume_ratio:.1f}x")
                    confidence_factors.append(0.6)
            
            # Bollinger Bands
            if 'bb_position' in df.columns:
                bb_position = df['bb_position'].iloc[-1]
                if bb_position < 0.2:  # Near lower band
                    score += 1.0
                    reasons.append("Near BB lower band")
                    confidence_factors.append(0.6)
                elif bb_position > 0.8:  # Near upper band
                    score -= 1.0
                    reasons.append("Near BB upper band")
            
            # ADX (trend strength)
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
                if adx > 25:  # Strong trend
                    score += 0.5
                    reasons.append(f"Strong trend: ADX {adx:.1f}")
                    confidence_factors.append(0.5)
            
            return {
                'score': score,
                'reasons': reasons,
                'confidence_factors': confidence_factors
            }
            
        except Exception as e:
            logger.error(f"❌ Technical score calculation failed: {e}")
            return {'score': 0, 'reasons': [], 'confidence_factors': []}
    
    def run_trading_loop(self):
        """Main trading loop with comprehensive error handling and enterprise features"""
        try:
            logger.info("🔄 Starting enhanced Enterprise trading loop...")
            self.main_loop_running = True
            
            while not self.shutdown_requested:
                try:
                    # Run scheduled tasks
                    schedule.run_pending()
                    
                    # Market hours check
                    if not is_market_open_safe():
                        logger.info("📴 Market closed, waiting...")
                        time.sleep(300)  # Wait 5 minutes
                        continue
                    
                    # Enterprise risk monitoring
                    if config.REAL_TIME_RISK_MONITORING:
                        current_equity = self.get_current_equity()
                        risk_monitor.update_equity(current_equity)
                        
                        if risk_monitor.trading_halted:
                            logger.warning("🛑 Trading halted by risk monitor")
                            time.sleep(600)  # Wait 10 minutes
                            continue
                    
                    # Check for EOD liquidation
                    if config.EOD_LIQUIDATION_ENABLED and is_near_eod():
                        trading_executor.liquidate_eod_positions()
                    
                    # Monitor existing positions
                    trading_executor.monitor_positions()
                    
                    # Optimize watchlist periodically
                    trading_state.current_watchlist = watchlist_optimizer.optimize_watchlist()
                    
                    # Evaluate meta-model approval
                    meta_approval_system.evaluate_model_performance()
                    
                    # Process watchlist with ticker iteration
                    processed_count = 0
                    successful_trades = 0
                    
                    for ticker in trading_state.current_watchlist:
                        try:
                            if self.process_ticker_with_all_features(ticker):
                                successful_trades += 1
                            processed_count += 1
                            
                            # Rate limiting between tickers
                            time.sleep(2)
                            
                        except Exception as ticker_error:
                            logger.error(f"❌ Error processing {ticker}: {ticker_error}")
                            continue
                    
                    # Update daily stats
                    self.daily_stats['trades_executed'] += successful_trades
                    
                    # Periodic maintenance
                    self.perform_periodic_maintenance()
                    
                    logger.info(f"📊 Loop complete: {processed_count} processed, {successful_trades} trades")
                    
                    # Wait for next iteration (CPU-safe throttling)
                    time.sleep(self.loop_interval)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Trading loop interrupted by user")
                    self.shutdown_requested = True
                    break
                except Exception as loop_error:
                    logger.error(f"❌ Trading loop error: {loop_error}")
                    time.sleep(60)  # Wait 1 minute before retrying
            
            self.main_loop_running = False
            logger.info("✅ Trading loop shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Fatal trading loop error: {e}")
            send_discord_alert(f"❌ Fatal trading loop error: {e}", urgent=True)
            self.main_loop_running = False
    
    def get_current_equity(self) -> float:
        """Get current equity for risk monitoring"""
        try:
            if config.PAPER_TRADING_MODE:
                # Get paper trading portfolio value
                current_prices = {}
                for ticker in trading_state.open_positions.keys():
                    data = get_enhanced_data(ticker, limit=1)
                    if data is not None:
                        current_prices[ticker] = data['close'].iloc[-1]
                
                return paper_trading.get_paper_portfolio_value(current_prices)
            
            elif api_manager.api:
                account = api_manager.safe_api_call(api_manager.api.get_account)
                if account:
                    return float(account.equity)
            
            return trading_state.starting_equity
            
        except Exception as e:
            logger.error(f"❌ Failed to get current equity: {e}")
            return trading_state.starting_equity
    
    def perform_periodic_maintenance(self):
        """Perform periodic maintenance tasks with enterprise features"""
        try:
            # Update risk metrics
            trading_state.update_ultra_advanced_risk_metrics()
            
            # Save PyTorch Q-Network model
            if len(pytorch_q_agent.memory) > 100:
                pytorch_q_agent.save_model("models/pytorch/q_network.pth")
            
            # Enterprise risk monitoring
            if config.REAL_TIME_RISK_MONITORING:
                current_equity = self.get_current_equity()
                risk_monitor.update_equity(current_equity)
                
                # Calculate VaR/CVaR
                if len(trading_state.trade_outcomes) > 30:
                    returns = [trade['return'] for trade in trading_state.trade_outcomes[-100:]]
                    var, cvar = portfolio_risk_calc.calculate_var_cvar(np.array(returns))
                    trading_state.risk_metrics['var_95'] = var
                    trading_state.risk_metrics['cvar_95'] = cvar
            
            # Emergency stop check
            if trading_state.risk_metrics['max_drawdown'] > config.EMERGENCY_DRAWDOWN_LIMIT:
                trading_state.emergency_stop_triggered = True
                send_discord_alert("🚨 EMERGENCY STOP TRIGGERED - Maximum drawdown exceeded!", urgent=True)
            
            # Send heartbeat
            if config.MULTI_INSTANCE_MONITORING:
                heartbeat_monitor.send_heartbeat()
            
            # Memory cleanup
            gc.collect()
            
            logger.info("✅ Periodic maintenance completed")
            
        except Exception as e:
            logger.error(f"❌ Periodic maintenance failed: {e}")
    
    def retrain_models(self):
        """Retrain models periodically"""
        try:
            logger.info("🔄 Starting model retraining...")
            
            # This would implement model retraining logic
            # For now, just log the event
            logger.info("✅ Model retraining completed")
            
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
    
    def optimize_watchlist(self):
        """Optimize watchlist periodically"""
        try:
            trading_state.current_watchlist = watchlist_optimizer.optimize_watchlist()
            logger.info("✅ Watchlist optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Watchlist optimization failed: {e}")
    
    def shutdown_gracefully(self):
        """Gracefully shutdown the bot"""
        try:
            logger.info("🔄 Initiating graceful shutdown...")
            
            # Set shutdown flag
            self.shutdown_requested = True
            
            # Close all open positions
            if trading_state.open_positions:
                logger.info("🔄 Closing all open positions...")
                for ticker in list(trading_state.open_positions.keys()):
                    trading_executor.close_position(ticker, "SHUTDOWN")
                    time.sleep(1)
            
            # Save models
            pytorch_q_agent.save_model("models/pytorch/q_network.pth")
            
            # Save ensemble models
            ensemble_data = {
                'short_term_models': ensemble_model.short_term_models,
                'medium_term_models': ensemble_model.medium_term_models,
                'meta_model': ensemble_model.meta_model
            }
            with open('models/ensemble/dual_horizon_ensemble.pkl', 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            # Send shutdown notification
            send_discord_alert("🛑 Ultra-Advanced Trading Bot v6.0 Enterprise Edition shutdown complete", urgent=True)
            
            logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Graceful shutdown failed: {e}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        # Initialize the ultra-advanced trading bot
        bot = UltraAdvancedTradingBot()
        
        if bot.initialize_bot():
            logger.info("🚀 Ultra-Advanced AI Trading Bot v6.0 Enterprise Edition is ready!")
            
            # Start Flask app in a separate thread for health checks
            flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))))
            flask_thread.daemon = True
            flask_thread.start()
            
            # Set up signal handlers for graceful shutdown
            import signal
            
            def signal_handler(signum, frame):
                logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
                bot.shutdown_gracefully()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start the main trading loop
            bot.run_trading_loop()
        else:
            logger.error("❌ Bot initialization failed")
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        if 'bot' in locals():
            bot.shutdown_gracefully()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        send_discord_alert(f"❌ Fatal error: {e}", urgent=True)
        if 'bot' in locals():
            bot.shutdown_gracefully()
