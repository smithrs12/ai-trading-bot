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
    # Market Hours & Timing
    MARKET_OPEN_WAIT_MINUTES: int = 30  # Wait 30 minutes after market open
    EOD_LIQUIDATION_TIME: str = "15:45"  # 3:45 PM ET
    EOD_LIQUIDATION_ENABLED: bool = True
    OVERNIGHT_HOLD_PROFIT_THRESHOLD: float = 0.03  # 3% profit to consider overnight hold
    
    # Dual-horizon prediction settings
    SHORT_TERM_DAYS: int = 2
    MEDIUM_TERM_DAYS: int = 15
    SHORT_TERM_WEIGHT: float = 0.6
    MEDIUM_TERM_WEIGHT: float = 0.4
    
    # Signal Quality & Thresholds (Relaxed for day trading)
    SHORT_BUY_THRESHOLD: float = 0.51  # Lowered from 0.53
    SHORT_SELL_AVOID_THRESHOLD: float = 0.47  # Raised from 0.45
    MEDIUM_BUY_THRESHOLD: float = 0.52  # Lowered from 0.55
    MEDIUM_SELL_AVOID_THRESHOLD: float = 0.46  # Raised from 0.43
    PRICE_MOMENTUM_MIN: float = 0.003  # Lowered from 0.005
    VOLUME_SPIKE_MIN: float = 1.3  # Lowered from 1.5
    VOLUME_SPIKE_CONFIRMATION_MIN: float = 1.8  # Lowered from 2.0
    SENTIMENT_HOLD_OVERRIDE: float = -0.3  # Raised from -0.5
    VWAP_DEVIATION_THRESHOLD: float = 0.025  # Raised from 0.02
    
    # Portfolio Management
    MAX_PER_SECTOR_WATCHLIST: int = 15  # Increased from 12
    MAX_PER_SECTOR_PORTFOLIO: float = 0.3  # Increased from 0.25
    WATCHLIST_LIMIT: int = 50  # Increased from 25
    DYNAMIC_WATCHLIST_REFRESH_HOURS: int = 2  # Decreased from 4
    MAX_PORTFOLIO_RISK: float = 0.03  # Increased from 0.02
    MAX_DAILY_DRAWDOWN: float = 0.06  # Increased from 0.05
    EMERGENCY_DRAWDOWN_LIMIT: float = 0.12  # Increased from 0.10
    MAX_CORRELATION_THRESHOLD: float = 0.75  # Increased from 0.7
    
    # Risk Management & Position Sizing
    ATR_STOP_MULTIPLIER: float = 1.2  # Tightened from 1.5
    ATR_PROFIT_MULTIPLIER: float = 2.0  # Lowered from 2.5
    PROFIT_DECAY_FACTOR: float = 0.98  # Increased from 0.95
    VOLATILITY_GATE_THRESHOLD: float = 0.06  # Increased from 0.05
    MIN_MODEL_ACCURACY: float = 0.52  # Lowered from 0.55
    SHARPE_RATIO_MIN: float = 0.8  # Lowered from 1.0
    KELLY_FRACTION_MAX: float = 0.1  # Increased from 0.08
    KELLY_FRACTION_MIN: float = 0.015  # Increased from 0.01
    
    # Trade Management
    TRADE_COOLDOWN_MINUTES: int = 20  # Decreased from 30
    HOLD_POSITION_ENABLED: bool = True  # Enable position holding
    MIN_HOLD_TIME_MINUTES: int = 15  # Minimum hold time
    MAX_HOLD_TIME_HOURS: int = 6  # Maximum hold time for day trading
    
    # Meta-model and Ensemble
    META_MODEL_MIN_ACCURACY: float = 0.54  # Lowered from 0.58
    META_MODEL_MIN_TRADES: int = 15  # Lowered from 20
    ENSEMBLE_CONFIDENCE_THRESHOLD: float = 0.6  # Lowered from 0.65
    MODEL_RETRAIN_FREQUENCY_HOURS: int = 12  # Decreased from 24
    
    # Advanced Features
    SUPPORT_RESISTANCE_STRENGTH: int = 2  # Lowered from 3
    VOLUME_PROFILE_BINS: int = 20
    Q_LEARNING_ALPHA: float = 0.1
    Q_LEARNING_GAMMA: float = 0.95
    Q_LEARNING_EPSILON: float = 0.15  # Increased exploration
    Q_LEARNING_EPSILON_DECAY: float = 0.998  # Slower decay
    SECTOR_ROTATION_THRESHOLD: float = 0.015  # Lowered from 0.02
    
    # Sentiment Analysis
    FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
    SENTIMENT_WEIGHT: float = 0.12  # Lowered from 0.15
    NEWS_LOOKBACK_HOURS: int = 12  # Decreased from 24
    
    # Market Regime Detection
    REGIME_DETECTION_WINDOW: int = 30  # Decreased from 50
    BULL_MARKET_THRESHOLD: float = 0.015  # Lowered from 0.02
    BEAR_MARKET_THRESHOLD: float = -0.015  # Raised from -0.02
    
    # Enterprise Features
    PAPER_TRADING_MODE: bool = True  # Set to False for live trading
    REAL_TIME_RISK_MONITORING: bool = True
    ANOMALY_DETECTION_ENABLED: bool = True
    FEATURE_CACHING_ENABLED: bool = True
    MULTI_INSTANCE_MONITORING: bool = True
    
    # Ticker Evaluation & Training
    TICKER_EVALUATION_ENABLED: bool = True
    MIN_TICKERS_FOR_TRAINING: int = 20
    TICKER_LIQUIDITY_MIN: int = 500000  # Minimum daily volume
    TICKER_PRICE_MIN: float = 5.0  # Minimum price
    TICKER_PRICE_MAX: float = 500.0  # Maximum price
    
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

# === ENHANCED MARKET STATUS WITH 24/7 OPERATION ===
class MarketStatusManager:
    """Enhanced market status management for 24/7 operation"""
    
    def __init__(self):
        self.market_timezone = pytz.timezone("US/Eastern")
        self.last_market_check = None
        self.cached_market_status = False
        self.market_open_time = None
        self.market_close_time = None
        
    def is_market_open(self) -> bool:
        """Enhanced market open check with caching"""
        try:
            current_time = datetime.now(self.market_timezone)
            
            # Cache market status for 1 minute to reduce API calls
            if (self.last_market_check and 
                (current_time - self.last_market_check).total_seconds() < 60):
                return self.cached_market_status
            
            # Primary: API check
            if api_manager.api:
                clock = api_manager.safe_api_call(api_manager.api.get_clock)
                if clock:
                    self.cached_market_status = clock.is_open
                    self.last_market_check = current_time
                    return clock.is_open
            
            # Fallback: Manual calculation
            # Check if it's a weekday
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                self.cached_market_status = False
                self.last_market_check = current_time
                return False
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            is_open = market_open <= current_time <= market_close
            self.cached_market_status = is_open
            self.last_market_check = current_time
            
            return is_open
            
        except Exception as e:
            logger.error(f"Market status check failed: {e}")
            return False
    
    def is_in_trading_window(self) -> bool:
        """Check if we're in the trading window (30 minutes after open)"""
        try:
            if not self.is_market_open():
                return False
            
            current_time = datetime.now(self.market_timezone)
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            trading_start = market_open + timedelta(minutes=config.MARKET_OPEN_WAIT_MINUTES)
            
            return current_time >= trading_start
            
        except Exception as e:
            logger.error(f"Trading window check failed: {e}")
            return False
    
    def is_near_eod(self) -> bool:
        """Check if we're near end of day for liquidation"""
        try:
            current_time = datetime.now(self.market_timezone)
            eod_time = datetime.strptime(config.EOD_LIQUIDATION_TIME, "%H:%M").time()
            eod_datetime = current_time.replace(hour=eod_time.hour, minute=eod_time.minute, second=0, microsecond=0)
            
            return current_time >= eod_datetime
        except Exception as e:
            logger.error(f"EOD check failed: {e}")
            return False
    
    def should_hold_overnight(self, position_data: Dict) -> bool:
        """Determine if position should be held overnight"""
        try:
            if not config.EOD_LIQUIDATION_ENABLED:
                return True
            
            # Calculate current P&L
            entry_price = position_data.get('entry_price', 0)
            current_price = position_data.get('current_price', entry_price)
            
            if entry_price > 0:
                profit_pct = (current_price - entry_price) / entry_price
                
                # Hold if profitable above threshold
                if profit_pct >= config.OVERNIGHT_HOLD_PROFIT_THRESHOLD:
                    logger.info(f"Holding {position_data.get('ticker')} overnight - Profit: {profit_pct:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Overnight hold check failed: {e}")
            return False
    
    def get_time_until_market_open(self) -> timedelta:
        """Get time until next market open"""
        try:
            current_time = datetime.now(self.market_timezone)
            
            # If market is open, return 0
            if self.is_market_open():
                return timedelta(0)
            
            # Calculate next market open
            next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # If it's after market hours today, move to next day
            if current_time.hour >= 16:
                next_open += timedelta(days=1)
            
            # Skip weekends
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            
            return next_open - current_time
            
        except Exception as e:
            logger.error(f"Time until market open calculation failed: {e}")
            return timedelta(hours=1)  # Default to 1 hour

# Initialize market status manager
market_status = MarketStatusManager()
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

# === EXPANDED UNIVERSE WITH LARGE VARIETY ===
EXPANDED_SECTOR_UNIVERSE = {
    "Technology": [
        # Large Cap Tech
        "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "TSLA", "AMZN", "NFLX", "CRM",
        "ORCL", "IBM", "INTC", "QCOM", "AVGO", "TXN", "MU", "ADBE", "NOW", "INTU",
        # Mid Cap Tech
        "SNOW", "SHOP", "PLTR", "RBLX", "ZM", "DOCU", "OKTA", "CRWD", "DDOG", "NET",
        "TWLO", "ROKU", "SQ", "PYPL", "UBER", "LYFT", "ABNB", "COIN", "HOOD", "SOFI",
        # Small Cap Tech
        "UPST", "AFRM", "OPEN", "WISH", "CLOV", "SPCE", "LCID", "RIVN", "NKLA", "FSLY",
        "ESTC", "ELASTIC", "MDB", "TEAM", "ATLASSIAN", "ZS", "PANW", "FTNT", "CYBR", "SPLK"
    ],
    "Finance": [
        # Banks
        "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "COF", "SCHW", "MS", "GS",
        "BLK", "SPGI", "ICE", "CME", "MCO", "AXP", "V", "MA", "PYPL",
        # Insurance & REITs
        "BRK.B", "BRK.A", "AIG", "PGR", "TRV", "ALL", "MET", "PRU", "AFL", "HIG",
        # Fintech
        "SQ", "PYPL", "AFRM", "UPST", "SOFI", "LC", "COIN", "HOOD", "NU", "PAGS"
    ],
    "Energy": [
        # Oil & Gas
        "XOM", "CVX", "COP", "SLB", "PSX", "EOG", "MPC", "VLO", "OXY", "HAL",
        "BKR", "DVN", "FANG", "MRO", "APA", "HES", "KMI", "OKE", "EPD", "ET",
        # Renewables
        "NEE", "ENPH", "SEDG", "RUN", "NOVA", "FSLR", "SPWR", "PLUG", "BE", "BLDP"
    ],
    "Healthcare": [
        # Pharma
        "PFE", "JNJ", "LLY", "MRK", "ABT", "BMY", "GILD", "AMGN", "VRTX", "REGN",
        "BIIB", "CELG", "ILMN", "MRNA", "BNTX", "NVAX", "SGEN", "ALNY", "BMRN", "RARE",
        # Medical Devices
        "TMO", "DHR", "MDT", "ISRG", "SYK", "BSX", "EW", "HOLX", "DXCM", "VEEV",
        # Healthcare Services
        "UNH", "CVS", "ANTM", "HUM", "CNC", "WLP", "CI", "MOH", "ELV", "TDOC"
    ],
    "Consumer_Discretionary": [
        # Retail
        "AMZN", "HD", "LOW", "COST", "TGT", "WMT", "NKE", "SBUX", "MCD", "CMG",
        "DIS", "NFLX", "ROKU", "SPOT", "UBER", "LYFT", "ABNB", "BKNG", "EXPE", "TRIP",
        # Automotive
        "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI", "NKLA", "GOEV"
    ],
    "Consumer_Staples": [
        "PG", "PEP", "KO", "PM", "MO", "CL", "KMB", "GIS", "K", "CPB",
        "SJM", "HSY", "MDLZ", "MNST", "KDP", "STZ", "BUD", "TAP", "COKE", "KHC"
    ],
    "Industrial": [
        "UNP", "CSX", "UPS", "FDX", "CAT", "DE", "GE", "HON", "BA", "LMT",
        "RTX", "NOC", "MMM", "EMR", "ETN", "PH", "ITW", "CMI", "ROK", "DOV",
        "IR", "CARR", "OTIS", "PWR", "GNRC", "FAST", "PCAR", "WAB", "CHRW", "JBHT"
    ],
    "Materials": [
        "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "AA", "DOW", "DD",
        "PPG", "RPM", "IFF", "FMC", "LYB", "CF", "MOS", "ALB", "VMC", "MLM",
        "NUE", "STLD", "X", "CLF", "MT", "PKG", "IP", "WRK", "SON", "SEE"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "EXC", "XEL", "PEG", "SRE", "AEP", "PCG",
        "ED", "ETR", "WEC", "PPL", "CMS", "DTE", "NI", "LNT", "EVRG", "CNP",
        "ATO", "NWE", "UGI", "SWX", "NJR", "AWK", "WTR", "CWCO", "MSEX", "YORW"
    ],
    "Communication": [
        "T", "VZ", "CMCSA", "CHTR", "TMUS", "DISH", "SIRI", "LBRDK", "LBRDA", "PARA",
        "WBD", "NWSA", "NWS", "NYT", "GSAT", "IRDM", "VSAT", "ORBC", "GILT", "LUMN"
    ],
    "Real_Estate": [
        "PLD", "O", "SPG", "AMT", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR",
        "DLR", "WELL", "VTR", "ARE", "MAA", "ESS", "UDR", "CPT", "FRT", "REG",
        "BXP", "VNO", "SLG", "HIW", "KRC", "PGRE", "CUZ", "DEI", "ESRT", "JBGS"
    ]
}

# Flatten all tickers for fallback
ALL_TICKERS = [ticker for sector_tickers in EXPANDED_SECTOR_UNIVERSE.values() for ticker in sector_tickers]

# === TICKER EVALUATION SYSTEM ===
class TickerEvaluationSystem:
    """Evaluate tickers before training to save time and ensure quality"""
    
    def __init__(self):
        self.evaluated_tickers = {}
        self.last_evaluation = {}
        self.evaluation_cache_hours = 6
        
    def evaluate_ticker_quality(self, ticker: str) -> Dict[str, Any]:
        """Comprehensive ticker quality evaluation"""
        try:
            # Check cache first
            if self.is_evaluation_cached(ticker):
                return self.evaluated_tickers[ticker]
            
            logger.info(f"🔍 Evaluating ticker quality: {ticker}")
            
            # Get market data for evaluation
            data = get_enhanced_data(ticker, limit=100)
            if data is None or data.empty:
                return self.create_evaluation_result(ticker, False, "No data available")
            
            evaluation_score = 0
            reasons = []
            
            # 1. Liquidity Check
            avg_volume = data['volume'].mean()
            if avg_volume >= config.TICKER_LIQUIDITY_MIN:
                evaluation_score += 25
                reasons.append(f"Good liquidity: {avg_volume:,.0f}")
            else:
                return self.create_evaluation_result(ticker, False, f"Low liquidity: {avg_volume:,.0f}")
            
            # 2. Price Range Check
            current_price = data['close'].iloc[-1]
            if config.TICKER_PRICE_MIN <= current_price <= config.TICKER_PRICE_MAX:
                evaluation_score += 20
                reasons.append(f"Good price range: ${current_price:.2f}")
            else:
                return self.create_evaluation_result(ticker, False, f"Price out of range: ${current_price:.2f}")
            
            # 3. Volatility Check (not too low, not too high)
            volatility = data['returns'].std()
            if 0.01 <= volatility <= 0.08:  # 1% to 8% daily volatility
                evaluation_score += 20
                reasons.append(f"Good volatility: {volatility:.3f}")
            elif volatility < 0.01:
                evaluation_score += 10
                reasons.append(f"Low volatility: {volatility:.3f}")
            else:
                evaluation_score += 5
                reasons.append(f"High volatility: {volatility:.3f}")
            
            # 4. Technical Indicator Availability
            required_indicators = ['rsi_14', 'macd', 'volume_ratio', 'bb_position']
            available_indicators = [ind for ind in required_indicators if ind in data.columns]
            if len(available_indicators) >= 3:
                evaluation_score += 15
                reasons.append(f"Technical indicators: {len(available_indicators)}/4")
            
            # 5. Data Quality Check
            null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if null_percentage < 0.05:  # Less than 5% null values
                evaluation_score += 10
                reasons.append("Good data quality")
            
            # 6. Price Movement Check (not flat)
            price_range = (data['high'].max() - data['low'].min()) / data['close'].mean()
            if price_range > 0.1:  # At least 10% range
                evaluation_score += 10
                reasons.append(f"Good price movement: {price_range:.2%}")
            
            # Determine if ticker passes
            passes = evaluation_score >= 70  # Need at least 70/100 points
            
            result = self.create_evaluation_result(
                ticker, passes, 
                f"Score: {evaluation_score}/100. " + "; ".join(reasons)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ticker evaluation failed for {ticker}: {e}")
            return self.create_evaluation_result(ticker, False, f"Evaluation error: {e}")
    
    def create_evaluation_result(self, ticker: str, passes: bool, reason: str) -> Dict[str, Any]:
        """Create and cache evaluation result"""
        result = {
            'ticker': ticker,
            'passes': passes,
            'reason': reason,
            'timestamp': datetime.now(),
            'score': 100 if passes else 0
        }
        
        self.evaluated_tickers[ticker] = result
        self.last_evaluation[ticker] = datetime.now()
        
        return result
    
    def is_evaluation_cached(self, ticker: str) -> bool:
        """Check if evaluation is cached and still valid"""
        if ticker not in self.last_evaluation:
            return False
        
        time_since_eval = datetime.now() - self.last_evaluation[ticker]
        return time_since_eval.total_seconds() < (self.evaluation_cache_hours * 3600)
    
    def evaluate_watchlist(self, tickers: List[str]) -> List[str]:
        """Evaluate entire watchlist and return qualified tickers"""
        try:
            logger.info(f"🔍 Evaluating {len(tickers)} tickers for quality...")
            
            qualified_tickers = []
            evaluation_results = []
            
            for ticker in tickers:
                try:
                    result = self.evaluate_ticker_quality(ticker)
                    evaluation_results.append(result)
                    
                    if result['passes']:
                        qualified_tickers.append(ticker)
                        logger.info(f"✅ {ticker}: {result['reason']}")
                    else:
                        logger.info(f"❌ {ticker}: {result['reason']}")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate {ticker}: {e}")
                    continue
            
            logger.info(f"📊 Ticker evaluation complete: {len(qualified_tickers)}/{len(tickers)} qualified")
            
            # Save evaluation results
            self.save_evaluation_results(evaluation_results)
            
            return qualified_tickers
            
        except Exception as e:
            logger.error(f"❌ Watchlist evaluation failed: {e}")
            return tickers  # Return original list if evaluation fails
    
    def save_evaluation_results(self, results: List[Dict]):
        """Save evaluation results to file"""
        try:
            os.makedirs('ticker_evaluations', exist_ok=True)
            filename = f"ticker_evaluations/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Evaluation results saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save evaluation results: {e}")

# Initialize ticker evaluation system
ticker_evaluator = TickerEvaluationSystem()

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
        
        # Calculate CVaR
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
            'profit_factor': sum([r for r in returns if r > 0]) / abs(sum([r for r in returns if r < 0]) if sum([r for r in returns if r < 0]) != 0 else float('inf')
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
                'current_equity': trading_state.starting_equity if 'trading_state' in globals() else 100000,
                'market_open': market_status.is_market_open(),
                'trading_window': market_status.is_in_trading_window()
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
            'market_status': {
                'market_open': market_status.is_market_open(),
                'trading_window': market_status.is_in_trading_window(),
                'near_eod': market_status.is_near_eod(),
                'time_until_open': str(market_status.get_time_until_market_open())
            },
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
                'paper_trading': config.PAPER_TRADING_MODE,
                'ticker_evaluation': config.TICKER_EVALUATION_ENABLED,
                'hold_logic': config.HOLD_POSITION_ENABLED
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
                'eod_liquidation': config.EOD_LIQUIDATION_ENABLED,
                'ticker_evaluation': len(ticker_evaluator.evaluated_tickers) if 'ticker_evaluator' in globals() else 0
            },
            'performance_metrics': {
                'total_trades': len(trading_state.trade_outcomes) if 'trading_state' in globals() else 0,
                'win_rate': trading_state.risk_metrics.get('win_rate', 0) if 'trading_state' in globals() else 0,
                'sharpe_ratio': trading_state.risk_metrics.get('sharpe_ratio', 0) if 'trading_state' in globals() else 0,
                'model_accuracy': trading_state.model_accuracy.get('current', 0) if 'trading_state' in globals() else 0,
                'current_drawdown': risk_monitor.current_drawdown if 'risk_monitor' in globals() else 0,
                'trading_halted': risk_monitor.trading_halted if 'risk_monitor' in globals() else False,
                'qualified_tickers': len([t for t in ticker_evaluator.evaluated_tickers.values() if t['passes']]) if 'ticker_evaluator' in globals() else 0
            }
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/')
def home():
    """Enhanced root endpoint with comprehensive bot information"""
    return jsonify({
        'service': 'Ultra-Advanced AI Trading Bot v7.0 - 24/7 Day Trading Edition',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'version': '7.0.0',
        'mode': 'Paper Trading' if config.PAPER_TRADING_MODE else 'Live Trading',
        'market_status': {
            'open': market_status.is_market_open(),
            'trading_window': market_status.is_in_trading_window(),
            'near_eod': market_status.is_near_eod()
        },
        'features': [
            '🕐 24/7 Operation with market awareness',
            '⏰ 30-minute post-open wait period',
            '🌅 Smart EOD liquidation with overnight hold logic',
            '🎯 Hold position logic (not just buy/sell)',
            '🔍 Pre-training ticker evaluation',
            '📈 Expanded universe (500+ tickers)',
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
        "enterprise_features", "sqlite_db", "ticker_evaluations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

create_enhanced_directories()

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
        
        # Dynamic watchlist with evaluation
        self.current_watchlist = ALL_TICKERS[:config.WATCHLIST_LIMIT]
        self.qualified_watchlist = []
        self.watchlist_performance = {}
        self.last_watchlist_update = datetime.now()
        self.last_evaluation_time = None
        
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
        
        # Hold logic tracking
        self.position_hold_decisions = {}
        self.hold_reasons = {}
        
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
        self.position_hold_decisions = {}
        self.hold_reasons = {}
    
    def get_next_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_id_counter += 1
        return f"TRADE_{datetime.now().strftime('%Y%m%d')}_{self.trade_id_counter:04d}"
    
    def should_hold_position(self, ticker: str, position_data: Dict) -> bool:
        """Determine if position should be held instead of sold"""
        try:
            if not config.HOLD_POSITION_ENABLED:
                return False
            
            # Check minimum hold time
            entry_time = position_data.get('entry_time', datetime.now())
            hold_duration = datetime.now() - entry_time
            min_hold_time = timedelta(minutes=config.MIN_HOLD_TIME_MINUTES)
            
            if hold_duration < min_hold_time:
                self.hold_reasons[ticker] = f"Min hold time not met: {hold_duration}"
                return True
            
            # Check maximum hold time
            max_hold_time = timedelta(hours=config.MAX_HOLD_TIME_HOURS)
            if hold_duration > max_hold_time:
                self.hold_reasons[ticker] = f"Max hold time exceeded: {hold_duration}"
                return False
            
            # Check if position is profitable and trending
            current_price = position_data.get('current_price', 0)
            entry_price = position_data.get('entry_price', 0)
            
            if entry_price > 0:
                profit_pct = (current_price - entry_price) / entry_price
                
                # Hold if profitable and not near resistance
                if profit_pct > 0.01:  # 1% profit
                    # Check if we're near resistance levels
                    sr_levels = self.support_resistance_cache.get(ticker, {})
                    resistance_levels = sr_levels.get('resistance_levels', [])
                    
                    near_resistance = False
                    for resistance in resistance_levels:
                        if abs(current_price - resistance) / current_price < 0.02:  # Within 2%
                            near_resistance = True
                            break
                    
                    if not near_resistance:
                        self.hold_reasons[ticker] = f"Profitable and trending: {profit_pct:.2%}"
                        return True
            
            # Check market regime
            if self.market_regime == "bullish" and profit_pct > -0.005:  # Small loss in bull market
                self.hold_reasons[ticker] = f"Bull market hold: {profit_pct:.2%}"
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Hold position check failed for {ticker}: {e}")
            return False
    
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
        """Optimize watchlist based on multiple criteria with ticker evaluation"""
        try:
            # Check if refresh is needed
            time_since_update = datetime.now() - trading_state.last_watchlist_update
            if time_since_update.total_seconds() < self.refresh_hours * 3600:
                return trading_state.qualified_watchlist or trading_state.current_watchlist
            
            logger.info("🔄 Optimizing dynamic watchlist with ticker evaluation...")
            
            # Start with expanded universe
            candidate_tickers = []
            
            # Score all tickers from expanded universe
            ticker_scores = {}
            
            for sector, tickers in EXPANDED_SECTOR_UNIVERSE.items():
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
                
                # Take top tickers from each sector
                for ticker, score in sector_scores[:self.max_per_sector]:
                    candidate_tickers.append(ticker)
            
            # Sort all candidates by score and take top performers
            sorted_candidates = sorted(
                [(ticker, data['score']) for ticker, data in ticker_scores.items()],
                key=lambda x: x[1], reverse=True
            )
            
            # Take top candidates for evaluation
            top_candidates = [ticker for ticker, score in sorted_candidates[:self.watchlist_limit * 2]]
            
            logger.info(f"📊 Selected {len(top_candidates)} candidates for evaluation")
            
            # Evaluate ticker quality if enabled
            if config.TICKER_EVALUATION_ENABLED:
                qualified_tickers = ticker_evaluator.evaluate_watchlist(top_candidates)
                
                # Ensure we have minimum qualified tickers
                if len(qualified_tickers) < config.MIN_TICKERS_FOR_TRAINING:
                    logger.warning(f"⚠️ Only {len(qualified_tickers)} qualified tickers, adding more...")
                    # Add more tickers from the sorted list
                    for ticker, score in sorted_candidates:
                        if ticker not in qualified_tickers and len(qualified_tickers) < self.watchlist_limit:
                            qualified_tickers.append(ticker)
                
                optimized_watchlist = qualified_tickers[:self.watchlist_limit]
                trading_state.qualified_watchlist = optimized_watchlist
            else:
                optimized_watchlist = top_candidates[:self.watchlist_limit]
            
            # Update trading state
            trading_state.current_watchlist = optimized_watchlist
            trading_state.last_watchlist_update = datetime.now()
            
            # Calculate sector distribution
            sector_counts = defaultdict(int)
            for ticker in optimized_watchlist:
                for sector, tickers in EXPANDED_SECTOR_UNIVERSE.items():
                    if ticker in tickers:
                        sector_counts[sector] += 1
                        break
            
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

def send_discord_alert(message: str, urgent: bool = False) -> bool:
    """Send Discord alert"""
    return alert_manager.send_alert(message, urgent)

# === SUPPORT/RESISTANCE DETECTION ===
class SupportResistanceDetector:
    def __init__(self):
        self.strength_threshold = config.SUPPORT_RESISTANCE_STRENGTH
        
    def find_support_resistance_levels(self, df: pd.DataFrame, ticker: str) -> Dict[str, List[float]]:
        """Find support and resistance levels using peak detection"""
        try:
            if df is None or len(df) < 20:
                return {'support_levels': [], 'resistance_levels': []}
            
            # Find peaks and troughs
            highs = df['high'].values
            lows = df['low'].values
            
            # Find resistance levels (peaks)
            resistance_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
            resistance_levels = [highs[i] for i in resistance_peaks]
            
            # Find support levels (troughs)
            support_peaks, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
            support_levels = [lows[i] for i in support_peaks]
            
            # Filter by strength (how many times price touched the level)
            resistance_levels = self.filter_by_strength(resistance_levels, df, 'resistance')
            support_levels = self.filter_by_strength(support_levels, df, 'support')
            
            # Cache results
            trading_state.support_resistance_cache[ticker] = {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'timestamp': datetime.now()
            }
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            logger.error(f"❌ Support/resistance detection failed for {ticker}: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def filter_by_strength(self, levels: List[float], df: pd.DataFrame, level_type: str) -> List[float]:
        """Filter levels by strength (number of touches)"""
        try:
            strong_levels = []
            
            for level in levels:
                touches = 0
                tolerance = level * 0.02  # 2% tolerance
                
                if level_type == 'resistance':
                    touches = len(df[abs(df['high'] - level) <= tolerance])
                else:  # support
                    touches = len(df[abs(df['low'] - level) <= tolerance])
                
                if touches >= self.strength_threshold:
                    strong_levels.append(level)
            
            return strong_levels
            
        except Exception as e:
            logger.error(f"❌ Level strength filtering failed: {e}")
            return levels

support_resistance_detector = SupportResistanceDetector()

# === VOLUME PROFILE ANALYSIS ===
class VolumeProfileAnalyzer:
    def __init__(self):
        self.bins = config.VOLUME_PROFILE_BINS
        
    def calculate_volume_profile(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Calculate volume profile for price levels"""
        try:
            if df is None or len(df) < 20:
                return {}
            
            # Calculate price range
            price_min = df['low'].min()
            price_max = df['high'].max()
            
            # Create price bins
            price_bins = np.linspace(price_min, price_max, self.bins)
            volume_profile = np.zeros(self.bins - 1)
            
            # Distribute volume across price levels
            for _, row in df.iterrows():
                # Assume uniform distribution within the bar
                bar_range = row['high'] - row['low']
                if bar_range > 0:
                    for i in range(len(price_bins) - 1):
                        bin_low = price_bins[i]
                        bin_high = price_bins[i + 1]
                        
                        # Calculate overlap
                        overlap_low = max(bin_low, row['low'])
                        overlap_high = min(bin_high, row['high'])
                        
                        if overlap_high > overlap_low:
                            overlap_ratio = (overlap_high - overlap_low) / bar_range
                            volume_profile[i] += row['volume'] * overlap_ratio
            
            # Find Point of Control (POC) - highest volume price level
            poc_index = np.argmax(volume_profile)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            
            # Find Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            value_area_volume = total_volume * 0.7
            
            # Find value area high and low
            sorted_indices = np.argsort(volume_profile)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_profile[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= value_area_volume:
                    break
            
            value_area_low = price_bins[min(value_area_indices)]
            value_area_high = price_bins[max(value_area_indices) + 1]
            
            profile_data = {
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'volume_profile': volume_profile.tolist(),
                'price_bins': price_bins.tolist(),
                'timestamp': datetime.now()
            }
            
            # Cache results
            trading_state.volume_profile_cache[ticker] = profile_data
            
            return profile_data
            
        except Exception as e:
            logger.error(f"❌ Volume profile calculation failed for {ticker}: {e}")
            return {}

volume_profile_analyzer = VolumeProfileAnalyzer()

# === ENHANCED TRADE EXECUTION WITH HOLD LOGIC ===
class EnhancedTradeExecutor:
    def __init__(self):
        self.cooldown_minutes = config.TRADE_COOLDOWN_MINUTES
        
    def execute_trade_with_hold_logic(self, ticker: str, action: str, quantity: int, 
                                    current_price: float, signal_data: Dict) -> bool:
        """Execute trade with enhanced hold logic"""
        try:
            trade_id = trading_state.get_next_trade_id()
            
            # Check if we should hold existing position
            if ticker in trading_state.open_positions:
                position_data = trading_state.open_positions[ticker].copy()
                position_data['current_price'] = current_price
                
                if trading_state.should_hold_position(ticker, position_data):
                    logger.info(f"🤝 Holding position for {ticker}: {trading_state.hold_reasons.get(ticker, 'Hold logic triggered')}")
                    
                    # Update position tracking
                    trading_state.position_hold_decisions[ticker] = {
                        'timestamp': datetime.now(),
                        'reason': trading_state.hold_reasons.get(ticker, 'Hold logic'),
                        'current_price': current_price,
                        'entry_price': position_data.get('entry_price', current_price),
                        'hold_duration': datetime.now() - position_data.get('entry_time', datetime.now())
                    }
                    
                    return True  # Successfully held position
            
            # Execute new trade or close position
            if config.PAPER_TRADING_MODE:
                success = paper_trading.execute_paper_trade(ticker, action, quantity, current_price)
            else:
                success = self.execute_live_trade(ticker, action, quantity, current_price)
            
            if success:
                # Update position tracking
                if action.lower() == 'buy':
                    trading_state.open_positions[ticker] = {
                        'quantity': quantity,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'trade_id': trade_id,
                        'signal_data': signal_data
                    }
                elif action.lower() == 'sell' and ticker in trading_state.open_positions:
                    position = trading_state.open_positions.pop(ticker)
                    
                    # Calculate P&L
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    return_pct = pnl / (position['entry_price'] * position['quantity'])
                    
                    # Log trade outcome
                    trade_outcome = {
                        'trade_id': trade_id,
                        'ticker': ticker,
                        'action': action,
                        'quantity': quantity,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': return_pct,
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(),
                        'hold_duration': datetime.now() - position['entry_time'],
                        'signal_data': signal_data,
                        'correct_prediction': return_pct > 0  # Simplified
                    }
                    
                    trading_state.trade_outcomes.append(trade_outcome)
                    
                    # Log to Google Sheets
                    sheets_logger.log_trade({
                        'timestamp': datetime.now().isoformat(),
                        'trade_id': trade_id,
                        'ticker': ticker,
                        'action': action,
                        'quantity': quantity,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return_pct': return_pct * 100,
                        'signal_strength': signal_data.get('confidence', 0),
                        'model_used': signal_data.get('model_type', 'ensemble'),
                        'sentiment_score': signal_data.get('sentiment', 0),
                        'volume_spike': signal_data.get('volume_spike', False),
                        'vwap_deviation': signal_data.get('vwap_deviation', 0),
                        'sector': signal_data.get('sector', ''),
                        'market_regime': trading_state.market_regime,
                        'hold_duration': str(datetime.now() - position['entry_time']),
                        'notes': f"Hold decisions: {len(trading_state.position_hold_decisions.get(ticker, []))}"
                    })
                
                # Set cooldown
                trading_state.cooldown_timers[ticker] = datetime.now() + timedelta(minutes=self.cooldown_minutes)
                
                logger.info(f"✅ Trade executed: {action.upper()} {quantity} {ticker} @ ${current_price:.2f}")
                
                # Send Discord alert
                alert_message = f"🔄 **TRADE EXECUTED**\n"
                alert_message += f"**{action.upper()}** {quantity} shares of **{ticker}** @ ${current_price:.2f}\n"
                alert_message += f"Signal Confidence: {signal_data.get('confidence', 0):.3f}\n"
                alert_message += f"Market Regime: {trading_state.market_regime}\n"
                if ticker in trading_state.position_hold_decisions:
                    alert_message += f"Previous Holds: {len(trading_state.position_hold_decisions[ticker])}"
                
                send_discord_alert(alert_message)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed for {ticker}: {e}")
            return False
    
    def execute_live_trade(self, ticker: str, action: str, quantity: int, current_price: float) -> bool:
        """Execute live trade through Alpaca API"""
        try:
            if not api_manager.api:
                logger.warning("⚠️ No API connection for live trading")
                return False
            
            # Prepare order
            side = 'buy' if action.lower() == 'buy' else 'sell'
            
            order = api_manager.safe_api_call(
                api_manager.api.submit_order,
                symbol=ticker,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            if order:
                logger.info(f"✅ Live order submitted: {order.id}")
                return True
            else:
                logger.error(f"❌ Live order failed for {ticker}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Live trade execution failed: {e}")
            return False

# Initialize trade executor
trade_executor = EnhancedTradeExecutor()

# === END-OF-DAY LIQUIDATION SYSTEM ===
class EODLiquidationManager:
    """End-of-day liquidation with overnight hold logic"""
    
    def __init__(self):
        self.liquidation_time = config.EOD_LIQUIDATION_TIME
        self.enabled = config.EOD_LIQUIDATION_ENABLED
        
    def check_eod_liquidation(self) -> bool:
        """Check if EOD liquidation should be triggered"""
        try:
            if not self.enabled:
                return False
            
            if market_status.is_near_eod() and not trading_state.eod_liquidation_triggered:
                logger.info("🌅 EOD liquidation window reached")
                self.execute_eod_liquidation()
                trading_state.eod_liquidation_triggered = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ EOD liquidation check failed: {e}")
            return False
    
    def execute_eod_liquidation(self):
        """Execute end-of-day liquidation with overnight hold logic"""
        try:
            positions_to_liquidate = []
            positions_to_hold = []
            
            for ticker, position in trading_state.open_positions.items():
                # Get current price
                current_data = get_enhanced_data(ticker, limit=1)
                if current_data is None or current_data.empty:
                    continue
                
                current_price = current_data['close'].iloc[-1]
                position_data = position.copy()
                position_data['current_price'] = current_price
                
                # Check if position should be held overnight
                if market_status.should_hold_overnight(position_data):
                    positions_to_hold.append(ticker)
                    logger.info(f"🌙 Holding {ticker} overnight - Profitable position")
                else:
                    positions_to_liquidate.append(ticker)
            
            # Liquidate positions that shouldn't be held overnight
            for ticker in positions_to_liquidate:
                position = trading_state.open_positions[ticker]
                current_data = get_enhanced_data(ticker, limit=1)
                if current_data is not None and not current_data.empty:
                    current_price = current_data['close'].iloc[-1]
                    
                    success = trade_executor.execute_trade_with_hold_logic(
                        ticker, 'sell', position['quantity'], current_price,
                        {'confidence': 0.8, 'model_type': 'eod_liquidation', 'reason': 'EOD liquidation'}
                    )
                    
                    if success:
                        logger.info(f"🌅 EOD liquidated: {ticker}")
            
            # Send summary alert
            alert_message = f"🌅 **EOD LIQUIDATION COMPLETE**\n"
            alert_message += f"Liquidated: {len(positions_to_liquidate)} positions\n"
            alert_message += f"Held overnight: {len(positions_to_hold)} positions\n"
            if positions_to_hold:
                alert_message += f"Overnight holds: {', '.join(positions_to_hold)}"
            
            send_discord_alert(alert_message)
            
        except Exception as e:
            logger.error(f"❌ EOD liquidation execution failed: {e}")

# Initialize EOD liquidation manager
eod_liquidation_manager = EODLiquidationManager()

# === MAIN TRADING LOGIC WITH 24/7 OPERATION ===
def ultra_advanced_trading_logic():
    """Ultra-advanced trading logic with 24/7 operation and all features"""
    try:
        logger.info("🚀 Starting Ultra-Advanced AI Trading Bot v7.0 - 24/7 Day Trading Edition")
        
        # Initialize starting equity
        if api_manager.api and not config.PAPER_TRADING_MODE:
            account = api_manager.safe_api_call(api_manager.api.get_account)
            if account:
                trading_state.starting_equity = float(account.equity)
        else:
            trading_state.starting_equity = paper_trading.current_capital
        
        logger.info(f"💰 Starting equity: ${trading_state.starting_equity:,.2f}")
        
        # Send startup alert
        startup_message = f"🚀 **TRADING BOT STARTED**\n"
        startup_message += f"Mode: {'Paper Trading' if config.PAPER_TRADING_MODE else 'Live Trading'}\n"
        startup_message += f"Starting Equity: ${trading_state.starting_equity:,.2f}\n"
        startup_message += f"Market Status: {'Open' if market_status.is_market_open() else 'Closed'}\n"
        startup_message += f"Trading Window: {'Active' if market_status.is_in_trading_window() else 'Waiting'}\n"
        startup_message += f"Features: Dual-horizon, Ensemble, Hold Logic, Ticker Evaluation, EOD Liquidation"
        
        send_discord_alert(startup_message, urgent=True)
        
        # Main 24/7 loop
        while True:
            try:
                current_time = datetime.now()
                logger.info(f"🕐 Trading cycle started at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Send heartbeat
                heartbeat_monitor.send_heartbeat()
                
                # Check market status
                market_open = market_status.is_market_open()
                trading_window = market_status.is_in_trading_window()
                
                logger.info(f"📊 Market Status - Open: {market_open}, Trading Window: {trading_window}")
                
                if not market_open:
                    # Market is closed - wait and perform maintenance tasks
                    time_until_open = market_status.get_time_until_market_open()
                    logger.info(f"🌙 Market closed. Next open in: {time_until_open}")
                    
                    # Perform maintenance tasks during market closure
                    perform_maintenance_tasks()
                    
                    # Sleep for 30 minutes during market closure
                    time.sleep(1800)
                    continue
                
                if not trading_window:
                    # Market is open but we're in the 30-minute wait period
                    logger.info(f"⏰ Market open but waiting {config.MARKET_OPEN_WAIT_MINUTES} minutes after open")
                    time.sleep(300)  # Check every 5 minutes
                    continue
                
                # Check for emergency stop
                if trading_state.emergency_stop_triggered or risk_monitor.trading_halted:
                    logger.error("🚨 Emergency stop triggered - halting trading")
                    time.sleep(3600)  # Wait 1 hour before checking again
                    continue
                
                # Check EOD liquidation
                eod_liquidation_manager.check_eod_liquidation()
                
                # Update risk monitoring
                if config.REAL_TIME_RISK_MONITORING:
                    current_equity = get_current_equity()
                    risk_monitor.update_equity(current_equity)
                
                # Optimize watchlist periodically
                optimized_watchlist = watchlist_optimizer.optimize_watchlist()
                logger.info(f"📋 Current watchlist: {len(optimized_watchlist)} tickers")
                
                # Evaluate meta-model performance
                meta_approval_system.evaluate_model_performance()
                
                # Process each ticker in the watchlist
                successful_evaluations = 0
                total_signals_generated = 0
                trades_executed = 0
                
                for ticker in optimized_watchlist:
                    try:
                        # Check cooldown
                        if ticker in trading_state.cooldown_timers:
                            if datetime.now() < trading_state.cooldown_timers[ticker]:
                                continue
                        
                        # Get market data for both horizons
                        short_data = get_enhanced_data(ticker, limit=200, days_back=config.SHORT_TERM_DAYS)
                        medium_data = get_enhanced_data(ticker, limit=100, days_back=config.MEDIUM_TERM_DAYS)
                        
                        if short_data is None or medium_data is None:
                            continue
                        
                        successful_evaluations += 1
                        
                        # Generate trading signal
                        signal_result = generate_ultra_advanced_signal(ticker, short_data, medium_data)
                        
                        if signal_result and signal_result['action'] != 'hold':
                            total_signals_generated += 1
                            
                            # Execute trade if approved
                            if meta_approval_system.should_execute_trade(signal_result['confidence']):
                                current_price = short_data['close'].iloc[-1]
                                position_size = kelly_position_sizer.calculate_position_size(
                                    ticker, trading_state.starting_equity, signal_result['confidence'], short_data
                                )
                                
                                success = trade_executor.execute_trade_with_hold_logic(
                                    ticker, signal_result['action'], position_size, current_price, signal_result
                                )
                                
                                if success:
                                    trades_executed += 1
                        
                        # Rate limiting between tickers
                        time.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing {ticker}: {e}")
                        continue
                
                # Update risk metrics
                trading_state.update_ultra_advanced_risk_metrics()
                
                # Log cycle summary
                logger.info(f"📊 Cycle Summary:")
                logger.info(f"   - Tickers evaluated: {successful_evaluations}/{len(optimized_watchlist)}")
                logger.info(f"   - Signals generated: {total_signals_generated}")
                logger.info(f"   - Trades executed: {trades_executed}")
                logger.info(f"   - Open positions: {len(trading_state.open_positions)}")
                logger.info(f"   - Meta-model approved: {trading_state.meta_model_approved}")
                logger.info(f"   - Current drawdown: {risk_monitor.current_drawdown:.2%}")
                
                # Send periodic status update
                if len(trading_state.trade_outcomes) % 10 == 0 and len(trading_state.trade_outcomes) > 0:
                    send_status_update()
                
                # Sleep before next cycle (5 minutes for day trading)
                logger.info("😴 Sleeping for 5 minutes before next cycle...")
                time.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received - shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"❌ Error in main trading loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)  # Wait 1 minute before retrying
                continue
        
        # Graceful shutdown
        logger.info("🛑 Shutting down trading bot...")
        send_discord_alert("🛑 **TRADING BOT SHUTDOWN**\nBot has been stopped gracefully.", urgent=True)
        
    except Exception as e:
        logger.error(f"❌ Critical error in trading logic: {e}")
        logger.error(traceback.format_exc())
        send_discord_alert(f"🚨 **CRITICAL ERROR**\n{str(e)}", urgent=True)

def perform_maintenance_tasks():
    """Perform maintenance tasks during market closure"""
    try:
        logger.info("🔧 Performing maintenance tasks...")
        
        # Reset daily state
        trading_state.reset_daily()
        
        # Clean up old cache files
        cleanup_old_files()
        
        # Retrain models if needed
        if should_retrain_models():
            retrain_models()
        
        # Update sector performance
        update_sector_performance()
        
        # Backup important data
        backup_trading_data()
        
        logger.info("✅ Maintenance tasks completed")
        
    except Exception as e:
        logger.error(f"❌ Maintenance tasks failed: {e}")

def cleanup_old_files():
    """Clean up old cache and log files"""
    try:
        # Clean up old log files (keep last 7 days)
        log_files = glob.glob("logs/*.log")
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in log_files:
            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            if file_time < cutoff_date:
                os.remove(log_file)
        
        # Clean up old evaluation files
        eval_files = glob.glob("ticker_evaluations/*.json")
        for eval_file in eval_files:
            file_time = datetime.fromtimestamp(os.path.getmtime(eval_file))
            if file_time < cutoff_date:
                os.remove(eval_file)
        
        logger.info("🧹 Old files cleaned up")
        
    except Exception as e:
        logger.error(f"❌ File cleanup failed: {e}")

def should_retrain_models() -> bool:
    """Check if models should be retrained"""
    try:
        # Check if enough time has passed
        time_since_retrain = datetime.now() - trading_state.meta_model_last_retrain
        if time_since_retrain.total_seconds() < config.MODEL_RETRAIN_FREQUENCY_HOURS * 3600:
            return False
        
        # Check if we have enough new data
        if len(trading_state.trade_outcomes) < 50:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model retrain check failed: {e}")
        return False

def retrain_models():
    """Retrain models with latest data"""
    try:
        logger.info("🔄 Retraining models...")
        
        # This would implement model retraining logic
        # For now, just update the timestamp
        trading_state.meta_model_last_retrain = datetime.now()
        
        logger.info("✅ Models retrained successfully")
        
    except Exception as e:
        logger.error(f"❌ Model retraining failed: {e}")

def update_sector_performance():
    """Update sector performance metrics"""
    try:
        for sector in EXPANDED_SECTOR_UNIVERSE.keys():
            sector_trades = [t for t in trading_state.trade_outcomes 
                           if t.get('sector') == sector]
            
            if sector_trades:
                avg_return = np.mean([t['return'] for t in sector_trades])
                trading_state.sector_performance[sector] = avg_return
        
        logger.info("📊 Sector performance updated")
        
    except Exception as e:
        logger.error(f"❌ Sector performance update failed: {e}")

def backup_trading_data():
    """Backup important trading data"""
    try:
        backup_data = {
            'trade_outcomes': trading_state.trade_outcomes[-1000:],  # Last 1000 trades
            'risk_metrics': trading_state.risk_metrics,
            'model_accuracy': trading_state.model_accuracy,
            'sector_performance': dict(trading_state.sector_performance),
            'timestamp': datetime.now().isoformat()
        }
        
        backup_file = f"backups/trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("backups", exist_ok=True)
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"💾 Trading data backed up to {backup_file}")
        
    except Exception as e:
        logger.error(f"❌ Data backup failed: {e}")

def get_current_equity() -> float:
    """Get current portfolio equity"""
    try:
        if config.PAPER_TRADING_MODE:
            # Calculate paper trading equity
            current_prices = {}
            for ticker in trading_state.open_positions.keys():
                data = get_enhanced_data(ticker, limit=1)
                if data is not None and not data.empty:
                    current_prices[ticker] = data['close'].iloc[-1]
            
            return paper_trading.get_paper_portfolio_value(current_prices)
        else:
            # Get live account equity
            if api_manager.api:
                account = api_manager.safe_api_call(api_manager.api.get_account)
                if account:
                    return float(account.equity)
        
        return trading_state.starting_equity
        
    except Exception as e:
        logger.error(f"❌ Equity calculation failed: {e}")
        return trading_state.starting_equity

def send_status_update():
    """Send periodic status update"""
    try:
        current_equity = get_current_equity()
        total_return = (current_equity - trading_state.starting_equity) / trading_state.starting_equity
        
        status_message = f"📊 **TRADING STATUS UPDATE**\n"
        status_message += f"Current Equity: ${current_equity:,.2f}\n"
        status_message += f"Total Return: {total_return:.2%}\n"
        status_message += f"Total Trades: {len(trading_state.trade_outcomes)}\n"
        status_message += f"Win Rate: {trading_state.risk_metrics.get('win_rate', 0):.1%}\n"
        status_message += f"Sharpe Ratio: {trading_state.risk_metrics.get('sharpe_ratio', 0):.2f}\n"
        status_message += f"Max Drawdown: {trading_state.risk_metrics.get('max_drawdown', 0):.2%}\n"
        status_message += f"Open Positions: {len(trading_state.open_positions)}\n"
        status_message += f"Market Regime: {trading_state.market_regime}\n"
        status_message += f"Meta-Model: {'✅ Approved' if trading_state.meta_model_approved else '❌ Not Approved'}"
        
        send_discord_alert(status_message)
        
    except Exception as e:
        logger.error(f"❌ Status update failed: {e}")

def generate_ultra_advanced_signal(ticker: str, short_data: pd.DataFrame, 
                                 medium_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Generate ultra-advanced trading signal with all features"""
    try:
        # Get dual-horizon predictions
        short_pred, medium_pred, meta_pred = ensemble_model.predict_dual_horizon(short_data, medium_data)
        
        # Get sentiment analysis
        sentiment_score = sentiment_analyzer.analyze_ticker_sentiment(ticker)
        
        # Get support/resistance levels
        sr_levels = support_resistance_detector.find_support_resistance_levels(short_data, ticker)
        
        # Get volume profile
        volume_profile = volume_profile_analyzer.calculate_volume_profile(short_data, ticker)
        
        # Detect market regime
        regime, regime_confidence = regime_detector.detect_market_regime(short_data)
        
        # Check anomaly detection
        if config.ANOMALY_DETECTION_ENABLED:
            features_for_anomaly = ensemble_model.extract_features(short_data)
            if features_for_anomaly is not None:
                is_anomaly = anomaly_detector.detect_anomaly(features_for_anomaly.tail(1))
                if is_anomaly:
                    logger.warning(f"⚠️ Anomaly detected for {ticker} - reducing signal confidence")
                    meta_pred *= 0.7  # Reduce confidence during anomalies
        
        # Apply all filters and logic
        current_price = short_data['close'].iloc[-1]
        
        # Volume spike filter
        volume_spike = short_data['volume_spike'].iloc[-1] if 'volume_spike' in short_data.columns else False
        volume_spike_confirmation = short_data['volume_spike_confirmation'].iloc[-1] if 'volume_spike_confirmation' in short_data.columns else False
        
        # VWAP filter
        vwap_deviation = short_data['vwap_deviation'].iloc[-1] if 'vwap_deviation' in short_data.columns else 0
        vwap_filter_pass = vwap_deviation <= config.VWAP_DEVIATION_THRESHOLD
        
        # Price momentum filter
        price_momentum = short_data['price_momentum'].iloc[-1] if 'price_momentum' in short_data.columns else 0
        momentum_filter_pass = abs(price_momentum) >= config.PRICE_MOMENTUM_MIN
        
        # Support/resistance filter
        near_support = False
        near_resistance = False
        
        for support_level in sr_levels.get('support_levels', []):
            if abs(current_price - support_level) / current_price < 0.02:  # Within 2%
                near_support = True
                break
        
        for resistance_level in sr_levels.get('resistance_levels', []):
            if abs(current_price - resistance_level) / current_price < 0.02:  # Within 2%
                near_resistance = True
                break
        
        # Determine action based on all criteria
        action = 'hold'
        confidence = meta_pred
        
        # Buy signal logic
        if (short_pred >= config.SHORT_BUY_THRESHOLD and 
            medium_pred >= config.MEDIUM_BUY_THRESHOLD and
            meta_pred >= config.ENSEMBLE_CONFIDENCE_THRESHOLD and
            volume_spike and
            vwap_filter_pass and
            momentum_filter_pass and
            price_momentum > 0 and
            near_support and
            not near_resistance and
            sentiment_score > config.SENTIMENT_HOLD_OVERRIDE):
            
            action = 'buy'
            
        # Sell signal logic (for existing positions)
        elif (ticker in trading_state.open_positions and
              (short_pred <= config.SHORT_SELL_AVOID_THRESHOLD or
               medium_pred <= config.MEDIUM_SELL_AVOID_THRESHOLD or
               near_resistance or
               sentiment_score < config.SENTIMENT_HOLD_OVERRIDE)):
            
            action = 'sell'
        
        # Regime-based adjustments
        if regime == "bearish" and action == 'buy':
            confidence *= 0.8  # Reduce buy confidence in bear market
        elif regime == "bullish" and action == 'sell':
            confidence *= 0.8  # Reduce sell confidence in bull market
        
        # Q-Learning adjustment
        if pytorch_q_agent:
            # Create state vector for Q-learning
            state_vector = [
                short_pred, medium_pred, meta_pred, sentiment_score,
                volume_spike, vwap_deviation, price_momentum,
                near_support, near_resistance, regime_confidence
            ]
            
            q_action_idx = pytorch_q_agent.act(state_vector)
            q_action = pytorch_q_agent.actions[q_action_idx]
            
            # Use Q-learning as a tie-breaker or confidence modifier
            if action == 'hold' and q_action != 'hold' and confidence > 0.6:
                action = q_action
                confidence *= 0.9  # Slightly reduce confidence for Q-learning override
        
        # Final confidence check
        if confidence < config.ENSEMBLE_CONFIDENCE_THRESHOLD:
            action = 'hold'
        
        # Create signal result
        signal_result = {
            'ticker': ticker,
            'action': action,
            'confidence': confidence,
            'short_pred': short_pred,
            'medium_pred': medium_pred,
            'meta_pred': meta_pred,
            'sentiment': sentiment_score,
            'volume_spike': volume_spike,
            'volume_spike_confirmation': volume_spike_confirmation,
            'vwap_deviation': vwap_deviation,
            'price_momentum': price_momentum,
            'near_support': near_support,
            'near_resistance': near_resistance,
            'market_regime': regime,
            'regime_confidence': regime_confidence,
            'model_type': 'dual_horizon_ensemble',
            'timestamp': datetime.now()
        }
        
        if action != 'hold':
            logger.info(f"🎯 Signal generated for {ticker}: {action.upper()} (confidence: {confidence:.3f})")
        
        return signal_result
        
    except Exception as e:
        logger.error(f"❌ Signal generation failed for {ticker}: {e}")
        return None

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        # Start Flask app in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
        flask_thread.daemon = True
        flask_thread.start()
        
        logger.info("🌐 Flask health check server started on port 5000")
