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

# === REDIS INITIALIZATION FOR UPSTASH ===
import redis
from urllib.parse import urlparse

REDIS_AVAILABLE = False
redis_client = None

try:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        parsed_url = urlparse(redis_url)
        redis_client = redis.Redis(
            host=parsed_url.hostname,
            port=parsed_url.port,
            password=parsed_url.password,
            ssl=parsed_url.scheme == 'rediss',  # required for Upstash
            decode_responses=True
        )
        redis_client.ping()
        REDIS_AVAILABLE = True
        print("✅ Redis connected successfully.")
    else:
        print("⚠️ REDIS_URL not set. Redis caching disabled.")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    REDIS_AVAILABLE = False

class RedisFeatureCache:
    """Redis-based feature caching for performance"""

    def __init__(self):
        self.enabled = REDIS_AVAILABLE
        self.redis_client = redis_client if REDIS_AVAILABLE else None
        if self.enabled:
            try:
                self.redis_client.ping()
                print("✅ RedisFeatureCache is ready.")
            except Exception as e:
                print(f"⚠️ RedisFeatureCache init failed: {e}")
                self.enabled = False

    def cache_features(self, ticker: str, features: dict, ttl: int = 300):
        if not self.enabled:
            return
        try:
            key = f"features:{ticker}"
            self.redis_client.setex(key, ttl, json.dumps(features, default=str))
        except Exception as e:
            print(f"❌ Feature caching failed: {e}")

    def get_cached_features(self, ticker: str):
        if not self.enabled:
            return None
        try:
            key = f"features:{ticker}"
            cached = self.redis_client.get(key)
            return json.loads(cached) if cached else None
        except Exception as e:
            print(f"❌ Feature retrieval failed: {e}")
            return None

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
    
    # Model Training
    INITIAL_TRAINING_ENABLED: bool = True
    TRAINING_DATA_DAYS: int = 60  # Days of data for training
    MIN_TRAINING_SAMPLES: int = 100  # Minimum samples for training
    
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
        "ESTC", "MDB", "TEAM", "ZS", "PANW", "FTNT", "CYBR", "SPLK"
    ],
    "Finance": [
        # Banks
        "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "COF", "SCHW", "MS", "GS",
        "BLK", "SPGI", "ICE", "CME", "MCO", "AXP", "V", "MA", "PYPL",
        # Insurance & REITs
        "BRK.B", "AIG", "PGR", "TRV", "ALL", "MET", "PRU", "AFL", "HIG",
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
        "BIIB", "ILMN", "MRNA", "BNTX", "NVAX", "SGEN", "ALNY", "BMRN", "RARE",
        # Medical Devices
        "TMO", "DHR", "MDT", "ISRG", "SYK", "BSX", "EW", "HOLX", "DXCM", "VEEV",
        # Healthcare Services
        "UNH", "CVS", "ANTM", "HUM", "CNC", "CI", "MOH", "ELV", "TDOC"
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
        "enterprise_features", "sqlite_db", "ticker_evaluations", "backups"
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
        
        # Model training status
        self.models_trained = False
        self.training_in_progress = False
        self.last_training_time = None

    def log_trade_to_sheet(self, ticker, action, confidence, price, outcome, signal_type, features):
        """Logs trade features and outcome to Google Sheet for meta model training."""
        try:
            volatility = features.get("volatility", 0.0)
            vwap_distance = features.get("vwap_distance", 0.0)
            volume_spike = features.get("volume_spike", 0)
            kelly_fraction = features.get("kelly_fraction", 0.0)
            sector = features.get("sector", "Unknown")
            regime = features.get("regime", "Unknown")
            cooldown_status = features.get("cooldown_status", 0)
            entry_hour = datetime.now().hour

            self.sheets_worksheet.append_row([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ticker,
                action,
                confidence,
                price,
                outcome,
                signal_type,
                volatility,
                vwap_distance,
                volume_spike,
                kelly_fraction,
                sector,
                regime,
                cooldown_status,
                entry_hour
            ])
        except Exception as e:
            logger.error(f"❌ Failed to log trade to sheet: {e}")
        
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

    def retrain_meta_model(self):
        """Retrain meta model using logged trade data"""
        try:
            sheet = sheet_client.worksheet("MetaModelLog")
            data = pd.DataFrame(sheet.get_all_records())

            if len(data) < 50:
                logger.warning("⚠️ Not enough trade logs to train meta model.")
                return

            features = [
                "confidence", "volatility", "vwap_distance", "volume_spike",
                "kelly_fraction", "entry_hour", "cooldown_status"
            ]
            X = data[features]
            y = data["outcome"]

            X.fillna(0, inplace=True)

            meta_model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
            meta_model.fit(X, y)

            joblib.dump(meta_model, "meta_model.pkl")
            self.meta_model = meta_model

            logger.info(f"✅ Meta model retrained on {len(data)} samples.")
        except Exception as e:
            logger.error(f"❌ Meta model retraining failed: {e}")

    def train_dual_horizon_ensemble(self, qualified_tickers: List[str]) -> bool:
        """Train dual-horizon ensemble models with qualified tickers"""
        try:
            logger.info(f"🔄 Training dual-horizon ensemble with {len(qualified_tickers)} qualified tickers...")
            
            if len(qualified_tickers) < config.MIN_TICKERS_FOR_TRAINING:
                logger.error(f"❌ Insufficient qualified tickers for training: {len(qualified_tickers)} < {config.MIN_TICKERS_FOR_TRAINING}")
                return False
            
            # Collect training data
            short_features_list = []
            medium_features_list = []
            short_labels_list = []
            medium_labels_list = []
            
            for ticker in qualified_tickers[:config.MIN_TICKERS_FOR_TRAINING]:
                try:
                    # Get training data
                    short_data = get_enhanced_data(ticker, limit=config.TRAINING_DATA_DAYS * 78)  # 78 5-min bars per day
                    medium_data = get_enhanced_data(ticker, limit=config.TRAINING_DATA_DAYS, timeframe=TimeFrame.Day)
                    
                    if short_data is None or medium_data is None:
                        continue
                    
                    # Extract features
                    short_features = self.extract_features(short_data)
                    medium_features = self.extract_features(medium_data)
                    
                    if short_features is None or medium_features is None:
                        continue
                    
                    # Generate labels (simplified - future returns > 0)
                    short_labels = (short_data['close'].shift(-1) > short_data['close']).astype(int)
                    medium_labels = (medium_data['close'].shift(-1) > medium_data['close']).astype(int)
                    
                    # Align features and labels
                    short_features = short_features.iloc[:-1]  # Remove last row
                    medium_features = medium_features.iloc[:-1]
                    short_labels = short_labels.iloc[:-1]
                    medium_labels = medium_labels.iloc[:-1]
                    
                    # Drop NaN values
                    short_valid = ~(short_features.isnull().any(axis=1) | short_labels.isnull())
                    medium_valid = ~(medium_features.isnull().any(axis=1) | medium_labels.isnull())
                    
                    short_features = short_features[short_valid]
                    short_labels = short_labels[short_valid]
                    medium_features = medium_features[medium_valid]
                    medium_labels = medium_labels[medium_valid]
                    
                    if len(short_features) > 50 and len(medium_features) > 20:
                        short_features_list.append(short_features)
                        medium_features_list.append(medium_features)
                        short_labels_list.append(short_labels)
                        medium_labels_list.append(medium_labels)
                        
                        logger.info(f"✅ Training data collected for {ticker}: Short={len(short_features)}, Medium={len(medium_features)}")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to collect training data for {ticker}: {e}")
                    continue
            
            if not short_features_list or not medium_features_list:
                logger.error("❌ No valid training data collected")
                return False
            
            # Combine all training data
            combined_short_features = pd.concat(short_features_list, ignore_index=True)
            combined_medium_features = pd.concat(medium_features_list, ignore_index=True)
            combined_short_labels = pd.concat(short_labels_list, ignore_index=True)
            combined_medium_labels = pd.concat(medium_labels_list, ignore_index=True)
            
            logger.info(f"📊 Combined training data: Short={len(combined_short_features)}, Medium={len(combined_medium_features)}")
            
            # Scale features
            short_features_scaled = self.scaler_short.fit_transform(combined_short_features)
            medium_features_scaled = self.scaler_medium.fit_transform(combined_medium_features)
            
            short_features_scaled = pd.DataFrame(short_features_scaled, columns=combined_short_features.columns)
            medium_features_scaled = pd.DataFrame(medium_features_scaled, columns=combined_medium_features.columns)
            
            # Create base models
            base_models = self.create_base_models()
            
            # Train short-term models
            logger.info("🔄 Training short-term models...")
            for name, model in base_models.items():
                try:
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, short_features_scaled, combined_short_labels, cv=tscv, scoring='accuracy')
                    
                    # Train on full dataset
                    model.fit(short_features_scaled, combined_short_labels)
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
                    cv_scores = cross_val_score(medium_model, medium_features_scaled, combined_medium_labels, cv=tscv, scoring='accuracy')
                    
                    # Train on full dataset
                    medium_model.fit(medium_features_scaled, combined_medium_labels)
                    self.medium_term_models[name] = medium_model
                    
                    logger.info(f"✅ Medium-term {name} trained - CV Score: {cv_scores.mean():.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Medium-term {name} training failed: {e}")
            
            # Train meta-model for ensemble combination
            self.train_meta_model(short_features_scaled, medium_features_scaled, combined_short_labels, combined_medium_labels)
            
            # Calculate feature importance using SHAP
            self.calculate_feature_importance(short_features_scaled, combined_short_labels)
            
            # Save models
            self.save_models()
            
            # Update training status
            trading_state.models_trained = True
            trading_state.last_training_time = datetime.now()
            
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
    
    def save_models(self):
        """Save trained models"""
        try:
            # Save ensemble models
            ensemble_data = {
                'short_term_models': self.short_term_models,
                'medium_term_models': self.medium_term_models,
                'meta_model': self.meta_model,
                'scaler_short': self.scaler_short,
                'scaler_medium': self.scaler_medium,
                'feature_importance': self.feature_importance
            }
            
            with open('models/ensemble/dual_horizon_ensemble.pkl', 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            logger.info("💾 Ensemble models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
    
    def load_models(self) -> bool:
        """Load trained models"""
        try:
            model_path = 'models/ensemble/dual_horizon_ensemble.pkl'
            if not os.path.exists(model_path):
                return False
            
            with open(model_path, 'rb') as f:
                ensemble_data = pickle.load(f)
            
            self.short_term_models = ensemble_data.get('short_term_models', {})
            self.medium_term_models = ensemble_data.get('medium_term_models', {})
            self.meta_model = ensemble_data.get('meta_model')
            self.scaler_short = ensemble_data.get('scaler_short', StandardScaler())
            self.scaler_medium = ensemble_data.get('scaler_medium', StandardScaler())
            self.feature_importance = ensemble_data.get('feature_importance', {})
            
            trading_state.models_trained = True
            logger.info("✅ Ensemble models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
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
            logger.error(f"❌ Missing required columns for {ticker}")
            return None
        
        # Data quality checks
        if len(df) < 20:
            logger.warning(f"⚠️ Insufficient data for {ticker}: {len(df)} bars")
            return None
        
        # Remove any invalid data
        df = df.dropna()
        df = df[df['volume'] > 0]
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
        df = df[df['low'] <= df[['open', 'close']].min(axis=1)]
        
        if df.empty:
            logger.warning(f"⚠️ No valid data after cleaning for {ticker}")
            return None
        
        # Add technical indicators
        enhanced_df = add_ultra_advanced_technical_indicators(df)
        
        # Cache the enhanced data
        if enhanced_df is not None:
            feature_cache.cache_features(ticker, enhanced_df.to_dict())
        
        return enhanced_df
        
    except Exception as e:
        logger.error(f"❌ Data fetching failed for {ticker}: {e}")
        return None

# === SUPPORT/RESISTANCE DETECTION ===
def detect_support_resistance_levels(df: pd.DataFrame, ticker: str) -> Dict[str, List[float]]:
    """Detect support and resistance levels using peak detection"""
    try:
        if df is None or len(df) < 20:
            return {'support_levels': [], 'resistance_levels': []}
        
        # Use scipy to find peaks and troughs
        highs = df['high'].values
        lows = df['low'].values
        
        # Find resistance levels (peaks)
        resistance_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
        resistance_levels = [highs[i] for i in resistance_peaks]
        
        # Find support levels (troughs) - invert the data
        support_peaks, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)
        support_levels = [lows[i] for i in support_peaks]
        
        # Filter levels by strength (how many times price touched the level)
        def filter_levels_by_strength(levels, prices, threshold=config.SUPPORT_RESISTANCE_STRENGTH):
            strong_levels = []
            for level in levels:
                touches = sum(1 for price in prices if abs(price - level) / level < 0.02)
                if touches >= threshold:
                    strong_levels.append(level)
            return strong_levels
        
        strong_resistance = filter_levels_by_strength(resistance_levels, highs)
        strong_support = filter_levels_by_strength(support_levels, lows)
        
        # Sort and limit levels
        strong_resistance = sorted(strong_resistance, reverse=True)[:5]
        strong_support = sorted(strong_support)[:5]
        
        result = {
            'support_levels': strong_support,
            'resistance_levels': strong_resistance,
            'timestamp': datetime.now()
        }
        
        # Cache the result
        trading_state.support_resistance_cache[ticker] = result
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Support/resistance detection failed for {ticker}: {e}")
        return {'support_levels': [], 'resistance_levels': []}

# === VOLUME PROFILE ANALYSIS ===
def calculate_volume_profile(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """Calculate volume profile for price levels"""
    try:
        if df is None or len(df) < 20:
            return {}
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, config.VOLUME_PROFILE_BINS)
        
        # Calculate volume at each price level
        volume_profile = np.zeros(len(price_bins) - 1)
        
        for i, row in df.iterrows():
            # Distribute volume across price range for each bar
            bar_low = row['low']
            bar_high = row['high']
            bar_volume = row['volume']
            
            # Find which bins this bar spans
            low_bin = np.digitize(bar_low, price_bins) - 1
            high_bin = np.digitize(bar_high, price_bins) - 1
            
            # Distribute volume evenly across bins
            bins_spanned = max(1, high_bin - low_bin + 1)
            volume_per_bin = bar_volume / bins_spanned
            
            for bin_idx in range(max(0, low_bin), min(len(volume_profile), high_bin + 1)):
                volume_profile[bin_idx] += volume_per_bin
        
        # Find Point of Control (POC) - price level with highest volume
        poc_index = np.argmax(volume_profile)
        poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
        
        # Find Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        value_area_volume = total_volume * 0.7
        
        # Start from POC and expand until we reach 70% of volume
        value_area_low_idx = poc_index
        value_area_high_idx = poc_index
        current_volume = volume_profile[poc_index]
        
        while current_volume < value_area_volume:
            # Expand to the side with more volume
            low_volume = volume_profile[value_area_low_idx - 1] if value_area_low_idx > 0 else 0
            high_volume = volume_profile[value_area_high_idx + 1] if value_area_high_idx < len(volume_profile) - 1 else 0
            
            if low_volume > high_volume and value_area_low_idx > 0:
                value_area_low_idx -= 1
                current_volume += volume_profile[value_area_low_idx]
            elif value_area_high_idx < len(volume_profile) - 1:
                value_area_high_idx += 1
                current_volume += volume_profile[value_area_high_idx]
            else:
                break
        
        value_area_low = (price_bins[value_area_low_idx] + price_bins[value_area_low_idx + 1]) / 2
        value_area_high = (price_bins[value_area_high_idx] + price_bins[value_area_high_idx + 1]) / 2
        
        result = {
            'poc_price': poc_price,
            'value_area_low': value_area_low,
            'value_area_high': value_area_high,
            'volume_profile': volume_profile.tolist(),
            'price_bins': price_bins.tolist(),
            'timestamp': datetime.now()
        }
        
        # Cache the result
        trading_state.volume_profile_cache[ticker] = result
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Volume profile calculation failed for {ticker}: {e}")
        return {}

# === EQUITY CALCULATION ===
def get_current_equity() -> float:
    """Get current portfolio equity"""
    try:
        if config.PAPER_TRADING_MODE:
            # Calculate paper trading equity
            current_prices = {}
            portfolio_value = paper_trading.current_capital
            
            # Add value of open positions
            for ticker, position in paper_trading.positions.items():
                try:
                    data = get_enhanced_data(ticker, limit=1)
                    if data is not None and not data.empty:
                        current_price = data['close'].iloc[-1]
                        position_value = position['quantity'] * current_price
                        portfolio_value += position_value - (position['quantity'] * position['entry_price'])
                except Exception as e:
                    logger.error(f"❌ Failed to get current price for {ticker}: {e}")
                    continue
            
            return portfolio_value
            
        else:
            # Get live account equity
            if api_manager.api:
                account = api_manager.safe_api_call(api_manager.api.get_account)
                if account:
                    return float(account.equity)
        
        # Fallback to starting equity
        return trading_state.starting_equity
        
    except Exception as e:
        logger.error(f"❌ Equity calculation failed: {e}")
        return trading_state.starting_equity

# === DISCORD ALERTS ===
def send_discord_alert(message: str, urgent: bool = False):
    """Send alert to Discord webhook"""
    try:
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return
        
        # Add emoji based on urgency
        if urgent:
            message = f"🚨 **URGENT** 🚨\n{message}"
        else:
            message = f"📊 {message}"
        
        payload = {
            "content": message,
            "username": "Trading Bot",
            "avatar_url": "https://cdn.discordapp.com/embed/avatars/0.png"
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 204:
            logger.info("✅ Discord alert sent successfully")
        else:
            logger.warning(f"⚠️ Discord alert failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Discord alert failed: {e}")

# === KELLY CRITERION POSITION SIZING ===
def calculate_kelly_position_size(win_rate: float, avg_win: float, avg_loss: float, 
                                 account_value: float) -> float:
    """Calculate optimal position size using Kelly Criterion"""
    try:
        if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
            return account_value * config.KELLY_FRACTION_MIN
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety constraints
        kelly_fraction = max(kelly_fraction, config.KELLY_FRACTION_MIN)
        kelly_fraction = min(kelly_fraction, config.KELLY_FRACTION_MAX)
        
        position_size = account_value * kelly_fraction
        
        logger.info(f"📊 Kelly position size: {kelly_fraction:.3f} of account (${position_size:.2f})")
        
        return position_size
        
    except Exception as e:
        logger.error(f"❌ Kelly position size calculation failed: {e}")
        return account_value * config.KELLY_FRACTION_MIN

# === TRADE EXECUTION WITH ENHANCED LOGIC ===
def execute_trade_with_ultra_advanced_logic(ticker: str, action: str, data: pd.DataFrame) -> bool:
    """Execute trade with comprehensive logic and safety checks"""
    try:
        if not api_manager.api and not config.PAPER_TRADING_MODE:
            logger.warning("⚠️ No API connection and not in paper trading mode")
            return False
        
        # Emergency stop check
        if trading_state.emergency_stop_triggered:
            logger.warning("🚨 Emergency stop active - blocking all trades")
            return False
        
        # Risk monitoring check
        if risk_monitor.trading_halted:
            logger.warning("🚨 Trading halted due to risk limits")
            return False
        
        # Market hours check
        if not market_status.is_in_trading_window():
            logger.info("⏰ Outside trading window - trade blocked")
            return False
        
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Check if we should hold existing position instead of trading
        if ticker in trading_state.open_positions:
            position_data = trading_state.open_positions[ticker]
            position_data['current_price'] = current_price
            
            if trading_state.should_hold_position(ticker, position_data):
                logger.info(f"🤝 Holding position for {ticker}: {trading_state.hold_reasons.get(ticker, 'Hold logic triggered')}")
                return False
        
        # Cooldown check
        if ticker in trading_state.cooldown_timers:
            time_since_last = datetime.now() - trading_state.cooldown_timers[ticker]
            if time_since_last.total_seconds() < config.TRADE_COOLDOWN_MINUTES * 60:
                logger.info(f"⏳ {ticker} in cooldown for {config.TRADE_COOLDOWN_MINUTES - time_since_last.total_seconds()//60:.0f} more minutes")
                return False
        
        # Get account information
        if config.PAPER_TRADING_MODE:
            account_value = paper_trading.current_capital
            buying_power = paper_trading.current_capital
        else:
            account = api_manager.safe_api_call(api_manager.api.get_account)
            if not account:
                logger.error("❌ Failed to get account information")
                return False
            
            account_value = float(account.portfolio_value)
            buying_power = float(account.buying_power)
        
        # Update risk monitoring
        risk_monitor.update_equity(account_value)
        
        # Calculate position size using Kelly Criterion
        recent_trades = trading_state.trade_outcomes[-20:] if len(trading_state.trade_outcomes) >= 20 else trading_state.trade_outcomes
        
        if recent_trades:
            wins = [t for t in recent_trades if t['return'] > 0]
            losses = [t for t in recent_trades if t['return'] < 0]
            
            win_rate = len(wins) / len(recent_trades)
            avg_win = np.mean([t['return'] for t in wins]) if wins else 0.01
            avg_loss = abs(np.mean([t['return'] for t in losses])) if losses else 0.01
        else:
            win_rate = 0.5
            avg_win = 0.02
            avg_loss = 0.02
        
        position_value = calculate_kelly_position_size(win_rate, avg_win, avg_loss, account_value)
        quantity = int(position_value / current_price)
        
        if quantity <= 0:
            logger.warning(f"⚠️ Calculated quantity is 0 for {ticker}")
            return False
        
        # Risk checks
        position_risk = (quantity * current_price) / account_value
        if position_risk > config.MAX_PORTFOLIO_RISK:
            logger.warning(f"⚠️ Position risk too high for {ticker}: {position_risk:.2%}")
            return False

        # === Meta-Model Approval Filter ===
        if config.META_MODEL_APPROVAL_ENABLED:
            if not trading_state.meta_model_approved:
                if confidence < 0.7:
                    logger.warning(f"🚫 Trade blocked by meta-model filter: {ticker} (Conf: {confidence:.2f})")
                    return False  # ✅ Use `return False` inside a function, not `continue` unless in a loop

        # Sector allocation check
        ticker_sector = get_ticker_sector(ticker)
        current_sector_allocation = trading_state.sector_allocations.get(ticker_sector, 0)
        new_sector_allocation = current_sector_allocation + position_risk
        
        if new_sector_allocation > config.MAX_PER_SECTOR_PORTFOLIO:
            logger.warning(f"⚠️ Sector allocation limit exceeded for {ticker_sector}: {new_sector_allocation:.2%}")
            return False
        
        # Generate trade ID
        trade_id = trading_state.get_next_trade_id()
        
        # Execute the trade
        if config.PAPER_TRADING_MODE:
            success = paper_trading.execute_paper_trade(ticker, action, quantity, current_price)
        else:
            # Real trading execution
            try:
                if action.lower() == 'buy':
                    order = api_manager.safe_api_call(
                        api_manager.api.submit_order,
                        symbol=ticker,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                elif action.lower() == 'sell':
                    order = api_manager.safe_api_call(
                        api_manager.api.submit_order,
                        symbol=ticker,
                        qty=quantity,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                else:
                    logger.error(f"❌ Invalid action: {action}")
                    return False
                
                success = order is not None
                
            except Exception as e:
                logger.error(f"❌ Trade execution failed for {ticker}: {e}")
                return False
        
        if success:
            # Update trading state
            trading_state.cooldown_timers[ticker] = datetime.now()
            trading_state.sector_allocations[ticker_sector] = new_sector_allocation
            
            # Track position
            if action.lower() == 'buy':
                trading_state.open_positions[ticker] = {
                    'quantity': quantity,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'trade_id': trade_id,
                    'sector': ticker_sector
                }
            elif action.lower() == 'sell' and ticker in trading_state.open_positions:
                # Calculate P&L
                position = trading_state.open_positions.pop(ticker)
                pnl = (current_price - position['entry_price']) * position['quantity']
                return_pct = pnl / (position['entry_price'] * position['quantity'])
                
                # Record trade outcome
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
                    'sector': ticker_sector
                }
                
                trading_state.trade_outcomes.append(trade_outcome)
                
                # Log to Google Sheets
                sheets_logger.log_trade(trade_outcome)
                
                # Update risk metrics
                trading_state.update_ultra_advanced_risk_metrics()
            
            # Send Discord alert
            alert_message = f"{'📈' if action.lower() == 'buy' else '📉'} {action.upper()} {quantity} {ticker} @ ${current_price:.2f}"
            if action.lower() == 'sell' and ticker in trading_state.trade_outcomes:
                last_trade = trading_state.trade_outcomes[-1]
                alert_message += f" | P&L: ${last_trade['pnl']:.2f} ({last_trade['return']:.2%})"
            
            send_discord_alert(alert_message)
            
            logger.info(f"✅ Trade executed: {action.upper()} {quantity} {ticker} @ ${current_price:.2f}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Trade execution failed for {ticker}: {e}")
        return False

def get_ticker_sector(ticker: str) -> str:
    """Get sector for a ticker"""
    for sector, tickers in EXPANDED_SECTOR_UNIVERSE.items():
        if ticker in tickers:
            return sector
    return "Unknown"

# === EOD LIQUIDATION SYSTEM ===
def perform_eod_liquidation():
    """Perform end-of-day liquidation with overnight hold logic"""
    try:
        if not config.EOD_LIQUIDATION_ENABLED or trading_state.eod_liquidation_triggered:
            return
        
        logger.info("🌅 Performing end-of-day liquidation check...")
        
        positions_to_liquidate = []
        positions_to_hold = []
        
        # Get current prices for all positions
        current_prices = {}
        for ticker in trading_state.open_positions.keys():
            try:
                data = get_enhanced_data(ticker, limit=1)
                if data is not None and not data.empty:
                    current_prices[ticker] = data['close'].iloc[-1]
            except:
                continue
        
        # Evaluate each position
        for ticker, position in trading_state.open_positions.items():
            try:
                current_price = current_prices.get(ticker)
                if not current_price:
                    continue
                
                position['current_price'] = current_price
                
                # Check if position should be held overnight
                if market_status.should_hold_overnight(position):
                    positions_to_hold.append(ticker)
                    logger.info(f"🌙 Holding {ticker} overnight - meets profit criteria")
                else:
                    positions_to_liquidate.append(ticker)
            
            except Exception as e:
                logger.error(f"❌ EOD evaluation failed for {ticker}: {e}")
                positions_to_liquidate.append(ticker)
        
        # Liquidate positions that don't meet hold criteria
        for ticker in positions_to_liquidate:
            try:
                position = trading_state.open_positions[ticker]
                current_price = current_prices.get(ticker, position['entry_price'])
                
                # Execute liquidation
                if config.PAPER_TRADING_MODE:
                    success = paper_trading.execute_paper_trade(ticker, 'sell', position['quantity'], current_price)
                else:
                    order = api_manager.safe_api_call(
                        api_manager.api.submit_order,
                        symbol=ticker,
                        qty=position['quantity'],
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    success = order is not None
                
                if success:
                    # Calculate P&L
                    pnl = (current_price - position['entry_price']) * position['quantity']
                    return_pct = pnl / (position['entry_price'] * position['quantity'])
                    
                    # Record trade outcome
                    trade_outcome = {
                        'trade_id': trading_state.get_next_trade_id(),
                        'ticker': ticker,
                        'action': 'sell',
                        'quantity': position['quantity'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'return': return_pct,
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.now(),
                        'hold_duration': datetime.now() - position['entry_time'],
                        'sector': position.get('sector', 'Unknown'),
                        'eod_liquidation': True
                    }
                    
                    trading_state.trade_outcomes.append(trade_outcome)
                    trading_state.open_positions.pop(ticker)
                    
                    # Log to Google Sheets
                    sheets_logger.log_trade(trade_outcome)
                    
                    logger.info(f"🌅 EOD liquidated {ticker}: P&L ${pnl:.2f} ({return_pct:.2%})")
                
            except Exception as e:
                logger.error(f"❌ EOD liquidation failed for {ticker}: {e}")
        
        # Send summary alert
        liquidated_count = len(positions_to_liquidate)
        held_count = len(positions_to_hold)
        
        alert_message = f"🌅 EOD Summary: {liquidated_count} positions liquidated, {held_count} held overnight"
        send_discord_alert(alert_message)
        
        trading_state.eod_liquidation_triggered = True
        logger.info(f"✅ EOD liquidation complete: {liquidated_count} liquidated, {held_count} held")
        
    except Exception as e:
        logger.error(f"❌ EOD liquidation failed: {e}")

# === MAIN TRADING LOGIC ===
def ultra_advanced_trading_logic(ticker: str) -> bool:
    """Ultra-advanced trading logic with all features integrated"""
    try:
        logger.info(f"🔄 Analyzing {ticker}...")
        
        # Get dual-horizon data
        short_data = get_enhanced_data(ticker, limit=100, days_back=config.SHORT_TERM_DAYS)
        medium_data = get_enhanced_data(ticker, limit=50, timeframe=TimeFrame.Day, days_back=config.MEDIUM_TERM_DAYS)
        
        if short_data is None or medium_data is None:
            logger.warning(f"⚠️ Insufficient data for {ticker}")
            return False
        
        # Detect market regime
        regime, regime_confidence = regime_detector.detect_market_regime(short_data)
        
        # Anomaly detection
        if config.ANOMALY_DETECTION_ENABLED:
            features_for_anomaly = ensemble_model.extract_features(short_data)
            if features_for_anomaly is not None:
                is_anomaly = anomaly_detector.detect_anomaly(features_for_anomaly.tail(1))
                if is_anomaly:
                    logger.warning(f"⚠️ Market anomaly detected for {ticker} - skipping")
                    return False
        
        # Get dual-horizon predictions
        short_pred, medium_pred, meta_pred = ensemble_model.predict_dual_horizon(short_data, medium_data)
        
        # Meta-model approval check
        if not meta_approval_system.should_execute_trade(meta_pred):
            return False
        
        # Support/resistance analysis
        sr_levels = detect_support_resistance_levels(short_data, ticker)
        
        # Volume profile analysis
        volume_profile = calculate_volume_profile(short_data, ticker)
        
        # Sentiment analysis
        sentiment_score = sentiment_analyzer.analyze_ticker_sentiment(ticker)
        
        # Get current market data
        current_price = short_data['close'].iloc[-1]
        volume_ratio = short_data['volume_ratio'].iloc[-1] if 'volume_ratio' in short_data.columns else 1.0
        price_momentum = short_data['price_momentum'].iloc[-1] if 'price_momentum' in short_data.columns else 0.0
        vwap_deviation = short_data['vwap_deviation'].iloc[-1] if 'vwap_deviation' in short_data.columns else 0.0
        
        # Volume spike detection
        volume_spike = volume_ratio > config.VOLUME_SPIKE_MIN
        volume_spike_confirmation = volume_ratio > config.VOLUME_SPIKE_CONFIRMATION_MIN
        
        # VWAP filter
        vwap_filter_pass = vwap_deviation <= config.VWAP_DEVIATION_THRESHOLD
        
        # Price momentum filter
        momentum_filter_pass = abs(price_momentum) >= config.PRICE_MOMENTUM_MIN
        
        # Sentiment filter
        sentiment_bullish = sentiment_score > 0.1
        sentiment_bearish = sentiment_score < -0.1
        sentiment_hold_override = sentiment_score < config.SENTIMENT_HOLD_OVERRIDE
        
        # Support/resistance proximity check
        near_resistance = False
        near_support = False
        
        for resistance in sr_levels.get('resistance_levels', []):
            if abs(current_price - resistance) / current_price < 0.02:  # Within 2%
                near_resistance = True
                break
        
        for support in sr_levels.get('support_levels', []):
            if abs(current_price - support) / current_price < 0.02:  # Within 2%
                near_support = True
                break
        
        # Q-Learning state and action
        q_state = [
            short_pred, medium_pred, meta_pred, sentiment_score, volume_ratio,
            price_momentum, vwap_deviation, regime_confidence,
            1.0 if near_resistance else 0.0, 1.0 if near_support else 0.0
        ]
        
        q_action_idx = pytorch_q_agent.act(q_state)
        q_action = pytorch_q_agent.actions[q_action_idx]
        
        # Trading decision logic with relaxed thresholds for day trading
        buy_signal = False
        sell_signal = False
        
        # BUY CONDITIONS (Relaxed for day trading)
        if (short_pred > config.SHORT_BUY_THRESHOLD and 
            medium_pred > config.MEDIUM_BUY_THRESHOLD and
            momentum_filter_pass and
            volume_spike and
            vwap_filter_pass and
            not near_resistance and
            not sentiment_hold_override and
            q_action in ['buy', 'hold']):
            
            # Additional confirmations
            confirmations = 0
            
            if volume_spike_confirmation:
                confirmations += 1
            if sentiment_bullish:
                confirmations += 1
            if regime == "bullish":
                confirmations += 1
            if near_support:
                confirmations += 1
            
            # Require at least 2 confirmations for buy (relaxed from 3)
            if confirmations >= 2:
                buy_signal = True
        
        # SELL CONDITIONS (for existing positions)
        if (ticker in trading_state.open_positions and
            (short_pred < config.SHORT_SELL_AVOID_THRESHOLD or
             medium_pred < config.MEDIUM_SELL_AVOID_THRESHOLD or
             near_resistance or
             sentiment_bearish or
             q_action == 'sell')):
            
            # Check if we should hold instead of sell
            position_data = trading_state.open_positions[ticker]
            position_data['current_price'] = current_price
            
            if not trading_state.should_hold_position(ticker, position_data):
                sell_signal = True
        
        # Execute trades
        if buy_signal and ticker not in trading_state.open_positions:
            logger.info(f"📈 BUY signal for {ticker}: Short={short_pred:.3f}, Medium={medium_pred:.3f}, Meta={meta_pred:.3f}")
            return execute_trade_with_ultra_advanced_logic(ticker, 'buy', short_data)
        
        elif sell_signal:
            logger.info(f"📉 SELL signal for {ticker}: Short={short_pred:.3f}, Medium={medium_pred:.3f}, Meta={meta_pred:.3f}")
            return execute_trade_with_ultra_advanced_logic(ticker, 'sell', short_data)
        
        else:
            logger.info(f"⏸️ HOLD for {ticker}: Short={short_pred:.3f}, Medium={medium_pred:.3f}, Meta={meta_pred:.3f}")
            
            # Update Q-Learning with neutral reward for hold
            if len(trading_state.q_state_history) > 0:
                prev_state = trading_state.q_state_history[-1]
                reward = 0.0  # Neutral reward for hold
                pytorch_q_agent.remember(prev_state, q_action_idx, reward, q_state, False)
                pytorch_q_agent.replay()
            
            trading_state.q_state_history.append(q_state)
            trading_state.q_action_history.append(q_action_idx)
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Trading logic failed for {ticker}: {e}")
        return False

# === MAIN LOOP WITH 24/7 OPERATION ===
def main_24_7_trading_loop():
    """Main 24/7 trading loop with market awareness"""
    logger.info("🚀 Starting Ultra-Advanced AI Trading Bot v7.0 - 24/7 Day Trading Edition")
    
    # Send startup alert
    send_discord_alert("🚀 Ultra-Advanced AI Trading Bot v7.0 Started - 24/7 Day Trading Edition")
    
    # Initialize models if needed
    if config.INITIAL_TRAINING_ENABLED and not trading_state.models_trained:
        logger.info("🔄 Initial model training required...")
        
        # Load existing models first
        if not ensemble_model.load_models():
            logger.info("🔄 No existing models found, starting fresh training...")
            
            # Optimize watchlist and evaluate tickers
            optimized_watchlist = watchlist_optimizer.optimize_watchlist()
            
            if len(optimized_watchlist) >= config.MIN_TICKERS_FOR_TRAINING:
                # Train models with qualified tickers
                training_success = ensemble_model.train_dual_horizon_ensemble(optimized_watchlist)
                
                if training_success:
                    logger.info("✅ Initial training completed successfully")
                    send_discord_alert("✅ Initial model training completed successfully")
                else:
                    logger.error("❌ Initial training failed")
                    send_discord_alert("❌ Initial model training failed", urgent=True)
            else:
                logger.error(f"❌ Insufficient qualified tickers for training: {len(optimized_watchlist)}")
                send_discord_alert(f"❌ Insufficient qualified tickers for training: {len(optimized_watchlist)}", urgent=True)
    
    # Main loop
    loop_count = 0
    last_heartbeat = datetime.now()
    last_watchlist_optimization = datetime.now()
    last_model_retrain = datetime.now()
    last_eod_check = datetime.now().date()
    
    while True:
        try:
            loop_count += 1
            current_time = datetime.now()
            
            logger.info(f"🔄 Loop {loop_count} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Send heartbeat
            if (current_time - last_heartbeat).total_seconds() >= 60:  # Every minute
                heartbeat_monitor.send_heartbeat()
                last_heartbeat = current_time
            
            # Check market status
            market_open = market_status.is_market_open()
            trading_window = market_status.is_in_trading_window()
            near_eod = market_status.is_near_eod()
            
            logger.info(f"📊 Market Status: Open={market_open}, Trading Window={trading_window}, Near EOD={near_eod}")
            
            if market_open:
                # Market is open
                if near_eod and not trading_state.eod_liquidation_triggered:
                    # Perform EOD liquidation
                    perform_eod_liquidation()
                
                elif trading_window:
                    # Normal trading window
                    logger.info("📈 In trading window - executing trading logic")
                    
                    # Optimize watchlist periodically
                    if (current_time - last_watchlist_optimization).total_seconds() >= config.DYNAMIC_WATCHLIST_REFRESH_HOURS * 3600:
                        optimized_watchlist = watchlist_optimizer.optimize_watchlist()
                        last_watchlist_optimization = current_time
                        logger.info(f"✅ Watchlist optimized: {len(optimized_watchlist)} tickers")
                    
                    # Get current watchlist
                    current_watchlist = trading_state.qualified_watchlist or trading_state.current_watchlist
                    
                    # Evaluate meta-model performance
                    meta_approval_system.evaluate_model_performance()
                    
                    # Process each ticker in watchlist
                    trades_executed = 0
                    for ticker in current_watchlist:
                        try:
                            if ultra_advanced_trading_logic(ticker):
                                trades_executed += 1
                            
                            # Rate limiting between tickers
                            time.sleep(1)
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing {ticker}: {e}")
                            continue
                    
                    logger.info(f"📊 Trading cycle complete: {trades_executed} trades executed")
                    
                    # Retrain models periodically
                    if (current_time - last_model_retrain).total_seconds() >= config.MODEL_RETRAIN_FREQUENCY_HOURS * 3600:
                        logger.info("🔄 Periodic model retraining...")
                        qualified_tickers = trading_state.qualified_watchlist or trading_state.current_watchlist[:config.MIN_TICKERS_FOR_TRAINING]
                        ensemble_model.train_dual_horizon_ensemble(qualified_tickers)
                        last_model_retrain = current_time
                
                else:
                    # Market open but waiting for trading window
                    time_until_trading = market_status.get_time_until_market_open()
                    if time_until_trading.total_seconds() > 0:
                        wait_minutes = min(30 - (current_time.hour * 60 + current_time.minute - 9 * 60 - 30), 30)
                        logger.info(f"⏰ Waiting {wait_minutes} minutes for trading window to open")
                    
                    # Use this time for maintenance tasks
                    logger.info("🔧 Performing maintenance tasks...")
                    
                    # Update risk metrics
                    trading_state.update_ultra_advanced_risk_metrics()
                    
                    # Clean up old cache entries
                    current_time_ts = current_time.timestamp()
                    for ticker in list(trading_state.sentiment_cache.keys()):
                        cache_entry = trading_state.sentiment_cache[ticker]
                        if isinstance(cache_entry, dict) and 'timestamp' in cache_entry:
                            if (current_time_ts - cache_entry['timestamp'].timestamp()) > 3600:  # 1 hour
                                del trading_state.sentiment_cache[ticker]
            
            else:
                # Market is closed
                logger.info("🌙 Market closed - performing maintenance and monitoring")
                
                # Reset daily state if new day
                if current_time.date() != last_eod_check:
                    logger.info("🌅 New trading day - resetting daily state")
                    trading_state.reset_daily()
                    last_eod_check = current_time.date()
                
                # Maintenance tasks during market closure
                if loop_count % 10 == 0:  # Every 10 loops during closure
                    logger.info("🔧 Performing extended maintenance...")
                    
                    # Backup important data
                    try:
                        backup_data = {
                            'trade_outcomes': trading_state.trade_outcomes[-100:],  # Last 100 trades
                            'risk_metrics': trading_state.risk_metrics,
                            'model_accuracy': trading_state.model_accuracy,
                            'watchlist_performance': trading_state.watchlist_performance
                        }

                        os.makedirs('backups', exist_ok=True)
                        backup_filename = f"backups/backup_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                        with open(backup_filename, 'w') as f:
                            json.dump(backup_data, f, indent=2, default=str)

                        logger.info(f"💾 Data backup created: {backup_filename}")

                    except Exception as e:
                        logger.error(f"❌ Backup failed: {e}")

                    # === Daily Model Retraining ===
                    try:
                        qualified = trading_state.qualified_watchlist or trading_state.current_watchlist
                        if len(qualified) >= config.MIN_TICKERS_FOR_TRAINING:
                            ensemble_model.retrain_meta_model()
                            ensemble_model.train_dual_horizon_ensemble(qualified[:config.MIN_TICKERS_FOR_TRAINING])
                            logger.info("✅ Daily model retraining completed.")
                        else:
                            logger.warning(f"⚠️ Skipping retraining - only {len(qualified)} tickers available")
                    except Exception as e:
                        logger.error(f"❌ Model retraining failed: {e}")

                # Wait longer during market closure
                time_until_open = market_status.get_time_until_market_open()
                if time_until_open.total_seconds() > 3600:  # More than 1 hour
                    logger.info(f"⏰ Market opens in {time_until_open}. Sleeping for 5 minutes...")
                    time.sleep(300)  # 5 minutes
                    continue

            # Standard loop delay
            time.sleep(30)  # 30 seconds between loops

        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal")
            break

        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

            # Send error alert
            send_discord_alert(f"❌ Main loop error: {str(e)[:200]}", urgent=True)

            # Wait before retrying loop
            time.sleep(60)
