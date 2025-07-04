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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict, deque
import json
import warnings
import sys
import traceback
import threading
from scipy import stats
from scipy.signal import find_peaks
from flask import Flask, jsonify, render_template_string
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
import optuna
import websocket
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ENTERPRISE IMPORTS - All SIP Data Providers
try:
    import polygon
    from polygon import RESTClient as PolygonClient
    POLYGON_AVAILABLE = True
except ImportError:
    polygon = None
    PolygonClient = None
    POLYGON_AVAILABLE = False
    print("⚠️ Polygon not installed. Install with: pip install polygon-api-client")

try:
    from iexfinance.stocks import Stock, get_historical_data
    from iexfinance import get_stats_intraday
    IEX_AVAILABLE = True
except ImportError:
    Stock = None
    get_historical_data = None
    get_stats_intraday = None
    IEX_AVAILABLE = False
    print("⚠️ IEX Finance not installed. Install with: pip install iexfinance")

try:
    import intrinio_sdk
    from intrinio_sdk.rest import ApiException
    INTRINIO_AVAILABLE = True
except ImportError:
    intrinio_sdk = None
    ApiException = None
    INTRINIO_AVAILABLE = False
    print("⚠️ Intrinio SDK not installed. Install with: pip install intrinio-sdk")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    talib = None
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not installed. Some advanced technical indicators may be limited.")

try:
    from scipy.optimize import minimize, differential_evolution
    import cvxpy as cp
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("⚠️ Some optimization libraries not installed. Portfolio optimization may be limited.")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Install with: pip install yfinance")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not installed. Feature caching will be disabled.")

try:
    import gym
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("⚠️ Reinforcement Learning libraries not available. RL features disabled.")

try:
    import ccxt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ CCXT not installed. Crypto trading features disabled.")

warnings.filterwarnings('ignore')
load_dotenv()

# === ULTRA-COMPREHENSIVE CONFIGURATION ===
@dataclass
class UltraAdvancedTradingConfig:
    """Ultra-comprehensive configuration for all enterprise features"""
    
    # === CORE SYSTEM SETTINGS ===
    PAPER_TRADING_MODE: bool = True
    INITIAL_CAPITAL: float = 100000.0
    MAX_POSITIONS: int = 15
    MAX_POSITION_SIZE: float = 0.08  # 8% of portfolio per position
    MIN_POSITION_SIZE: float = 0.01  # 1% minimum
    
    # === API CREDENTIALS ===
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    GOOGLE_SHEETS_CREDENTIALS: str = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "google_sheets_credentials.json")
    GOOGLE_SHEETS_NAME: str = os.getenv("GOOGLE_SHEETS_NAME", "Ultra Trading Bot Logs")
    
    # === ENTERPRISE FEATURE 1: LATENCY-LEVEL SIP DATA ===
    SIP_DATA_ENABLED: bool = True
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    IEX_API_KEY: str = os.getenv("IEX_API_KEY", "")
    INTRINIO_API_KEY: str = os.getenv("INTRINIO_API_KEY", "")
    TARGET_LATENCY_MS: int = 150
    SIP_FALLBACK_ENABLED: bool = True
    SIP_QUALITY_THRESHOLD: float = 0.95
    REAL_TIME_QUOTES_ENABLED: bool = True
    LEVEL_2_DATA_ENABLED: bool = True
    
    # === ENTERPRISE FEATURE 2: DYNAMIC HYPERPARAMETER TUNING ===
    AUTOML_ENABLED: bool = True
    OPTUNA_N_TRIALS: int = 200
    HYPERPARAMETER_RETUNE_FREQUENCY: int = 12  # Hours
    OPTUNA_STUDY_NAME: str = "ultra_trading_optimization"
    OPTUNA_STORAGE: str = "sqlite:///optuna_ultra.db"
    BAYESIAN_OPTIMIZATION_ENABLED: bool = True
    HYPERPARAMETER_BOUNDS: Dict = None
    
    # === ENTERPRISE FEATURE 3: SLIPPAGE & EXECUTION COST MODELING ===
    SLIPPAGE_MODELING_ENABLED: bool = True
    EXPECTED_SLIPPAGE_BPS: float = 4.5
    EXECUTION_COST_TRACKING: bool = True
    SLIPPAGE_HISTORY_WINDOW: int = 200
    MARKET_IMPACT_MODELING: bool = True
    LIQUIDITY_COST_MODELING: bool = True
    
    # === ENTERPRISE FEATURE 4: LIVE TRADE FEEDBACK LOOP ===
    FEEDBACK_LOOP_ENABLED: bool = True
    META_MODEL_RETRAIN_FREQUENCY: int = 25  # Trades
    FEEDBACK_WEIGHT_DECAY: float = 0.92
    ONLINE_LEARNING_ENABLED: bool = True
    ADAPTIVE_LEARNING_RATE: bool = True
    
    # === ENTERPRISE FEATURE 5: ADVANCED MARKET REGIME DETECTION ===
    REGIME_MODELS_ENABLED: bool = True
    VOLATILITY_REGIME_THRESHOLD: float = 0.018
    REGIME_DETECTION_LOOKBACK: int = 60
    REGIME_CONFIDENCE_THRESHOLD: float = 0.65
    HIDDEN_MARKOV_MODELS: bool = True
    REGIME_SWITCHING_MODELS: bool = True
    
    # === ENTERPRISE FEATURE 6: LIVE MONITORING DASHBOARD ===
    DASHBOARD_ENABLED: bool = True
    DASHBOARD_PORT: int = 8501
    DASHBOARD_UPDATE_FREQUENCY: int = 15  # Seconds
    DASHBOARD_HISTORY_WINDOW: int = 2000
    REAL_TIME_CHARTS: bool = True
    ADVANCED_ANALYTICS: bool = True
    
    # === ENTERPRISE FEATURE 7: PORTFOLIO OPTIMIZATION WITH SECTOR ROTATION ===
    SECTOR_ROTATION_ENABLED: bool = True
    PORTFOLIO_OPTIMIZATION_FREQUENCY: int = 4  # Hours
    MIN_SECTOR_ALLOCATION: float = 0.03
    MAX_SECTOR_ALLOCATION: float = 0.35
    REBALANCE_THRESHOLD: float = 0.08
    MEAN_REVERSION_ENABLED: bool = True
    MOMENTUM_ROTATION_ENABLED: bool = True
    
    # === ENTERPRISE FEATURE 8: POSITION SCALING VIA CONFIDENCE ===
    POSITION_SCALING_ENABLED: bool = True
    CONFIDENCE_SCALING_THRESHOLD: float = 0.82
    MAX_POSITION_SCALING: float = 3.0
    PYRAMIDING_ENABLED: bool = True
    PYRAMIDING_PROFIT_THRESHOLD: float = 0.015
    DYNAMIC_POSITION_SIZING: bool = True
    
    # === ENTERPRISE FEATURE 9: RISK-AWARE REINFORCEMENT LEARNING ===
    RISK_AWARE_RL_ENABLED: bool = True
    DRAWDOWN_PENALTY_WEIGHT: float = 2.5
    VOLATILITY_PENALTY_WEIGHT: float = 0.8
    SHARPE_BONUS_WEIGHT: float = 0.4
    RL_MEMORY_SIZE: int = 100000
    RL_BATCH_SIZE: int = 128
    RL_LEARNING_RATE: float = 0.0003
    
    # === ENTERPRISE FEATURE 10: MULTI-TIMEFRAME AGGREGATION ===
    MULTI_TIMEFRAME_ENABLED: bool = True
    TIMEFRAMES: List[str] = None
    MTF_CONFIRMATION_THRESHOLD: float = 0.75
    MTF_WEIGHT_1MIN: float = 0.25
    MTF_WEIGHT_5MIN: float = 0.35
    MTF_WEIGHT_15MIN: float = 0.25
    MTF_WEIGHT_DAILY: float = 0.15
    
    # === ADVANCED RISK MANAGEMENT ===
    MAX_DAILY_DRAWDOWN: float = 0.04
    MAX_TOTAL_DRAWDOWN: float = 0.12
    EMERGENCY_STOP_DRAWDOWN: float = 0.18
    POSITION_SIZE_KELLY_ENABLED: bool = True
    KELLY_FRACTION_CAP: float = 0.2
    VAR_LIMIT: float = 0.03  # 3% VaR limit
    CORRELATION_LIMIT: float = 0.7
    SECTOR_CONCENTRATION_LIMIT: float = 0.4
    
    # === ULTRA-ADVANCED MODEL SETTINGS ===
    MODEL_RETRAIN_FREQUENCY: int = 8  # Hours
    MIN_TRAINING_SAMPLES: int = 2000
    ENSEMBLE_MODELS: List[str] = None
    FEATURE_SELECTION_ENABLED: bool = True
    FEATURE_IMPORTANCE_THRESHOLD: float = 0.005
    CROSS_VALIDATION_FOLDS: int = 5
    MODEL_VALIDATION_ENABLED: bool = True
    
    # === MARKET DATA & ANALYSIS ===
    DATA_LOOKBACK_DAYS: int = 365
    TECHNICAL_INDICATORS_ENABLED: bool = True
    SENTIMENT_ANALYSIS_ENABLED: bool = True
    NEWS_SENTIMENT_WEIGHT: float = 0.18
    SOCIAL_SENTIMENT_ENABLED: bool = True
    EARNINGS_ANALYSIS_ENABLED: bool = True
    
    # === EXECUTION & ORDERS ===
    ORDER_TYPE: str = "market"
    TIME_IN_FORCE: str = "day"
    EXECUTION_DELAY_MS: int = 50
    SMART_ORDER_ROUTING: bool = True
    ICEBERG_ORDERS_ENABLED: bool = True
    TWAP_EXECUTION_ENABLED: bool = True
    
    # === MONITORING & ALERTS ===
    HEALTH_CHECK_ENABLED: bool = True
    HEALTH_CHECK_PORT: int = 5000
    HEARTBEAT_FREQUENCY: int = 45  # Seconds
    LOG_LEVEL: str = "INFO"
    ALERT_THRESHOLDS: Dict = None
    
    # === CACHING & PERFORMANCE ===
    REDIS_ENABLED: bool = True
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    CACHE_TTL: int = 180  # 3 minutes
    PARALLEL_PROCESSING: bool = True
    MAX_WORKERS: int = 8
    
    # === TRADING LOGIC PARAMETERS ===
    DUAL_HORIZON_ENABLED: bool = True
    SHORT_TERM_DAYS: int = 2
    MEDIUM_TERM_DAYS: int = 12
    LONG_TERM_DAYS: int = 30
    SHORT_TERM_WEIGHT: float = 0.4
    MEDIUM_TERM_WEIGHT: float = 0.35
    LONG_TERM_WEIGHT: float = 0.25
    
    # Signal thresholds
    SHORT_BUY_THRESHOLD: float = 0.58
    SHORT_SELL_THRESHOLD: float = 0.42
    MEDIUM_BUY_THRESHOLD: float = 0.62
    MEDIUM_SELL_THRESHOLD: float = 0.38
    LONG_BUY_THRESHOLD: float = 0.65
    LONG_SELL_THRESHOLD: float = 0.35
    
    # Technical filters
    PRICE_MOMENTUM_MIN: float = 0.002
    VOLUME_SPIKE_MIN: float = 1.4
    VOLUME_SPIKE_CONFIRMATION_MIN: float = 2.0
    SENTIMENT_HOLD_OVERRIDE: float = -0.25
    VWAP_DEVIATION_THRESHOLD: float = 0.02
    RSI_OVERBOUGHT: float = 75
    RSI_OVERSOLD: float = 25
    
    # === WATCHLIST & UNIVERSE ===
    MAX_WATCHLIST_SIZE: int = 100
    DYNAMIC_WATCHLIST_ENABLED: bool = True
    WATCHLIST_REFRESH_HOURS: int = 6
    UNIVERSE_FILTERS_ENABLED: bool = True
    MIN_MARKET_CAP: float = 1e9  # $1B
    MIN_DAILY_VOLUME: int = 1000000
    MIN_PRICE: float = 10.0
    MAX_PRICE: float = 1000.0
    
    # === CRYPTO TRADING (if enabled) ===
    CRYPTO_ENABLED: bool = False
    CRYPTO_EXCHANGES: List[str] = None
    CRYPTO_PAIRS: List[str] = None
    
    # === BACKTESTING ===
    BACKTESTING_ENABLED: bool = True
    BACKTEST_START_DATE: str = "2023-01-01"
    BACKTEST_END_DATE: str = "2024-01-01"
    WALK_FORWARD_ANALYSIS: bool = True
    
    # === MINIMUM TICKERS FOR TRAINING ===
    MIN_TICKERS_FOR_TRAINING: int = 10

    def __post_init__(self):
        """Initialize default values for complex types"""
        if self.TIMEFRAMES is None:
            self.TIMEFRAMES = ['1min', '5min', '15min', '1day']
        
        if self.ENSEMBLE_MODELS is None:
            self.ENSEMBLE_MODELS = [
                'xgboost', 'random_forest', 'gradient_boosting',
                'logistic_regression', 'svm', 'neural_network'
            ]
        
        if self.HYPERPARAMETER_BOUNDS is None:
            self.HYPERPARAMETER_BOUNDS = {
                'n_estimators': (50, 300),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0)
            }
        
        if self.ALERT_THRESHOLDS is None:
            self.ALERT_THRESHOLDS = {
                'daily_loss': -0.03,
                'position_loss': -0.05,
                'drawdown': 0.08,
                'correlation': 0.8
            }
        
        if self.CRYPTO_EXCHANGES is None:
            self.CRYPTO_EXCHANGES = ['binance', 'coinbase', 'kraken']
        
        if self.CRYPTO_PAIRS is None:
            self.CRYPTO_PAIRS = ['BTC/USD', 'ETH/USD', 'SOL/USD']

# Load configuration
config = UltraAdvancedTradingConfig()

# === ULTRA-ADVANCED LOGGING SYSTEM ===
class UltraAdvancedLogger:
    def __init__(self, name: str = "UltraTradingBot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handlers
        os.makedirs("logs", exist_ok=True)
        
        # Main log file
        file_handler = logging.FileHandler(f"logs/ultra_trading_bot_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Error log file
        error_handler = logging.FileHandler(f"logs/errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # Trade log file
        trade_handler = logging.FileHandler(f"logs/trades_{datetime.now().strftime('%Y%m%d')}.log")
        trade_handler.setFormatter(detailed_formatter)
        self.trade_logger = logging.getLogger(f"{name}.trades")
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
    
    def info(self, message: str, extra: Dict = None):
        self.logger.info(message, extra=extra or {})
    
    def error(self, message: str, extra: Dict = None):
        self.logger.error(message, extra=extra or {})
    
    def warning(self, message: str, extra: Dict = None):
        self.logger.warning(message, extra=extra or {})
    
    def debug(self, message: str, extra: Dict = None):
        self.logger.debug(message, extra=extra or {})
    
    def trade(self, message: str, extra: Dict = None):
        self.trade_logger.info(message, extra=extra or {})

logger = UltraAdvancedLogger()

# === ENTERPRISE SIP DATA MANAGER ===
class EnterpriseSIPDataManager:
    """Ultra-advanced SIP data management with multiple providers"""
    
    def __init__(self):
        self.polygon_client = None
        self.iex_client = None
        self.intrinio_client = None
        self.active_providers = []
        self.latency_stats = {}
        self.data_quality_stats = {}
        self.initialize_providers()
    
    def initialize_providers(self):
        """Initialize all available SIP data providers"""
        try:
            # Polygon
            if POLYGON_AVAILABLE and config.POLYGON_API_KEY:
                self.polygon_client = PolygonClient(config.POLYGON_API_KEY)
                self.active_providers.append('polygon')
                logger.info("✅ Polygon SIP client initialized")
            
            # IEX Cloud
            if IEX_AVAILABLE and config.IEX_API_KEY:
                os.environ['IEX_API_KEY'] = config.IEX_API_KEY
                self.active_providers.append('iex')
                logger.info("✅ IEX Cloud SIP client initialized")
            
            # Intrinio
            if INTRINIO_AVAILABLE and config.INTRINIO_API_KEY:
                intrinio_sdk.ApiClient().configuration.api_key['api_key'] = config.INTRINIO_API_KEY
                self.intrinio_client = intrinio_sdk.SecurityApi()
                self.active_providers.append('intrinio')
                logger.info("✅ Intrinio SIP client initialized")
            
            if not self.active_providers:
                logger.warning("⚠️ No SIP data providers available - using fallback")
            
        except Exception as e:
            logger.error(f"❌ SIP provider initialization failed: {e}")
    
    def get_real_time_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote with sub-200ms latency"""
        start_time = time.time()
        
        for provider in self.active_providers:
            try:
                quote = None
                
                if provider == 'polygon' and self.polygon_client:
                    quote = self._get_polygon_quote(symbol)
                elif provider == 'iex' and IEX_AVAILABLE:
                    quote = self._get_iex_quote(symbol)
                elif provider == 'intrinio' and self.intrinio_client:
                    quote = self._get_intrinio_quote(symbol)
                
                if quote:
                    latency = (time.time() - start_time) * 1000
                    self._update_latency_stats(provider, latency)
                    
                    if latency <= config.TARGET_LATENCY_MS:
                        quote['latency_ms'] = latency
                        quote['provider'] = provider
                        return quote
                
            except Exception as e:
                logger.warning(f"⚠️ {provider} quote failed for {symbol}: {e}")
                continue
        
        return None
    
    def _get_polygon_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote from Polygon"""
        try:
            quote = self.polygon_client.get_last_quote(symbol)
            return {
                'symbol': symbol,
                'bid': quote.bid,
                'ask': quote.ask,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp
            }
        except Exception:
            return None
    
    def _get_iex_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote from IEX"""
        try:
            stock = Stock(symbol)
            quote = stock.get_quote()
            return {
                'symbol': symbol,
                'bid': quote.get('iexBidPrice', 0),
                'ask': quote.get('iexAskPrice', 0),
                'bid_size': quote.get('iexBidSize', 0),
                'ask_size': quote.get('iexAskSize', 0),
                'timestamp': quote.get('latestUpdate', 0)
            }
        except Exception:
            return None
    
    def _get_intrinio_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote from Intrinio"""
        try:
            quote = self.intrinio_client.get_security_realtime_price(symbol)
            return {
                'symbol': symbol,
                'bid': quote.bid,
                'ask': quote.ask,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': quote.updated_on
            }
        except Exception:
            return None
    
    def _update_latency_stats(self, provider: str, latency: float):
        """Update latency statistics"""
        if provider not in self.latency_stats:
            self.latency_stats[provider] = []
        
        self.latency_stats[provider].append(latency)
        
        # Keep only last 100 measurements
        if len(self.latency_stats[provider]) > 100:
            self.latency_stats[provider] = self.latency_stats[provider][-100:]
    
    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics for all providers"""
        stats = {}
        for provider, latencies in self.latency_stats.items():
            if latencies:
                stats[provider] = {
                    'avg': np.mean(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                }
        return stats

sip_data_manager = EnterpriseSIPDataManager()

# === ULTRA-ADVANCED API MANAGEMENT ===
class UltraAdvancedAPIManager:
    def __init__(self):
        self.api = None
        self.news_api = None
        self.rate_limiters = {}
        self.request_history = defaultdict(list)
        self.circuit_breakers = {}
        self.initialize_apis()
    
    def initialize_apis(self):
        """Initialize all APIs with advanced error handling"""
        try:
            # Alpaca API with retry logic
            if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
                self.api = REST(
                    config.ALPACA_API_KEY, 
                    config.ALPACA_SECRET_KEY, 
                    base_url=config.ALPACA_BASE_URL
                )
                
                # Test connection
                account = self.safe_api_call(self.api.get_account)
                if account:
                    logger.info(f"✅ Alpaca API connected - Account: ${float(account.equity):,.2f}")
                else:
                    logger.warning("⚠️ Alpaca API connection test failed")
            
            # News API
            if config.NEWS_API_KEY:
                self.news_api = NewsApiClient(api_key=config.NEWS_API_KEY)
                logger.info("✅ News API initialized")
            
        except Exception as e:
            logger.error(f"❌ API initialization failed: {e}")
    
    def safe_api_call(self, func, *args, max_retries: int = 3, **kwargs):
        """Ultra-safe API call with circuit breaker pattern"""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # Check circuit breaker
        if self._is_circuit_open(func_name):
            logger.warning(f"⚠️ Circuit breaker open for {func_name}")
            return None
        
        # Rate limiting
        if not self._check_rate_limit(func_name):
            logger.warning(f"⚠️ Rate limit exceeded for {func_name}")
            return None
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record successful call
                self._record_api_call(func_name, True, time.time() - start_time)
                self._reset_circuit_breaker(func_name)
                
                return result
                
            except Exception as e:
                self._record_api_call(func_name, False, time.time() - start_time)
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")
                    self._trip_circuit_breaker(func_name)
                    return None
    
    def _check_rate_limit(self, func_name: str) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Clean old requests
        cutoff_time = current_time - 60  # 1 minute window
        self.request_history[func_name] = [
            req_time for req_time in self.request_history[func_name] 
            if req_time > cutoff_time
        ]
        
        # Check limit (60 requests per minute)
        if len(self.request_history[func_name]) >= 60:
            return False
        
        self.request_history[func_name].append(current_time)
        return True
    
    def _is_circuit_open(self, func_name: str) -> bool:
        """Check if circuit breaker is open"""
        if func_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[func_name]
        
        # Check if enough time has passed to try again
        if time.time() - breaker['trip_time'] > breaker['timeout']:
            return False
        
        return breaker['is_open']
    
    def _trip_circuit_breaker(self, func_name: str):
        """Trip the circuit breaker"""
        self.circuit_breakers[func_name] = {
            'is_open': True,
            'trip_time': time.time(),
            'timeout': 300,  # 5 minutes
            'failure_count': self.circuit_breakers.get(func_name, {}).get('failure_count', 0) + 1
        }
    
    def _reset_circuit_breaker(self, func_name: str):
        """Reset the circuit breaker"""
        if func_name in self.circuit_breakers:
            self.circuit_breakers[func_name]['is_open'] = False
            self.circuit_breakers[func_name]['failure_count'] = 0
    
    def _record_api_call(self, func_name: str, success: bool, duration: float):
        """Record API call metrics"""
        # This could be expanded to store in a database or monitoring system
        pass

api_manager = UltraAdvancedAPIManager()

# === ULTRA-ADVANCED MARKET STATUS MANAGER ===
class UltraAdvancedMarketStatusManager:
    def __init__(self):
        self.market_timezone = pytz.timezone("US/Eastern")
        self.market_calendar = {}
        self.extended_hours_enabled = True
        self.premarket_start = "04:00"
        self.premarket_end = "09:30"
        self.afterhours_start = "16:00"
        self.afterhours_end = "20:00"
        self.load_market_calendar()
    
    def load_market_calendar(self):
        """Load market calendar with holidays"""
        try:
            if api_manager.api:
                calendar = api_manager.safe_api_call(api_manager.api.get_calendar)
                if calendar:
                    for day in calendar:
                        date_str = day.date.strftime('%Y-%m-%d')
                        self.market_calendar[date_str] = {
                            'open': day.open,
                            'close': day.close,
                            'is_open': True
                        }
        except Exception as e:
            logger.warning(f"⚠️ Failed to load market calendar: {e}")
    
    def is_market_open(self, include_extended: bool = False) -> bool:
        """Check if market is open (including extended hours)"""
        try:
            current_time = datetime.now(self.market_timezone)
            current_date = current_time.date()
            
            # Check if it's a weekend
            if current_time.weekday() >= 5:
                return False
            
            # Check market calendar for holidays
            date_str = current_date.strftime('%Y-%m-%d')
            if date_str in self.market_calendar and not self.market_calendar[date_str]['is_open']:
                return False
            
            current_time_str = current_time.strftime('%H:%M')
            
            if include_extended and self.extended_hours_enabled:
                # Extended hours: 4:00 AM - 8:00 PM ET
                return self.premarket_start <= current_time_str <= self.afterhours_end
            else:
                # Regular hours: 9:30 AM - 4:00 PM ET
                return "09:30" <= current_time_str <= "16:00"
                
        except Exception as e:
            logger.error(f"❌ Market status check failed: {e}")
            return False
    
    def get_market_session(self) -> str:
        """Get current market session"""
        try:
            current_time = datetime.now(self.market_timezone)
            current_time_str = current_time.strftime('%H:%M')
            
            if not self.is_market_open(include_extended=True):
                return "closed"
            elif self.premarket_start <= current_time_str < "09:30":
                return "premarket"
            elif "09:30" <= current_time_str <= "16:00":
                return "regular"
            elif "16:00" < current_time_str <= self.afterhours_end:
                return "afterhours"
            else:
                return "closed"
                
        except Exception as e:
            logger.error(f"❌ Market session check failed: {e}")
            return "unknown"
    
    def time_to_market_open(self) -> Optional[timedelta]:
        """Get time until market opens"""
        try:
            current_time = datetime.now(self.market_timezone)
            
            if self.is_market_open():
                return timedelta(0)
            
            # Calculate next market open
            next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # If it's after market close today, next open is tomorrow
            if current_time.hour >= 16:
                next_open += timedelta(days=1)
            
            # Skip weekends
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            
            return next_open - current_time
            
        except Exception as e:
            logger.error(f"❌ Time to market open calculation failed: {e}")
            return None

market_status = UltraAdvancedMarketStatusManager()

# === ULTRA-ADVANCED TRADING STATE ===
class UltraAdvancedTradingState:
    def __init__(self):
        # Core state
        self.open_positions = {}
        self.trade_history = []
        self.current_watchlist = []
        self.qualified_watchlist = []
        self.models_trained = False
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = config.INITIAL_CAPITAL
        self.current_equity = config.INITIAL_CAPITAL
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        
        # Risk metrics
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        
        # Sector allocation
        self.sector_allocations = {}
        self.sector_performance = {}
        
        # Model performance
        self.model_accuracy = {}
        self.model_predictions = defaultdict(list)
        self.model_outcomes = defaultdict(list)
        
        # Enterprise features state
        self.regime_state = "neutral"
        self.volatility_regime = "normal"
        self.market_sentiment = 0.0
        
        # Caching
        self.feature_cache = {}
        self.sentiment_cache = {}
        self.price_cache = {}
        
        # Risk monitoring
        self.trading_halted = False
        self.halt_reason = ""
        self.risk_alerts = []
        
        # Position tracking
        self.position_entry_times = {}
        self.position_stop_losses = {}
        self.position_take_profits = {}
        self.position_sectors = {}
        self.position_confidence = {}
        
        # Feedback loop
        self.feedback_history = deque(maxlen=1000)
        self.meta_model_performance = {}

        # Pending feedback
        self.pending_feedback = {}
        
        self.initialize_default_watchlist()
        self.load_state()
    
    def initialize_default_watchlist(self):
        """Initialize comprehensive default watchlist"""
        self.current_watchlist = [
            # Large Cap Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
            'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC',
            
            # Healthcare & Biotech
            'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'LLY', 'MRK', 'GILD',
            'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'EW', 'SYK',
            
            # Financial Services
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',
            'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO', 'V', 'MA', 'PYPL',
            
            # Consumer & Retail
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'LULU', 'TGT',
            'COST', 'LOW', 'TJX', 'ROST', 'ULTA', 'DG', 'DLTR', 'BBY', 'ETSY', 'SHOP',
            
            # Energy & Utilities
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'KMI', 'WMB', 'EPD', 'ET',
            
            # Industrial & Materials
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            
            # Communication Services
            'VZ', 'T', 'CMCSA', 'DIS', 'NFLX', 'GOOGL', 'META', 'TWTR', 'SNAP', 'PINS',
            
            # Real Estate & REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'UDR', 'ESS'
        ]
    
    def get_ticker_sector(self, ticker: str) -> str:
        """Get sector for ticker with comprehensive mapping"""
        sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Technology', 'NVDA': 'Technology', 'META': 'Technology', 'NFLX': 'Technology',
            'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology', 'INTC': 'Technology',
            'AMD': 'Technology', 'QCOM': 'Technology', 'AVGO': 'Technology', 'TXN': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'TMO': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'LLY': 'Healthcare',
            'MRK': 'Healthcare', 'GILD': 'Healthcare', 'AMGN': 'Healthcare', 'BIIB': 'Healthcare',
            
            # Finance
            'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance',
            'MS': 'Finance', 'C': 'Finance', 'BLK': 'Finance', 'SCHW': 'Finance',
            'AXP': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'PYPL': 'Finance',
            
            # Consumer
            'PG': 'Consumer', 'KO': 'Consumer', 'PEP': 'Consumer', 'WMT': 'Consumer',
            'HD': 'Consumer', 'MCD': 'Consumer', 'SBUX': 'Consumer', 'NKE': 'Consumer',
            'COST': 'Consumer', 'TGT': 'Consumer', 'LOW': 'Consumer',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'SLB': 'Energy', 'OXY': 'Energy', 'KMI': 'Energy',
            
            # Industrial
            'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial',
            'HON': 'Industrial', 'UPS': 'Industrial', 'FDX': 'Industrial', 'LMT': 'Industrial',
            
            # Communication
            'VZ': 'Communication', 'T': 'Communication', 'CMCSA': 'Communication', 'DIS': 'Communication',
            
            # Real Estate
            'AMT': 'RealEstate', 'PLD': 'RealEstate', 'CCI': 'RealEstate', 'EQIX': 'RealEstate'
        }
        return sector_mapping.get(ticker, 'Other')
    
    def add_position(self, ticker: str, quantity: int, entry_price: float, 
                    confidence: float = 0.5, stop_loss: float = None, 
                    take_profit: float = None) -> bool:
        """Add position with comprehensive tracking"""
        try:
            sector = self.get_ticker_sector(ticker)
            
            # Check position limits
            if len(self.open_positions) >= config.MAX_POSITIONS:
                logger.warning(f"⚠️ Maximum positions reached ({config.MAX_POSITIONS})")
                return False
            
            # Check sector concentration
            sector_value = sum([
                pos['quantity'] * pos['entry_price'] 
                for pos in self.open_positions.values() 
                if pos.get('sector') == sector
            ])
            
            position_value = quantity * entry_price
            total_value = self.current_equity
            
            if (sector_value + position_value) / total_value > config.MAX_SECTOR_ALLOCATION:
                logger.warning(f"⚠️ Sector concentration limit exceeded for {sector}")
                return False
            
            # Add position
            self.open_positions[ticker] = {
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'sector': sector,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'unrealized_pnl': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'current_price': entry_price
            }
            
            # Update tracking
            self.position_entry_times[ticker] = datetime.now()
            self.position_sectors[ticker] = sector
            self.position_confidence[ticker] = confidence
            
            if stop_loss:
                self.position_stop_losses[ticker] = stop_loss
            if take_profit:
                self.position_take_profits[ticker] = take_profit
            
            self.update_sector_allocations()
            
            logger.info(f"📈 Position added: {ticker} - {quantity} shares @ ${entry_price:.2f} (Confidence: {confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add position for {ticker}: {e}")
            return False
    
    def remove_position(self, ticker: str, exit_price: float, reason: str = "manual") -> Optional[Dict]:
        """Remove position with comprehensive tracking"""
        try:
            if ticker not in self.open_positions:
                logger.warning(f"⚠️ Attempted to remove non-existent position: {ticker}")
                return None
            
            position = self.open_positions[ticker]
            quantity = position['quantity']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            sector = position.get('sector', 'Unknown')
            confidence = position.get('confidence', 0.5)
            
            # Calculate trade outcome
            pnl = (exit_price - entry_price) * quantity
            return_pct = (exit_price - entry_price) / entry_price
            hold_duration = (datetime.now() - entry_time).total_seconds() / 60
            
            # Create trade record
            trade_record = {
                'ticker': ticker,
                'action': 'sell',
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_time,
                'exit_time': datetime.now(),
                'pnl': pnl,
                'return': return_pct,
                'hold_duration': hold_duration,
                'sector': sector,
                'confidence': confidence,
                'reason': reason
            }
            
            # Update statistics
            self.trade_history.append(trade_record)
            self.total_trades += 1
            self.total_pnl += pnl
            self.daily_pnl += pnl
            self.current_equity += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Remove from tracking
            del self.open_positions[ticker]
            self.position_entry_times.pop(ticker, None)
            self.position_stop_losses.pop(ticker, None)
            self.position_take_profits.pop(ticker, None)
            self.position_sectors.pop(ticker, None)
            self.position_confidence.pop(ticker, None)
            
            self.update_sector_allocations()
            
            logger.trade(f"📉 Position closed: {ticker} - P&L: ${pnl:.2f} ({return_pct:.2%}) - {reason}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"❌ Failed to remove position for {ticker}: {e}")
            return None
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """Update current prices for all positions"""
        try:
            for ticker, current_price in price_updates.items():
                if ticker in self.open_positions:
                    position = self.open_positions[ticker]
                    position['current_price'] = current_price
                    
                    # Calculate unrealized P&L
                    unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
                    position['unrealized_pnl'] = unrealized_pnl
                    
                    # Track max profit/loss
                    if unrealized_pnl > position['max_profit']:
                        position['max_profit'] = unrealized_pnl
                    if unrealized_pnl < position['max_loss']:
                        position['max_loss'] = unrealized_pnl
            
            # Update current equity
            total_unrealized = sum([pos['unrealized_pnl'] for pos in self.open_positions.values()])
            self.current_equity = config.INITIAL_CAPITAL + self.total_pnl + total_unrealized
            
            # Update peak equity and drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            
        except Exception as e:
            logger.error(f"❌ Failed to update position prices: {e}")
    
    def update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            if not self.trade_history:
                return
            
            returns = [trade['return'] for trade in self.trade_history]
            
            if len(returns) < 2:
                return
            
            # Basic metrics
            self.win_rate = self.winning_trades / max(self.total_trades, 1)
            
            winning_returns = [r for r in returns if r > 0]
            losing_returns = [r for r in returns if r < 0]
            
            self.avg_win = np.mean(winning_returns) if winning_returns else 0
            self.avg_loss = np.mean(losing_returns) if losing_returns else 0
            
            # Profit factor
            gross_profit = sum(winning_returns)
            gross_loss = abs(sum(losing_returns))
            self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk-adjusted metrics
            returns_array = np.array(returns)
            
            if np.std(returns_array) > 0:
                self.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    self.sortino_ratio = np.mean(returns_array) / downside_std * np.sqrt(252)
            
            # Calmar ratio
            if self.max_drawdown > 0:
                annual_return = np.mean(returns_array) * 252
                self.calmar_ratio = annual_return / self.max_drawdown
            
            # VaR and CVaR
            if len(returns_array) >= 20:
                self.var_95 = abs(np.percentile(returns_array, 5))
                tail_returns = returns_array[returns_array <= np.percentile(returns_array, 5)]
                self.cvar_95 = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else 0
                
        except Exception as e:
            logger.error(f"❌ Performance metrics update failed: {e}")
    
    def update_sector_allocations(self):
        """Update sector allocation tracking"""
        try:
            sector_values = defaultdict(float)
            total_value = 0.0
            
            for ticker, position in self.open_positions.items():
                sector = position.get('sector', 'Other')
                position_value = position['quantity'] * position.get('current_price', position['entry_price'])
                sector_values[sector] += position_value
                total_value += position_value
            
            if total_value > 0:
                self.sector_allocations = {
                    sector: value / total_value 
                    for sector, value in sector_values.items()
                }
            else:
                self.sector_allocations = {}
                
        except Exception as e:
            logger.error(f"❌ Failed to update sector allocations: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            total_unrealized = sum([pos['unrealized_pnl'] for pos in self.open_positions.values()])
            
            return {
                'current_equity': self.current_equity,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'unrealized_pnl': total_unrealized,
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'current_drawdown': self.current_drawdown,
                'open_positions': len(self.open_positions),
                'sector_allocations': self.sector_allocations,
                'trading_halted': self.trading_halted
            }
            
        except Exception as e:
            logger.error(f"❌ Portfolio summary failed: {e}")
            return {}
    
    def save_state(self):
        """Save comprehensive trading state"""
        try:
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'open_positions': self.open_positions,
                'trade_history': self.trade_history[-1000:],  # Keep last 1000 trades
                'performance_metrics': {
                    'total_pnl': self.total_pnl,
                    'daily_pnl': self.daily_pnl,
                    'current_equity': self.current_equity,
                    'peak_equity': self.peak_equity,
                    'max_drawdown': self.max_drawdown,
                    'total_trades': self.total_trades,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades,
                    'win_rate': self.win_rate,
                    'sharpe_ratio': self.sharpe_ratio,
                    'profit_factor': self.profit_factor
                },
                'sector_allocations': self.sector_allocations,
                'model_accuracy': self.model_accuracy,
                'current_watchlist': self.current_watchlist,
                'qualified_watchlist': self.qualified_watchlist,
                'models_trained': self.models_trained,
                'regime_state': self.regime_state,
                'volatility_regime': self.volatility_regime
            }
            
            os.makedirs('state', exist_ok=True)
            
            # Save current state
            with open('state/trading_state.json', 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Save backup with timestamp
            backup_filename = f"state/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_filename, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info("💾 Trading state saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save trading state: {e}")
    
    def load_state(self):
        """Load saved trading state"""
        try:
            if not os.path.exists('state/trading_state.json'):
                logger.info("📂 No saved state found, starting fresh")
                return
            
            with open('state/trading_state.json', 'r') as f:
                state_data = json.load(f)
            
            # Load performance metrics
            perf_metrics = state_data.get('performance_metrics', {})
            self.total_pnl = perf_metrics.get('total_pnl', 0.0)
            self.daily_pnl = perf_metrics.get('daily_pnl', 0.0)
            self.current_equity = perf_metrics.get('current_equity', config.INITIAL_CAPITAL)
            self.peak_equity = perf_metrics.get('peak_equity', config.INITIAL_CAPITAL)
            self.max_drawdown = perf_metrics.get('max_drawdown', 0.0)
            self.total_trades = perf_metrics.get('total_trades', 0)
            self.winning_trades = perf_metrics.get('winning_trades', 0)
            self.losing_trades = perf_metrics.get('losing_trades', 0)
            self.win_rate = perf_metrics.get('win_rate', 0.0)
            self.sharpe_ratio = perf_metrics.get('sharpe_ratio', 0.0)
            self.profit_factor = perf_metrics.get('profit_factor', 0.0)
            
            # Load other state
            self.trade_history = state_data.get('trade_history', [])
            self.sector_allocations = state_data.get('sector_allocations', {})
            self.model_accuracy = state_data.get('model_accuracy', {})
            self.current_watchlist = state_data.get('current_watchlist', self.current_watchlist)
            self.qualified_watchlist = state_data.get('qualified_watchlist', [])
            self.models_trained = state_data.get('models_trained', False)
            self.regime_state = state_data.get('regime_state', 'neutral')
            self.volatility_regime = state_data.get('volatility_regime', 'normal')
            
            logger.info(f"📂 Trading state loaded: {self.total_trades} trades, ${self.total_pnl:.2f} P&L")
            
        except Exception as e:
            logger.error(f"❌ Failed to load trading state: {e}")

trading_state = UltraAdvancedTradingState()

# === ULTRA-ADVANCED DATA FETCHING ===
def get_ultra_advanced_data(ticker: str, limit: int = 200, timeframe=TimeFrame.Minute, 
                           days_back: int = None, use_sip: bool = True) -> Optional[pd.DataFrame]:
    """Ultra-advanced data fetching with SIP integration"""
    try:
        # Try SIP data first if enabled
        if use_sip and config.SIP_DATA_ENABLED:
            sip_data = sip_data_manager.get_real_time_quote(ticker)
            if sip_data and sip_data.get('latency_ms', 1000) <= config.TARGET_LATENCY_MS:
                logger.debug(f"✅ Using SIP data for {ticker} (latency: {sip_data['latency_ms']:.1f}ms)")
        
        # Fallback to Alpaca
        if not api_manager.api:
            logger.warning(f"⚠️ No API available for {ticker}")
            return None
        
        if days_back:
            start_time = datetime.now() - timedelta(days=days_back)
        else:
            start_time = datetime.now() - timedelta(days=10)
        
        bars = api_manager.safe_api_call(
            api_manager.api.get_bars,
            ticker,
            timeframe,
            start=start_time.strftime('%Y-%m-%d'),
            limit=limit,
            adjustment='raw'
        )
        
        if not bars:
            logger.warning(f"⚠️ No data received for {ticker}")
            return None
        
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Add ultra-advanced technical indicators
        df = add_ultra_comprehensive_technical_indicators(df)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Ultra-advanced data fetch failed for {ticker}: {e}")
        return None

def add_ultra_comprehensive_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add the most comprehensive set of technical indicators"""
    try:
        if data is None or data.empty or len(data) < 50:
            return data
        
        # === PRICE-BASED INDICATORS ===
        # Moving Averages
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_100'] = data['close'].rolling(window=100).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['ema_5'] = data['close'].ewm(span=5).mean()
        data['ema_10'] = data['close'].ewm(span=10).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_20'] = data['close'].ewm(span=20).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        
        # === MOMENTUM INDICATORS ===
        # RSI with multiple periods
        rsi_14 = RSIIndicator(close=data['close'], window=14)
        data['rsi_14'] = rsi_14.rsi()
        
        rsi_7 = RSIIndicator(close=data['close'], window=7)
        data['rsi_7'] = rsi_7.rsi()
        
        rsi_21 = RSIIndicator(close=data['close'], window=21)
        data['rsi_21'] = rsi_21.rsi()
        
        # MACD with multiple settings
        macd = MACD(close=data['close'], window_slow=26, window_fast=12, window_sign=9)
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        # Fast MACD
        macd_fast = MACD(close=data['close'], window_slow=21, window_fast=8, window_sign=5)
        data['macd_fast'] = macd_fast.macd()
        data['macd_fast_signal'] = macd_fast.macd_signal()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=data['high'], low=data['low'], close=data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        williams = WilliamsRIndicator(high=data['high'], low=data['low'], close=data['close'])
        data['williams_r'] = williams.williams_r()
        
        # Rate of Change
        data['roc_5'] = data['close'].pct_change(periods=5) * 100
        data['roc_10'] = data['close'].pct_change(periods=10) * 100
        data['roc_20'] = data['close'].pct_change(periods=20) * 100
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb = BollingerBands(close=data['close'], window=20, window_dev=2)
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Keltner Channels
        keltner = KeltnerChannel(high=data['high'], low=data['low'], close=data['close'])
        data['keltner_upper'] = keltner.keltner_channel_hband()
        data['keltner_lower'] = keltner.keltner_channel_lband()
        data['keltner_middle'] = keltner.keltner_channel_mband()
        
        # Average True Range
        atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'])
        data['atr_14'] = atr.average_true_range()
        data['atr_7'] = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=7).average_true_range()
        
        # === VOLUME INDICATORS ===
        # Volume Moving Averages
        data['volume_sma_10'] = data['volume'].rolling(window=10).mean()
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        data['volume_sma_50'] = data['volume'].rolling(window=50).mean()
        
        # Volume Ratios
        data['volume_ratio_10'] = data['volume'] / data['volume_sma_10']
        data['volume_ratio_20'] = data['volume'] / data['volume_sma_20']
        data['volume_ratio_50'] = data['volume'] / data['volume_sma_50']
        
        # On-Balance Volume
        obv = OnBalanceVolumeIndicator(close=data['close'], volume=data['volume'])
        data['obv'] = obv.on_balance_volume()
        data['obv_sma'] = data['obv'].rolling(window=20).mean()
        
        # Money Flow Index
        mfi = MFIIndicator(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'])
        data['mfi'] = mfi.money_flow_index()
        
        # Volume Price Trend
        data['vpt'] = (data['volume'] * data['close'].pct_change()).cumsum()
        
        # === TREND INDICATORS ===
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=data['high'], low=data['low'], close=data['close'])
        data['adx'] = adx.adx()
        data['adx_pos'] = adx.adx_pos()
        data['adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        if TALIB_AVAILABLE:
            data['sar'] = talib.SAR(data['high'].values, data['low'].values)
        
        # === PRICE ACTION INDICATORS ===
        # VWAP (Volume Weighted Average Price)
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
        
        # Price momentum
        data['price_momentum_1'] = data['close'].pct_change(periods=1)
        data['price_momentum_3'] = data['close'].pct_change(periods=3)
        data['price_momentum_5'] = data['close'].pct_change(periods=5)
        data['price_momentum_10'] = data['close'].pct_change(periods=10)
        
        # High-Low ratios
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # === ADVANCED PATTERN INDICATORS ===
        # Doji detection
        body_size = abs(data['close'] - data['open'])
        candle_range = data['high'] - data['low']
        data['is_doji'] = (body_size / candle_range < 0.1).astype(int)
        
        # Hammer/Shooting Star detection
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        data['is_hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
        data['is_shooting_star'] = ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)
        
        # === STATISTICAL INDICATORS ===
        # Z-Score
        data['price_zscore_20'] = (data['close'] - data['close'].rolling(20).mean()) / data['close'].rolling(20).std()
        data['volume_zscore_20'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
        
        # Percentile ranks
        data['price_percentile_20'] = data['close'].rolling(20).rank(pct=True)
        data['volume_percentile_20'] = data['volume'].rolling(20).rank(pct=True)
        
        # === VOLATILITY MEASURES ===
        # Historical volatility
        data['volatility_10'] = data['close'].pct_change().rolling(10).std() * np.sqrt(252)
        data['volatility_20'] = data['close'].pct_change().rolling(20).std() * np.sqrt(252)
        data['volatility_50'] = data['close'].pct_change().rolling(50).std() * np.sqrt(252)
        
        # Volatility ratio
        data['volatility_ratio'] = data['volatility_10'] / data['volatility_50']
        
        # === SUPPORT/RESISTANCE LEVELS ===
        # Pivot points
        data['pivot'] = (data['high'] + data['low'] + data['close']) / 3
        data['r1'] = 2 * data['pivot'] - data['low']
        data['s1'] = 2 * data['pivot'] - data['high']
        data['r2'] = data['pivot'] + (data['high'] - data['low'])
        data['s2'] = data['pivot'] - (data['high'] - data['low'])
        
        # Distance to pivot levels
        data['dist_to_pivot'] = abs(data['close'] - data['pivot']) / data['close']
        data['dist_to_r1'] = abs(data['close'] - data['r1']) / data['close']
        data['dist_to_s1'] = abs(data['close'] - data['s1']) / data['close']
        
        # === MARKET MICROSTRUCTURE ===
        # Bid-Ask Spread (simulated)
        data['spread_estimate'] = (data['high'] - data['low']) / data['close'] * 0.1
        
        # Trade imbalance (simulated)
        data['trade_imbalance'] = np.where(data['close'] > data['open'], 1, -1) * data['volume']
        data['trade_imbalance_sma'] = data['trade_imbalance'].rolling(20).mean()
        
        # === LAG FEATURES ===
        # Price lags
        for lag in [1, 2, 3, 5, 10]:
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            data[f'rsi_lag_{lag}'] = data['rsi_14'].shift(lag)
            data[f'macd_lag_{lag}'] = data['macd'].shift(lag)
        
        # === ROLLING STATISTICS ===
        for window in [5, 10, 20, 50]:
            data[f'close_mean_{window}'] = data['close'].rolling(window=window).mean()
            data[f'close_std_{window}'] = data['close'].rolling(window=window).std()
            data[f'close_min_{window}'] = data['close'].rolling(window=window).min()
            data[f'close_max_{window}'] = data['close'].rolling(window=window).max()
            data[f'volume_mean_{window}'] = data['volume'].rolling(window=window).mean()
            data[f'volume_std_{window}'] = data['volume'].rolling(window=window).std()
        
        # === TREND STRENGTH ===
        # Linear regression slope
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope, _ = np.polyfit(x, series, 1)
            return slope
        
        data['price_slope_5'] = data['close'].rolling(5).apply(calculate_slope)
        data['price_slope_10'] = data['close'].rolling(10).apply(calculate_slope)
        data['price_slope_20'] = data['close'].rolling(20).apply(calculate_slope)
        
        # === REGIME INDICATORS ===
        # Trend regime
        data['trend_regime'] = np.where(
            (data['close'] > data['sma_20']) & (data['sma_20'] > data['sma_50']), 1,
            np.where((data['close'] < data['sma_20']) & (data['sma_20'] < data['sma_50']), -1, 0)
        )
        
        # Volatility regime
        vol_median = data['volatility_20'].rolling(100).median()
        data['vol_regime'] = np.where(data['volatility_20'] > vol_median * 1.5, 1, 0)
        
        # === CUSTOM COMPOSITE INDICATORS ===
        # Momentum composite
        data['momentum_composite'] = (
            data['rsi_14'] / 100 * 0.3 +
            (data['macd'] / data['close'] * 1000 + 1) / 2 * 0.3 +
            (data['stoch_k'] / 100) * 0.2 +
            (data['williams_r'] + 100) / 100 * 0.2
        )
        
        # Trend strength composite
        data['trend_strength'] = (
            abs(data['price_slope_20']) * 0.4 +
            (data['adx'] / 100) * 0.3 +
            abs(data['close'] - data['sma_20']) / data['sma_20'] * 0.3
        )
        
        # Volume strength composite
        data['volume_strength'] = (
            data['volume_ratio_20'] * 0.4 +
            (data['mfi'] / 100) * 0.3 +
            abs(data['volume_zscore_20']) / 3 * 0.3
        )
        
        # === FILL NaN VALUES ===
        # Forward fill then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Fill any remaining NaN with 0
        data = data.fillna(0)
        
        logger.debug(f"✅ Added {len(data.columns)} technical indicators")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Technical indicators calculation failed: {e}")
        return data

# === ULTRA-ADVANCED ENSEMBLE MODEL ===
class UltraAdvancedEnsembleModel:
    def __init__(self):
        self.short_term_models = {}
        self.medium_term_models = {}
        self.long_term_models = {}
        self.meta_model = None
        self.regime_models = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.last_training_time = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.hyperparameters = {}
        self.model_versions = {}
        self.online_learners = {}
        
        # AutoML components
        self.optuna_study = None
        self.best_hyperparameters = {}
        
        # Feedback loop
        self.feedback_buffer = deque(maxlen=config.META_MODEL_RETRAIN_FREQUENCY)
        self.prediction_history = defaultdict(list)
        self.outcome_history = defaultdict(list)
        
    def initialize_automl(self):
        """Initialize AutoML optimization"""
        try:
            if not config.AUTOML_ENABLED:
                return
            
            import optuna
            
            # Create or load study
            study_name = config.OPTUNA_STUDY_NAME
            storage = config.OPTUNA_STORAGE
            
            self.optuna_study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction='maximize',
                load_if_exists=True
            )
            
            logger.info(f"✅ AutoML initialized with {len(self.optuna_study.trials)} previous trials")
            
        except Exception as e:
            logger.error(f"❌ AutoML initialization failed: {e}")
    
    def optimize_hyperparameters(self, features: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Optimize hyperparameters using Optuna"""
        try:
            if not config.AUTOML_ENABLED or self.optuna_study is None:
                return self._get_default_hyperparameters()
            
            def objective(trial):
                # Suggest hyperparameters
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                }
                
                # Train model with suggested parameters
                model = XGBClassifier(**params, random_state=42, n_jobs=-1)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, features, labels, 
                    cv=config.CROSS_VALIDATION_FOLDS, 
                    scoring='accuracy'
                )
                
                return cv_scores.mean()
            
            # Optimize
            self.optuna_study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)
            
            # Get best parameters
            best_params = self.optuna_study.best_params
            self.best_hyperparameters = best_params
            
            logger.info(f"✅ Hyperparameter optimization completed. Best score: {self.optuna_study.best_value:.4f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"❌ Hyperparameter optimization failed: {e}")
            return self._get_default_hyperparameters()
    
    def _get_default_hyperparameters(self) -> Dict:
        """Get default hyperparameters"""
        return {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
    
    def extract_ultra_advanced_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract ultra-comprehensive features for ML models"""
        try:
            if data is None or data.empty or len(data) < 50:
                return None
            
            features = pd.DataFrame(index=data.index)
            
            # === CORE PRICE FEATURES ===
            features['close'] = data['close']
            features['high'] = data['high']
            features['low'] = data['low']
            features['volume'] = data['volume']
            features['open'] = data['open']
            
            # === TECHNICAL INDICATORS ===
            technical_columns = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
                'ema_5', 'ema_10', 'ema_12', 'ema_20', 'ema_26', 'ema_50',
                'rsi_7', 'rsi_14', 'rsi_21',
                'macd', 'macd_signal', 'macd_histogram', 'macd_fast', 'macd_fast_signal',
                'stoch_k', 'stoch_d', 'williams_r',
                'bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position',
                'keltner_upper', 'keltner_lower', 'keltner_middle',
                'atr_7', 'atr_14',
                'volume_ratio_10', 'volume_ratio_20', 'volume_ratio_50',
                'obv', 'obv_sma', 'mfi', 'vpt',
                'adx', 'adx_pos', 'adx_neg',
                'vwap', 'vwap_deviation',
                'price_momentum_1', 'price_momentum_3', 'price_momentum_5', 'price_momentum_10',
                'high_low_ratio', 'close_open_ratio',
                'volatility_10', 'volatility_20', 'volatility_50', 'volatility_ratio',
                'momentum_composite', 'trend_strength', 'volume_strength'
            ]
            
            for col in technical_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0.0
            
            # === PATTERN FEATURES ===
            pattern_columns = ['is_doji', 'is_hammer', 'is_shooting_star']
            for col in pattern_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0
            
            # === STATISTICAL FEATURES ===
            stat_columns = [
                'price_zscore_20', 'volume_zscore_20',
                'price_percentile_20', 'volume_percentile_20'
            ]
            for col in stat_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0.0
            
            # === SUPPORT/RESISTANCE FEATURES ===
            sr_columns = ['pivot', 'r1', 's1', 'r2', 's2', 'dist_to_pivot', 'dist_to_r1', 'dist_to_s1']
            for col in sr_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0.0
            
            # === LAG FEATURES ===
            lag_columns = []
            for lag in [1, 2, 3, 5, 10]:
                for base in ['close', 'volume', 'rsi_14', 'macd']:
                    col = f'{base}_lag_{lag}'
                    lag_columns.append(col)
            
            for col in lag_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0.0
            
            # === ROLLING STATISTICS ===
            rolling_columns = []
            for window in [5, 10, 20, 50]:
                for stat in ['mean', 'std', 'min', 'max']:
                    for base in ['close', 'volume']:
                        col = f'{base}_{stat}_{window}'
                        rolling_columns.append(col)
            
            for col in rolling_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0.0
            
            # === TREND FEATURES ===
            trend_columns = ['price_slope_5', 'price_slope_10', 'price_slope_20', 'trend_regime', 'vol_regime']
            for col in trend_columns:
                if col in data.columns:
                    features[col] = data[col]
                else:
                    features[col] = 0.0
            
            # === DERIVED FEATURES ===
            # Price ratios
            features['close_to_sma20'] = features['close'] / features['sma_20']
            features['close_to_sma50'] = features['close'] / features['sma_50']
            features['sma20_to_sma50'] = features['sma_20'] / features['sma_50']
            
            # Volume features
            features['volume_price_trend'] = features['volume'] * features['price_momentum_1']
            features['volume_volatility'] = features['volume'] * features['volatility_20']
            
            # Momentum combinations
            features['rsi_macd_combo'] = features['rsi_14'] * features['macd']
            features['stoch_williams_combo'] = features['stoch_k'] * features['williams_r']
            
            # Volatility features
            features['atr_price_ratio'] = features['atr_14'] / features['close']
            features['bb_squeeze'] = features['bb_width'] < features['bb_width'].rolling(20).quantile(0.2)
            
            # === TIME-BASED FEATURES ===
            # Hour of day (if intraday data)
            if hasattr(features.index, 'hour'):
                features['hour'] = features.index.hour
                features['is_market_open'] = ((features['hour'] >= 9) & (features['hour'] < 16)).astype(int)
                features['is_power_hour'] = ((features['hour'] >= 15) & (features['hour'] < 16)).astype(int)
            else:
                features['hour'] = 12  # Default to noon
                features['is_market_open'] = 1
                features['is_power_hour'] = 0
            
            # Day of week
            if hasattr(features.index, 'dayofweek'):
                features['day_of_week'] = features.index.dayofweek
                features['is_monday'] = (features['day_of_week'] == 0).astype(int)
                features['is_friday'] = (features['day_of_week'] == 4).astype(int)
            else:
                features['day_of_week'] = 2  # Default to Wednesday
                features['is_monday'] = 0
                features['is_friday'] = 0
            
            # === INTERACTION FEATURES ===
            # RSI and volume interaction
            features['rsi_volume_interaction'] = features['rsi_14'] * features['volume_ratio_20']
            
            # MACD and trend interaction
            features['macd_trend_interaction'] = features['macd'] * features['trend_strength']
            
            # Bollinger and volatility interaction
            features['bb_vol_interaction'] = features['bb_position'] * features['volatility_20']
            
            # === REGIME-SPECIFIC FEATURES ===
            # Bull market features
            features['bull_momentum'] = np.where(
                features['trend_regime'] > 0,
                features['momentum_composite'],
                0
            )
            
            # Bear market features
            features['bear_momentum'] = np.where(
                features['trend_regime'] < 0,
                features['momentum_composite'],
                0
            )
            
            # High volatility features
            features['high_vol_momentum'] = np.where(
                features['vol_regime'] > 0,
                features['momentum_composite'],
                0
            )
            
            # === FEATURE SELECTION ===
            if config.FEATURE_SELECTION_ENABLED and self.feature_selector is not None:
                try:
                    selected_features = self.feature_selector.transform(features)
                    feature_names = features.columns[self.feature_selector.get_support()]
                    features = pd.DataFrame(selected_features, columns=feature_names, index=features.index)
                except Exception as e:
                    logger.warning(f"⚠️ Feature selection failed: {e}")
            
            # === FINAL PROCESSING ===
            # Replace infinite values
            features = features.replace([np.inf, -np.inf], 0)
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Ensure all values are numeric
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
            
            logger.debug(f"✅ Extracted {len(features.columns)} features for ML models")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return None
    
    def create_multi_horizon_labels(self, data: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """Create labels for multi-horizon prediction"""
        try:
            if data is None or data.empty:
                return None, None, None
            
            # Short-term labels (next 2 periods)
            short_future_returns = data['close'].shift(-config.SHORT_TERM_DAYS) / data['close'] - 1
            short_labels = (short_future_returns > 0.005).astype(int)  # 0.5% threshold
            
            # Medium-term labels (next 12 periods)
            medium_future_returns = data['close'].shift(-config.MEDIUM_TERM_DAYS) / data['close'] - 1
            medium_labels = (medium_future_returns > 0.01).astype(int)  # 1% threshold
            
            # Long-term labels (next 30 periods)
            long_future_returns = data['close'].shift(-config.LONG_TERM_DAYS) / data['close'] - 1
            long_labels = (long_future_returns > 0.02).astype(int)  # 2% threshold
            
            return short_labels, medium_labels, long_labels
            
        except Exception as e:
            logger.error(f"❌ Label creation failed: {e}")
            return None, None, None
    
    def train_ultra_advanced_ensemble(self, tickers: List[str]) -> bool:
        """Train ultra-advanced ensemble with all enterprise features"""
        try:
            logger.info("🔄 Starting ultra-advanced ensemble training with enterprise features...")
            
            # Initialize AutoML
            self.initialize_automl()
            
            # Collect training data
            all_features = []
            short_labels = []
            medium_labels = []
            long_labels = []
            
            successful_tickers = 0
            
            for ticker in tickers[:config.MIN_TICKERS_FOR_TRAINING]:
                try:
                    logger.info(f"📊 Collecting training data for {ticker}...")
                    
                    # Get multi-timeframe data
                    data_1min = get_ultra_advanced_data(ticker, limit=500, timeframe=TimeFrame.Minute, days_back=config.DATA_LOOKBACK_DAYS)
                    data_daily = get_ultra_advanced_data(ticker, limit=200, timeframe=TimeFrame.Day, days_back=config.DATA_LOOKBACK_DAYS)
                    
                    if data_1min is None or len(data_1min) < config.MIN_TRAINING_SAMPLES:
                        logger.warning(f"⚠️ Insufficient 1min data for {ticker}")
                        continue
                    
                    # Extract features
                    features = self.extract_ultra_advanced_features(data_1min)
                    if features is None or len(features) < 100:
                        logger.warning(f"⚠️ Feature extraction failed for {ticker}")
                        continue
                    
                    # Create labels
                    short_lbls, medium_lbls, long_lbls = self.create_multi_horizon_labels(data_1min)
                    if short_lbls is None or medium_lbls is None or long_lbls is None:
                        logger.warning(f"⚠️ Label creation failed for {ticker}")
                        continue
                    
                    # Align data
                    min_length = min(len(features), len(short_lbls), len(medium_lbls), len(long_lbls))
                    if min_length < 50:
                        logger.warning(f"⚠️ Insufficient aligned data for {ticker}: {min_length}")
                        continue
                    
                    features = features.iloc[:min_length]
                    short_lbls = short_lbls.iloc[:min_length]
                    medium_lbls = medium_lbls.iloc[:min_length]
                    long_lbls = long_lbls.iloc[:min_length]
                    
                    # Remove invalid samples
                    valid_idx = ~(
                        features.isnull().any(axis=1) | 
                        short_lbls.isnull() | 
                        medium_lbls.isnull() | 
                        long_lbls.isnull()
                    )
                    
                    features = features[valid_idx]
                    short_lbls = short_lbls[valid_idx]
                    medium_lbls = medium_lbls[valid_idx]
                    long_lbls = long_lbls[valid_idx]
                    
                    if len(features) < 30:
                        logger.warning(f"⚠️ Too few valid samples for {ticker}: {len(features)}")
                        continue
                    
                    # Add to training set
                    all_features.append(features)
                    short_labels.extend(short_lbls.tolist())
                    medium_labels.extend(medium_lbls.tolist())
                    long_labels.extend(long_lbls.tolist())
                    
                    successful_tickers += 1
                    logger.info(f"✅ Training data collected for {ticker}: {len(features)} samples")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to collect training data for {ticker}: {e}")
                    continue
            
            if not all_features or successful_tickers < 5:
                logger.error(f"❌ Insufficient training data: {successful_tickers} successful tickers")
                return False
            
            # Combine all features
            logger.info("🔄 Combining and preprocessing training data...")
            combined_features = pd.concat(all_features, ignore_index=True)
            short_labels = np.array(short_labels)
            medium_labels = np.array(medium_labels)
            long_labels = np.array(long_labels)
            
            logger.info(f"📊 Total training samples: {len(combined_features)}")
            logger.info(f"📊 Feature dimensions: {combined_features.shape}")
            logger.info(f"📊 Short-term positive rate: {np.mean(short_labels):.2%}")
            logger.info(f"📊 Medium-term positive rate: {np.mean(medium_labels):.2%}")
            logger.info(f"📊 Long-term positive rate: {np.mean(long_labels):.2%}")
            
            # Feature selection
            if config.FEATURE_SELECTION_ENABLED:
                logger.info("🔄 Performing feature selection...")
                from sklearn.feature_selection import SelectKBest, f_classif
                
                self.feature_selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(100, len(combined_features.columns))
                )
                
                selected_features = self.feature_selector.fit_transform(combined_features, short_labels)
                feature_names = combined_features.columns[self.feature_selector.get_support()]
                combined_features = pd.DataFrame(selected_features, columns=feature_names)
                
                logger.info(f"✅ Selected {len(feature_names)} features")
            
            # Scale features
            logger.info("🔄 Scaling features...")
            scaled_features = self.scaler.fit_transform(combined_features)
            scaled_features_df = pd.DataFrame(scaled_features, columns=combined_features.columns)
            
            # Optimize hyperparameters
            logger.info("🔄 Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(scaled_features_df, short_labels)
            
            # Train models for each horizon
            logger.info("🔄 Training short-term models...")
            self.short_term_models = self.train_model_ensemble(
                scaled_features_df, short_labels, "short_term", best_params
            )
            
            logger.info("🔄 Training medium-term models...")
            self.medium_term_models = self.train_model_ensemble(
                scaled_features_df, medium_labels, "medium_term", best_params
            )
            
            logger.info("🔄 Training long-term models...")
            self.long_term_models = self.train_model_ensemble(
                scaled_features_df, long_labels, "long_term", best_params
            )
            
            # Train regime-specific models
            if config.REGIME_MODELS_ENABLED:
                logger.info("🔄 Training regime-specific models...")
                self.train_regime_models(scaled_features_df, short_labels, medium_labels, long_labels)
            
            # Train meta-model
            logger.info("🔄 Training meta-model...")
            self.train_ultra_advanced_meta_model(scaled_features_df, short_labels, medium_labels, long_labels)
            
            # Initialize online learners
            if config.FEEDBACK_LOOP_ENABLED:
                logger.info("🔄 Initializing online learners...")
                self.initialize_online_learners()
            
            # Save models
            self.save_ultra_advanced_models()
            
            # Update state
            self.last_training_time = datetime.now()
            trading_state.models_trained = True
            
            # Calculate and log model performance
            self.evaluate_model_performance(scaled_features_df, short_labels, medium_labels, long_labels)
            
            logger.info("✅ Ultra-advanced ensemble training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ultra-advanced ensemble training failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
    
    def train_model_ensemble(self, features: pd.DataFrame, labels: np.ndarray, 
                           model_type: str, hyperparams: Dict) -> Dict:
        """Train comprehensive ensemble of models"""
        try:
            models = {}
            
            # XGBoost
            models['xgboost'] = XGBClassifier(
                **hyperparams,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            # Random Forest
            models['random_forest'] = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 150),
                max_depth=hyperparams.get('max_depth', 8),
                min_samples_split=hyperparams.get('min_samples_split', 5),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
            
            # Gradient Boosting
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=hyperparams.get('n_estimators', 150),
                max_depth=hyperparams.get('max_depth', 8),
                learning_rate=hyperparams.get('learning_rate', 0.1),
                random_state=42
            )
            
            # Logistic Regression
            models['logistic_regression'] = LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
            
            # Support Vector Machine
            try:
                from sklearn.svm import SVC
                models['svm'] = SVC(
                    probability=True,
                    random_state=42,
                    kernel='rbf'
                )
            except ImportError:
                logger.warning("⚠️ SVM not available")
            
            # Neural Network
            try:
                from sklearn.neural_network import MLPClassifier
                models['neural_network'] = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42
                )
            except ImportError:
                logger.warning("⚠️ Neural Network not available")
            
            # Train each model
            for name, model in models.items():
                try:
                    logger.info(f"🔄 Training {model_type} {name}...")
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, features, labels, 
                        cv=config.CROSS_VALIDATION_FOLDS, 
                        scoring='accuracy'
                    )
                    
                    # Train on full dataset
                    model.fit(features, labels)
                    
                    # Calculate metrics
                    train_score = model.score(features, labels)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Store performance
                    self.model_performance[f"{model_type}_{name}"] = {
                        'train_score': train_score,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'training_time': datetime.now(),
                        'feature_count': len(features.columns),
                        'sample_count': len(features)
                    }
                    
                    logger.info(f"✅ {model_type} {name} trained: CV={cv_mean:.3f}±{cv_std:.3f}, Train={train_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to train {model_type} {name}: {e}")
                    # Remove failed model
                    if name in models:
                        del models[name]
                    continue
            
            return models
            
        except Exception as e:
            logger.error(f"❌ Model ensemble training failed for {model_type}: {e}")
            return {}
    
    def train_regime_models(self, features: pd.DataFrame, short_labels: np.ndarray, 
                          medium_labels: np.ndarray, long_labels: np.ndarray):
        """Train regime-specific models"""
        try:
            if 'vol_regime' not in features.columns or 'trend_regime' not in features.columns:
                logger.warning("⚠️ Regime features not available")
                return
            
            # High volatility regime
            high_vol_mask = features['vol_regime'] == 1
            if np.sum(high_vol_mask) > 100:
                high_vol_features = features[high_vol_mask]
                high_vol_labels = short_labels[high_vol_mask]
                
                self.regime_models['high_volatility'] = XGBClassifier(
                    **self.best_hyperparameters,
                    random_state=42
                )
                self.regime_models['high_volatility'].fit(high_vol_features, high_vol_labels)
                
                logger.info(f"✅ High volatility regime model trained: {len(high_vol_features)} samples")
            
            # Bull market regime
            bull_mask = features['trend_regime'] == 1
            if np.sum(bull_mask) > 100:
                bull_features = features[bull_mask]
                bull_labels = short_labels[bull_mask]
                
                self.regime_models['bull_market'] = XGBClassifier(
                    **self.best_hyperparameters,
                    random_state=42
                )
                self.regime_models['bull_market'].fit(bull_features, bull_labels)
                
                logger.info(f"✅ Bull market regime model trained: {len(bull_features)} samples")
            
            # Bear market regime
            bear_mask = features['trend_regime'] == -1
            if np.sum(bear_mask) > 100:
                bear_features = features[bear_mask]
                bear_labels = short_labels[bear_mask]
                
                self.regime_models['bear_market'] = XGBClassifier(
                    **self.best_hyperparameters,
                    random_state=42
                )
                self.regime_models['bear_market'].fit(bear_features, bear_labels)
                
                logger.info(f"✅ Bear market regime model trained: {len(bear_features)} samples")
            
        except Exception as e:
            logger.error(f"❌ Regime model training failed: {e}")
    
    def train_ultra_advanced_meta_model(self, features: pd.DataFrame, short_labels: np.ndarray, 
                                      medium_labels: np.ndarray, long_labels: np.ndarray):
        """Train ultra-advanced meta-model"""
        try:
            if not self.short_term_models or not self.medium_term_models or not self.long_term_models:
                logger.warning("⚠️ Base models not available for meta-model training")
                return
            
            # Generate base model predictions
            meta_features = []
            
            # Short-term model predictions
            for name, model in self.short_term_models.items():
                try:
                    pred_proba = model.predict_proba(features)[:, 1]
                    meta_features.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get predictions from short-term {name}: {e}")
                    continue
            
            # Medium-term model predictions
            for name, model in self.medium_term_models.items():
                try:
                    pred_proba = model.predict_proba(features)[:, 1]
                    meta_features.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get predictions from medium-term {name}: {e}")
                    continue
            
            # Long-term model predictions
            for name, model in self.long_term_models.items():
                try:
                    pred_proba = model.predict_proba(features)[:, 1]
                    meta_features.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get predictions from long-term {name}: {e}")
                    continue
            
            # Regime model predictions
            for name, model in self.regime_models.items():
                try:
                    pred_proba = model.predict_proba(features)[:, 1]
                    meta_features.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get predictions from regime {name}: {e}")
                    continue
            
            if not meta_features:
                logger.warning("⚠️ No base model predictions for meta-model")
                return
            
            # Combine meta features
            meta_features_array = np.column_stack(meta_features)
            
            # Add original features (subset)
            important_features = ['rsi_14', 'macd', 'bb_position', 'volume_ratio_20', 'momentum_composite']
            for feat in important_features:
                if feat in features.columns:
                    meta_features_array = np.column_stack([meta_features_array, features[feat].values])
            
            # Train meta-model
            self.meta_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.meta_model.fit(meta_features_array, short_labels)
            
            # Evaluate meta-model
            meta_score = self.meta_model.score(meta_features_array, short_labels)
            
            self.model_performance['meta_model'] = {
                'train_score': meta_score,
                'training_time': datetime.now(),
                'feature_count': meta_features_array.shape[1],
                'sample_count': len(meta_features_array)
            }
            
            logger.info(f"✅ Ultra-advanced meta-model trained: {meta_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Meta-model training failed: {e}")
    
    def initialize_online_learners(self):
        """Initialize online learning models for feedback loop"""
        try:
            from sklearn.linear_model import SGDClassifier
            
            # Online SGD classifier for quick adaptation
            self.online_learners['sgd'] = SGDClassifier(
                loss='log',
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
            
            # Fit with dummy data to initialize
            dummy_features = np.random.random((10, 50))
            dummy_labels = np.random.randint(0, 2, 10)
            self.online_learners['sgd'].fit(dummy_features, dummy_labels)
            
            logger.info("✅ Online learners initialized")
            
        except Exception as e:
            logger.error(f"❌ Online learner initialization failed: {e}")
    
    def update_with_feedback(self, features: np.ndarray, actual_outcome: int, prediction: float):
        """Update models with feedback from actual trade outcomes"""
        try:
            if not config.FEEDBACK_LOOP_ENABLED:
                return
            
            # Add to feedback buffer
            self.feedback_buffer.append({
                'features': features,
                'outcome': actual_outcome,
                'prediction': prediction,
                'timestamp': datetime.now()
            })
            
            # Update online learners
            if 'sgd' in self.online_learners:
                try:
                    self.online_learners['sgd'].partial_fit(
                        features.reshape(1, -1), 
                        [actual_outcome]
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Online learner update failed: {e}")
            
            # Retrain meta-model if buffer is full
            if len(self.feedback_buffer) >= config.META_MODEL_RETRAIN_FREQUENCY:
                self.retrain_meta_model_with_feedback()
            
        except Exception as e:
            logger.error(f"❌ Feedback update failed: {e}")
    
    def retrain_meta_model_with_feedback(self):
        """Retrain meta-model with feedback data"""
        try:
            if len(self.feedback_buffer) < 10:
                return
            
            logger.info("🔄 Retraining meta-model with feedback...")
            
            # Extract feedback data
            feedback_features = []
            feedback_outcomes = []
            
            for feedback in self.feedback_buffer:
                feedback_features.append(feedback['features'])
                feedback_outcomes.append(feedback['outcome'])
            
            feedback_features = np.array(feedback_features)
            feedback_outcomes = np.array(feedback_outcomes)
            
            # Retrain meta-model
            if self.meta_model is not None:
                self.meta_model.fit(feedback_features, feedback_outcomes)
                
                # Update performance tracking
                score = self.meta_model.score(feedback_features, feedback_outcomes)
                self.model_performance['meta_model_feedback'] = {
                    'score': score,
                    'samples': len(feedback_features),
                    'timestamp': datetime.now()
                }
                
                logger.info(f"✅ Meta-model retrained with feedback: {score:.3f}")
            
            # Clear buffer
            self.feedback_buffer.clear()
            
        except Exception as e:
            logger.error(f"❌ Meta-model feedback retraining failed: {e}")
    
    def predict_ultra_advanced_multi_horizon(self, data_1min: pd.DataFrame, 
                                           data_daily: pd.DataFrame = None) -> Tuple[float, float, float, float]:
        """Make ultra-advanced multi-horizon predictions"""
        try:
            if not self.short_term_models or not self.medium_term_models or not self.long_term_models:
                logger.warning("⚠️ Models not trained")
                return 0.5, 0.5, 0.5, 0.5
            
            # Extract features
            features = self.extract_ultra_advanced_features(data_1min)
            if features is None or features.empty:
                return 0.5, 0.5, 0.5, 0.5
            
            # Get latest features
            latest_features = features.iloc[[-1]]
            
            # Scale features
            try:
                scaled_features = self.scaler.transform(latest_features)
            except Exception as e:
                logger.warning(f"⚠️ Feature scaling failed: {e}")
                return 0.5, 0.5, 0.5, 0.5
            
            # Short-term predictions
            short_predictions = []
            for name, model in self.short_term_models.items():
                try:
                    pred_proba = model.predict_proba(scaled_features)[0, 1]
                    short_predictions.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Short-term prediction failed for {name}: {e}")
                    continue
            
            # Medium-term predictions
            medium_predictions = []
            for name, model in self.medium_term_models.items():
                try:
                    pred_proba = model.predict_proba(scaled_features)[0, 1]
                    medium_predictions.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Medium-term prediction failed for {name}: {e}")
                    continue
            
            # Long-term predictions
            long_predictions = []
            for name, model in self.long_term_models.items():
                try:
                    pred_proba = model.predict_proba(scaled_features)[0, 1]
                    long_predictions.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Long-term prediction failed for {name}: {e}")
                    continue
            
            # Regime-specific predictions
            regime_predictions = []
            for name, model in self.regime_models.items():
                try:
                    pred_proba = model.predict_proba(scaled_features)[0, 1]
                    regime_predictions.append(pred_proba)
                except Exception as e:
                    logger.warning(f"⚠️ Regime prediction failed for {name}: {e}")
                    continue
            
            # Calculate ensemble predictions
            short_pred = np.mean(short_predictions) if short_predictions else 0.5
            medium_pred = np.mean(medium_predictions) if medium_predictions else 0.5
            long_pred = np.mean(long_predictions) if long_predictions else 0.5
            
            # Meta-model prediction
            meta_pred = 0.5
            if self.meta_model is not None:
                try:
                    # Prepare meta features
                    meta_features = short_predictions + medium_predictions + long_predictions + regime_predictions
                    
                    # Add important original features
                    important_features = ['rsi_14', 'macd', 'bb_position', 'volume_ratio_20', 'momentum_composite']
                    for feat in important_features:
                        if feat in latest_features.columns:
                            meta_features.append(latest_features[feat].iloc[0])
                    
                    if meta_features:
                        meta_features_array = np.array(meta_features).reshape(1, -1)
                        meta_pred = self.meta_model.predict_proba(meta_features_array)[0, 1]
                except Exception as e:
                    logger.warning(f"⚠️ Meta-model prediction failed: {e}")
                    # Fallback to weighted average
                    meta_pred = (
                        short_pred * config.SHORT_TERM_WEIGHT +
                        medium_pred * config.MEDIUM_TERM_WEIGHT +
                        long_pred * config.LONG_TERM_WEIGHT
                    )
            
            return float(short_pred), float(medium_pred), float(long_pred), float(meta_pred)
            
        except Exception as e:
            logger.error(f"❌ Ultra-advanced multi-horizon prediction failed: {e}")
            return 0.5, 0.5, 0.5, 0.5
    
    def evaluate_model_performance(self, features: pd.DataFrame, short_labels: np.ndarray, 
                                 medium_labels: np.ndarray, long_labels: np.ndarray):
        """Evaluate comprehensive model performance"""
        try:
            logger.info("📊 Evaluating model performance...")
            
            # Evaluate each model type
            for model_type, models in [
                ('short_term', self.short_term_models),
                ('medium_term', self.medium_term_models),
                ('long_term', self.long_term_models)
            ]:
                if not models:
                    continue
                
                # Select appropriate labels
                if model_type == 'short_term':
                    labels = short_labels
                elif model_type == 'medium_term':
                    labels = medium_labels
                else:
                    labels = long_labels
                
                for name, model in models.items():
                    try:
                        # Predictions
                        predictions = model.predict(features)
                        pred_proba = model.predict_proba(features)[:, 1]
                        
                        # Metrics
                        accuracy = accuracy_score(labels, predictions)
                        precision = precision_score(labels, predictions, zero_division=0)
                        recall = recall_score(labels, predictions, zero_division=0)
                        
                        # Update model accuracy tracking
                        model_key = f"{model_type}_{name}"
                        trading_state.model_accuracy[model_key] = accuracy
                        
                        logger.info(f"📊 {model_key}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Evaluation failed for {model_type}_{name}: {e}")
                        continue
            
            # Feature importance analysis
            self.analyze_feature_importance(features)
            
        except Exception as e:
            logger.error(f"❌ Model performance evaluation failed: {e}")
    
    def analyze_feature_importance(self, features: pd.DataFrame):
        """Analyze feature importance across models"""
        try:
            feature_importance_scores = defaultdict(list)
            
            # Collect feature importance from tree-based models
            for model_dict in [self.short_term_models, self.medium_term_models, self.long_term_models]:
                for name, model in model_dict.items():
                    if hasattr(model, 'feature_importances_'):
                        for i, importance in enumerate(model.feature_importances_):
                            if i < len(features.columns):
                                feature_importance_scores[features.columns[i]].append(importance)
            
            # Calculate average importance
            avg_importance = {}
            for feature, scores in feature_importance_scores.items():
                avg_importance[feature] = np.mean(scores)
            
            # Sort by importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Log top features
            logger.info("📊 Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                logger.info(f"  {i+1}. {feature}: {importance:.4f}")
            
            self.feature_importance = dict(sorted_features)
            
        except Exception as e:
            logger.error(f"❌ Feature importance analysis failed: {e}")
    
    def save_ultra_advanced_models(self):
        """Save all models and metadata"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save model ensembles
            if self.short_term_models:
                joblib.dump(self.short_term_models, 'models/short_term_models.pkl')
            
            if self.medium_term_models:
                joblib.dump(self.medium_term_models, 'models/medium_term_models.pkl')
            
            if self.long_term_models:
                joblib.dump(self.long_term_models, 'models/long_term_models.pkl')
            
            if self.regime_models:
                joblib.dump(self.regime_models, 'models/regime_models.pkl')
            
            if self.meta_model:
                joblib.dump(self.meta_model, 'models/meta_model.pkl')
            
            if self.online_learners:
                joblib.dump(self.online_learners, 'models/online_learners.pkl')
            
            # Save preprocessing components
            joblib.dump(self.scaler, 'models/scaler.pkl')
            
            if self.feature_selector:
                joblib.dump(self.feature_selector, 'models/feature_selector.pkl')
            
            # Save metadata
            metadata = {
                'training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'hyperparameters': self.best_hyperparameters,
                'config': {
                    'short_term_days': config.SHORT_TERM_DAYS,
                    'medium_term_days': config.MEDIUM_TERM_DAYS,
                    'long_term_days': config.LONG_TERM_DAYS,
                    'automl_enabled': config.AUTOML_ENABLED,
                    'feedback_loop_enabled': config.FEEDBACK_LOOP_ENABLED
                }
            }
            
            with open('models/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("💾 Ultra-advanced models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")
    
    def load_ultra_advanced_models(self) -> bool:
        """Load all saved models and metadata"""
        try:
            if not os.path.exists('models'):
                logger.info("📂 No saved models found")
                return False
            
            models_loaded = 0
            
            # Load model ensembles
            if os.path.exists('models/short_term_models.pkl'):
                self.short_term_models = joblib.load('models/short_term_models.pkl')
                models_loaded += 1
                logger.info("✅ Short-term models loaded")
            
            if os.path.exists('models/medium_term_models.pkl'):
                self.medium_term_models = joblib.load('models/medium_term_models.pkl')
                models_loaded += 1
                logger.info("✅ Medium-term models loaded")
            
            if os.path.exists('models/long_term_models.pkl'):
                self.long_term_models = joblib.load('models/long_term_models.pkl')
                models_loaded += 1
                logger.info("✅ Long-term models loaded")
            
            if os.path.exists('models/regime_models.pkl'):
                self.regime_models = joblib.load('models/regime_models.pkl')
                models_loaded += 1
                logger.info("✅ Regime models loaded")
            
            if os.path.exists('models/meta_model.pkl'):
                self.meta_model = joblib.load('models/meta_model.pkl')
                models_loaded += 1
                logger.info("✅ Meta-model loaded")
            
            if os.path.exists('models/online_learners.pkl'):
                self.online_learners = joblib.load('models/online_learners.pkl')
                models_loaded += 1
                logger.info("✅ Online learners loaded")
            
            # Load preprocessing components
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
                logger.info("✅ Scaler loaded")
            
            if os.path.exists('models/feature_selector.pkl'):
                self.feature_selector = joblib.load('models/feature_selector.pkl')
                logger.info("✅ Feature selector loaded")
            
            # Load metadata
            if os.path.exists('models/metadata.json'):
                with open('models/metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                self.model_performance = metadata.get('model_performance', {})
                self.feature_importance = metadata.get('feature_importance', {})
                self.best_hyperparameters = metadata.get('hyperparameters', {})
                
                training_time_str = metadata.get('training_time')
                if training_time_str:
                    self.last_training_time = datetime.fromisoformat(training_time_str)
                
                logger.info("✅ Model metadata loaded")
            
            if models_loaded > 0:
                logger.info(f"✅ Successfully loaded {models_loaded} model components")
                return True
            else:
                logger.warning("⚠️ No models were loaded")
                return False
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False

ensemble_model = UltraAdvancedEnsembleModel()

# === ULTRA-ADVANCED SENTIMENT ANALYSIS ===
class UltraAdvancedSentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.finbert_pipeline = None
        self.news_cache = {}
        self.sentiment_cache = {}
        self.social_sentiment_cache = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all sentiment analysis models"""
        try:
            # FinBERT for financial sentiment
            if config.SENTIMENT_ANALYSIS_ENABLED:
                try:
                    self.finbert_pipeline = pipeline(
                        "sentiment-analysis",
                        model="ProsusAI/finbert",
                        tokenizer="ProsusAI/finbert"
                    )
                    logger.info("✅ FinBERT model initialized")
                except Exception as e:
                    logger.warning(f"⚠️ FinBERT initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Sentiment model initialization failed: {e}")
    
    def analyze_comprehensive_sentiment(self, ticker: str) -> Dict[str, float]:
        """Analyze comprehensive sentiment from multiple sources"""
        try:
            cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.sentiment_cache:
                return self.sentiment_cache[cache_key]
            
            sentiment_scores = {
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'sentiment_volume': 0
            }
            
            # News sentiment
            news_sentiment = self.analyze_news_sentiment(ticker)
            sentiment_scores['news_sentiment'] = news_sentiment['score']
            sentiment_scores['sentiment_volume'] += news_sentiment['volume']
            
            # Social sentiment (simulated)
            social_sentiment = self.analyze_social_sentiment(ticker)
            sentiment_scores['social_sentiment'] = social_sentiment['score']
            sentiment_scores['sentiment_volume'] += social_sentiment['volume']
            
            # Overall sentiment (weighted average)
            total_volume = sentiment_scores['sentiment_volume']
            if total_volume > 0:
                sentiment_scores['overall_sentiment'] = (
                    news_sentiment['score'] * news_sentiment['volume'] +
                    social_sentiment['score'] * social_sentiment['volume']
                ) / total_volume
            
            # Sentiment strength (absolute value)
            sentiment_scores['sentiment_strength'] = abs(sentiment_scores['overall_sentiment'])
            
            # Cache results
            self.sentiment_cache[cache_key] = sentiment_scores
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"❌ Comprehensive sentiment analysis failed for {ticker}: {e}")
            return {
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'sentiment_volume': 0
            }
    
    def analyze_news_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze news sentiment for ticker"""
        try:
            articles = self.get_ticker_news(ticker)
            
            if not articles:
                return {'score': 0.0, 'volume': 0}
            
            sentiment_scores = []
            
            for article in articles[:20]:  # Analyze up to 20 articles
                try:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    
                    if not text.strip():
                        continue
                    
                    # Try FinBERT first
                    if self.finbert_pipeline:
                        try:
                            result = self.finbert_pipeline(text[:512])
                            if result and len(result) > 0:
                                label = result[0]['label'].lower()
                                score = result[0]['score']
                                
                                if label == 'positive':
                                    sentiment_scores.append(score)
                                elif label == 'negative':
                                    sentiment_scores.append(-score)
                                else:
                                    sentiment_scores.append(0.0)
                                continue
                        except Exception as e:
                            logger.debug(f"FinBERT analysis failed: {e}")
                    
                    # Fallback to VADER
                    vader_scores = self.vader_analyzer.polarity_scores(text)
                    sentiment_scores.append(vader_scores['compound'])
                    
                except Exception as e:
                    logger.debug(f"Article sentiment analysis failed: {e}")
                    continue
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
            else:
                avg_sentiment = 0.0
            
            return {
                'score': avg_sentiment,
                'volume': len(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"❌ News sentiment analysis failed for {ticker}: {e}")
            return {'score': 0.0, 'volume': 0}
    
    def analyze_social_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze social media sentiment (simulated)"""
        try:
            # This would integrate with Twitter API, Reddit API, etc.
            # For now, we'll simulate social sentiment
            
            # Simulate social sentiment based on recent price action
            recent_data = get_ultra_advanced_data(ticker, limit=10)
            if recent_data is not None and not recent_data.empty:
                recent_returns = recent_data['close'].pct_change().dropna()
                if len(recent_returns) > 0:
                    avg_return = recent_returns.mean()
                    # Convert to sentiment score
                    social_score = np.tanh(avg_return * 50)  # Scale and bound
                else:
                    social_score = 0.0
            else:
                social_score = 0.0
            
            # Add some noise to make it more realistic
            social_score += np.random.normal(0, 0.1)
            social_score = np.clip(social_score, -1, 1)
            
            return {
                'score': social_score,
                'volume': np.random.randint(10, 100)  # Simulated volume
            }
            
        except Exception as e:
            logger.error(f"❌ Social sentiment analysis failed for {ticker}: {e}")
            return {'score': 0.0, 'volume': 0}
    
    def get_ticker_news(self, ticker: str) -> List[Dict]:
        """Get news articles for ticker"""
        try:
            cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]
            
            articles = []
            
            if api_manager.news_api:
                try:
                    # Get company name for better search
                    company_names = {
                        'AAPL': 'Apple',
                        'MSFT': 'Microsoft',
                        'GOOGL': 'Google',
                        'AMZN': 'Amazon',
                        'TSLA': 'Tesla',
                        'NVDA': 'NVIDIA',
                        'META': 'Meta',
                        'NFLX': 'Netflix'
                    }
                    
                    search_term = company_names.get(ticker, ticker)
                    
                    news_response = api_manager.news_api.get_everything(
                        q=f"{search_term} stock OR {ticker}",
                        language='en',
                        sort_by='publishedAt',
                        page_size=30,
                        from_param=(datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d')
                    )
                    
                    if news_response and 'articles' in news_response:
                        articles.extend(news_response['articles'])
                        
                except Exception as e:
                    logger.warning(f"News API failed for {ticker}: {e}")
            
            # Cache results
            self.news_cache[cache_key] = articles
            
            return articles
            
        except Exception as e:
            logger.error(f"❌ News fetching failed for {ticker}: {e}")
            return []

sentiment_analyzer = UltraAdvancedSentimentAnalyzer()

# === ULTRA-ADVANCED RISK MONITOR ===
class UltraAdvancedRiskMonitor:
    def __init__(self):
        self.current_drawdown = 0.0
        self.peak_equity = config.INITIAL_CAPITAL
        self.trading_halted = False
        self.halt_reason = ""
        self.risk_alerts = []
        self.last_risk_check = datetime.now()
        self.var_calculator = None
        self.correlation_matrix = None
        self.position_risks = {}
        
    def update_comprehensive_risk_metrics(self):
        """Update comprehensive risk metrics"""
        try:
            current_equity = trading_state.current_equity
            
            # Update peak equity and drawdown
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            
            # Check all risk thresholds
            self.check_comprehensive_risk_thresholds()
            
            # Update position-level risks
            self.update_position_risks()
            
            # Calculate portfolio VaR
            self.calculate_portfolio_var()
            
            # Check correlation limits
            self.check_correlation_limits()
            
            # Update risk alerts
            self.update_risk_alerts()
            
        except Exception as e:
            logger.error(f"❌ Risk metrics update failed: {e}")
    
    def check_comprehensive_risk_thresholds(self):
        """Check all risk thresholds"""
        try:
            # Daily drawdown check
            if trading_state.daily_pnl < 0:
                daily_drawdown = abs(trading_state.daily_pnl) / self.peak_equity
                
                if daily_drawdown >= config.MAX_DAILY_DRAWDOWN:
                    self.halt_trading(f"Daily drawdown limit exceeded: {daily_drawdown:.2%}")
            
            # Total drawdown check
            if self.current_drawdown >= config.MAX_TOTAL_DRAWDOWN:
                self.halt_trading(f"Total drawdown limit exceeded: {self.current_drawdown:.2%}")
            
            # Emergency drawdown check
            if self.current_drawdown >= config.EMERGENCY_STOP_DRAWDOWN:
                self.emergency_liquidation(f"Emergency drawdown limit exceeded: {self.current_drawdown:.2%}")
            
            # Portfolio concentration check
            total_position_value = sum([
                pos['quantity'] * pos.get('current_price', pos['entry_price'])
                for pos in trading_state.open_positions.values()
            ])
            
            if total_position_value > 0:
                portfolio_exposure = total_position_value / trading_state.current_equity
                if portfolio_exposure > 0.95:  # 95% exposure limit
                    logger.warning(f"⚠️ High portfolio exposure: {portfolio_exposure:.2%}")
            
            # Sector concentration check
            for sector, allocation in trading_state.sector_allocations.items():
                if allocation > config.SECTOR_CONCENTRATION_LIMIT:
                    logger.warning(f"⚠️ High sector concentration in {sector}: {allocation:.2%}")
            
            # Position count check
            if len(trading_state.open_positions) > config.MAX_POSITIONS:
                logger.warning(f"⚠️ Position count limit exceeded: {len(trading_state.open_positions)}")
            
        except Exception as e:
            logger.error(f"❌ Risk threshold check failed: {e}")
    
    def update_position_risks(self):
        """Update individual position risks"""
        try:
            for ticker, position in trading_state.open_positions.items():
                try:
                    entry_price = position['entry_price']
                    current_price = position.get('current_price', entry_price)
                    quantity = position['quantity']
                    
                    # Calculate position metrics
                    unrealized_pnl = (current_price - entry_price) * quantity
                    unrealized_return = (current_price - entry_price) / entry_price
                    position_value = current_price * quantity
                    portfolio_weight = position_value / trading_state.current_equity
                    
                    # Get recent volatility
                    recent_data = get_ultra_advanced_data(ticker, limit=20)
                    if recent_data is not None and not recent_data.empty:
                        returns = recent_data['close'].pct_change().dropna()
                        if len(returns) > 1:
                            volatility = returns.std() * np.sqrt(252)  # Annualized
                        else:
                            volatility = 0.2  # Default 20%
                    else:
                        volatility = 0.2
                    
                    # Calculate position VaR (95% confidence)
                    position_var = position_value * volatility * 1.645 / np.sqrt(252)  # Daily VaR
                    
                    # Store position risk metrics
                    self.position_risks[ticker] = {
                        'unrealized_pnl': unrealized_pnl,
                        'unrealized_return': unrealized_return,
                        'position_value': position_value,
                        'portfolio_weight': portfolio_weight,
                        'volatility': volatility,
                        'var_95': position_var,
                        'risk_score': portfolio_weight * volatility  # Simple risk score
                    }
                    
                    # Check position-specific alerts
                    if unrealized_return < -0.1:  # 10% loss
                        self.add_risk_alert(f"Large position loss in {ticker}: {unrealized_return:.2%}")
                    
                    if portfolio_weight > config.MAX_POSITION_SIZE:
                        self.add_risk_alert(f"Position size limit exceeded for {ticker}: {portfolio_weight:.2%}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Position risk calculation failed for {ticker}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Position risk update failed: {e}")
    
    def calculate_portfolio_var(self):
        """Calculate portfolio Value at Risk"""
        try:
            if len(trading_state.open_positions) < 2:
                return
            
            # Get position weights and volatilities
            tickers = list(trading_state.open_positions.keys())
            weights = []
            volatilities = []
            
            for ticker in tickers:
                if ticker in self.position_risks:
                    weights.append(self.position_risks[ticker]['portfolio_weight'])
                    volatilities.append(self.position_risks[ticker]['volatility'])
                else:
                    weights.append(0.0)
                    volatilities.append(0.2)
            
            weights = np.array(weights)
            volatilities = np.array(volatilities)
            
            # Simple correlation assumption (could be enhanced with actual correlation calculation)
            correlation = 0.3  # Assume 30% correlation between positions
            
            # Portfolio volatility calculation
            portfolio_variance = 0
            for i in range(len(weights)):
                for j in range(len(weights)):
                    if i == j:
                        portfolio_variance += weights[i]**2 * volatilities[i]**2
                    else:
                        portfolio_variance += 2 * weights[i] * weights[j] * volatilities[i] * volatilities[j] * correlation
            
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Portfolio VaR (95% confidence, daily)
            portfolio_var = trading_state.current_equity * portfolio_volatility * 1.645 / np.sqrt(252)
            
            # Check VaR limit
            var_limit = trading_state.current_equity * config.VAR_LIMIT
            if portfolio_var > var_limit:
                self.add_risk_alert(f"Portfolio VaR exceeds limit: ${portfolio_var:,.0f} > ${var_limit:,.0f}")
            
            # Store in trading state
            trading_state.var_95 = portfolio_var / trading_state.current_equity
            
        except Exception as e:
            logger.error(f"❌ Portfolio VaR calculation failed: {e}")
    
    def check_correlation_limits(self):
        """Check position correlation limits"""
        try:
            if len(trading_state.open_positions) < 2:
                return
            
            # Get recent returns for all positions
            tickers = list(trading_state.open_positions.keys())
            returns_data = {}
            
            for ticker in tickers:
                recent_data = get_ultra_advanced_data(ticker, limit=50)
                if recent_data is not None and not recent_data.empty:
                    returns = recent_data['close'].pct_change().dropna()
                    if len(returns) >= 20:  # Minimum data for correlation
                        returns_data[ticker] = returns.tail(20)
            
            if len(returns_data) < 2:
                return
            
            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Check for high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > config.CORRELATION_LIMIT:
                        ticker1 = correlation_matrix.columns[i]
                        ticker2 = correlation_matrix.columns[j]
                        high_correlations.append((ticker1, ticker2, corr))
            
            # Alert on high correlations
            for ticker1, ticker2, corr in high_correlations:
                self.add_risk_alert(f"High correlation between {ticker1} and {ticker2}: {corr:.2f}")
            
            self.correlation_matrix = correlation_matrix
            
        except Exception as e:
            logger.error(f"❌ Correlation check failed: {e}")
    
    def add_risk_alert(self, message: str):
        """Add risk alert"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'message': message,
                'severity': 'warning'
            }
            
            self.risk_alerts.append(alert)
            
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.risk_alerts = [
                alert for alert in self.risk_alerts 
                if alert['timestamp'] > cutoff_time
            ]
            
            logger.warning(f"⚠️ Risk Alert: {message}")
            
        except Exception as e:
            logger.error(f"❌ Risk alert failed: {e}")
    
    def update_risk_alerts(self):
        """Update and clean risk alerts"""
        try:
            # Remove old alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.risk_alerts = [
                alert for alert in self.risk_alerts 
                if alert['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"❌ Risk alert update failed: {e}")
    
    def halt_trading(self, reason: str):
        """Halt trading due to risk breach"""
        try:
            if not self.trading_halted:
                self.trading_halted = True
                self.halt_reason = reason
                trading_state.trading_halted = True
                trading_state.halt_reason = reason
                
                logger.error(f"🛑 TRADING HALTED: {reason}")
                send_discord_alert(f"🛑 TRADING HALTED: {reason}", urgent=True)
                
                # Save state
                trading_state.save_state()
        
        except Exception as e:
            logger.error(f"❌ Trading halt failed: {e}")
    
    def emergency_liquidation(self, reason: str):
        """Emergency liquidation of all positions"""
        try:
            logger.error(f"🚨 EMERGENCY LIQUIDATION INITIATED: {reason}")
            send_discord_alert(f"🚨 EMERGENCY LIQUIDATION: {reason}", urgent=True)
            
            liquidated_positions = []
            
            for ticker in list(trading_state.open_positions.keys()):
                try:
                    # Get current price
                    current_data = get_ultra_advanced_data(ticker, limit=1)
                    if current_data is not None and not current_data.empty:
                        current_price = current_data['close'].iloc[-1]
                    else:
                        # Use last known price
                        current_price = trading_state.open_positions[ticker].get('current_price', 
                                                                               trading_state.open_positions[ticker]['entry_price'])
                    
                    # Remove position
                    trade_record = trading_state.remove_position(ticker, current_price, "emergency_liquidation")
                    if trade_record:
                        liquidated_positions.append(trade_record)
                    
                    logger.error(f"🚨 Emergency liquidation: {ticker} @ ${current_price:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to liquidate {ticker}: {e}")
                    continue
            
            # Halt trading
            self.halt_trading(f"Emergency liquidation completed: {len(liquidated_positions)} positions")
            
            # Send summary
            total_pnl = sum([pos['pnl'] for pos in liquidated_positions])
            send_discord_alert(
                f"🚨 Emergency liquidation completed: {len(liquidated_positions)} positions, Total P&L: ${total_pnl:.2f}",
                urgent=True
            )
            
        except Exception as e:
            logger.error(f"❌ Emergency liquidation failed: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            return {
                'current_drawdown': self.current_drawdown,
                'peak_equity': self.peak_equity,
                'trading_halted': self.trading_halted,
                'halt_reason': self.halt_reason,
                'daily_pnl': trading_state.daily_pnl,
                'total_pnl': trading_state.total_pnl,
                'var_95': trading_state.var_95,
                'position_count': len(trading_state.open_positions),
                'sector_allocations': trading_state.sector_allocations,
                'recent_alerts': len([
                    alert for alert in self.risk_alerts 
                    if alert['timestamp'] > datetime.now() - timedelta(hours=1)
                ]),
                'position_risks': self.position_risks
            }
            
        except Exception as e:
            logger.error(f"❌ Risk summary failed: {e}")
            return {}

risk_monitor = UltraAdvancedRiskMonitor()

# === DISCORD ALERTS ===
def send_discord_alert(message: str, urgent: bool = False):
    """Send enhanced alert to Discord webhook"""
    try:
        webhook_url = config.DISCORD_WEBHOOK_URL
        if not webhook_url:
            return
        
        color = 0xFF0000 if urgent else 0x00FF00
        
        # Add portfolio summary to alerts
        portfolio_summary = trading_state.get_portfolio_summary()
        
        embed_description = f"{message}\n\n"
        embed_description += f"💰 Equity: ${portfolio_summary.get('current_equity', 0):,.2f}\n"
        embed_description += f"📊 Daily P&L: ${portfolio_summary.get('daily_pnl', 0):+.2f}\n"
        embed_description += f"📈 Positions: {portfolio_summary.get('open_positions', 0)}\n"
        embed_description += f"🎯 Win Rate: {portfolio_summary.get('win_rate', 0):.1%}"
        
        payload = {
            "embeds": [{
                "title": "🤖 Ultra Trading Bot Alert",
                "description": embed_description,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": f"Sharpe: {portfolio_summary.get('sharpe_ratio', 0):.2f} | Drawdown: {portfolio_summary.get('current_drawdown', 0):.1%}"
                }
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 204:
            logger.info(f"✅ Discord alert sent: {message[:50]}...")
        else:
            logger.warning(f"⚠️ Discord alert failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Discord alert failed: {e}")

# === ULTRA-ADVANCED TRADING LOGIC ===
def ultra_advanced_enterprise_trading_logic(ticker: str) -> bool:
    """Ultra-advanced trading logic with all enterprise features"""
    try:
        logger.info(f"🔄 Analyzing {ticker} with ultra-advanced enterprise features...")
        
        # Check if trading is halted
        if risk_monitor.trading_halted:
            logger.warning(f"⚠️ Trading halted: {risk_monitor.halt_reason}")
            return False
        
        # Check market status
        if not market_status.is_market_open(include_extended=True):
            logger.info(f"⏰ Market is closed, waiting...")
            return False
        
        # === 1. GET REAL-TIME SIP DATA ===
        sip_quote = None
        if config.SIP_DATA_ENABLED:
            sip_quote = sip_data_manager.get_real_time_quote(ticker)
            if sip_quote and sip_quote.get('latency_ms', 1000) <= config.TARGET_LATENCY_MS:
                current_price = (sip_quote['bid'] + sip_quote['ask']) / 2
                logger.info(f"📡 SIP data for {ticker}: ${current_price:.2f} (latency: {sip_quote['latency_ms']:.1f}ms)")
            else:
                sip_quote = None
        
        # === 2. GET MULTI-TIMEFRAME DATA ===
        data_1min = get_ultra_advanced_data(ticker, limit=200, timeframe=TimeFrame.Minute)
        data_5min = get_ultra_advanced_data(ticker, limit=100, timeframe=TimeFrame.Minute)
        data_daily = get_ultra_advanced_data(ticker, limit=50, timeframe=TimeFrame.Day)
        
        if data_1min is None or len(data_1min) < 50:
            logger.warning(f"⚠️ Insufficient data for {ticker}")
            return False
        
        # Get current price from data if no SIP
        if sip_quote is None:
            current_price = data_1min['close'].iloc[-1]
        
        # === 3. ULTRA-ADVANCED PREDICTIONS ===
        short_pred, medium_pred, long_pred, meta_pred = ensemble_model.predict_ultra_advanced_multi_horizon(
            data_1min, data_daily
        )
        
        logger.info(f"🤖 Predictions for {ticker}: Short={short_pred:.3f}, Medium={medium_pred:.3f}, Long={long_pred:.3f}, Meta={meta_pred:.3f}")
        
        # === 4. COMPREHENSIVE SENTIMENT ANALYSIS ===
        sentiment_data = sentiment_analyzer.analyze_comprehensive_sentiment(ticker)
        sentiment_score = sentiment_data['overall_sentiment']
        sentiment_strength = sentiment_data['sentiment_strength']
        
        logger.info(f"📰 Sentiment for {ticker}: {sentiment_score:.3f} (strength: {sentiment_strength:.3f})")
        
        # === 5. TECHNICAL ANALYSIS ===
        latest_data = data_1min.iloc[-1]
        
        # Volume analysis
        volume_ratio = latest_data.get('volume_ratio_20', 1.0)
        volume_spike = volume_ratio > config.VOLUME_SPIKE_MIN
        volume_confirmation = volume_ratio > config.VOLUME_SPIKE_CONFIRMATION_MIN
        
        # Price momentum
        price_momentum = latest_data.get('price_momentum_5', 0.0)
        momentum_filter = abs(price_momentum) >= config.PRICE_MOMENTUM_MIN
        
        # VWAP analysis
        vwap_deviation = latest_data.get('vwap_deviation', 0.0)
        vwap_filter = abs(vwap_deviation) <= config.VWAP_DEVIATION_THRESHOLD
        
        # Technical indicators
        rsi_14 = latest_data.get('rsi_14', 50)
        macd = latest_data.get('macd', 0)
        macd_signal = latest_data.get('macd_signal', 0)
        bb_position = latest_data.get('bb_position', 0.5)
        atr_14 = latest_data.get('atr_14', 1.0)
        
        # Trend analysis
        trend_strength = latest_data.get('trend_strength', 0.0)
        momentum_composite = latest_data.get('momentum_composite', 0.5)
        
        # === 6. REGIME DETECTION ===
        regime_state = "neutral"
        regime_confidence = 0.5
        
        # Simple regime detection based on volatility and trend
        volatility_20 = latest_data.get('volatility_20', 0.2)
        if volatility_20 > config.VOLATILITY_REGIME_THRESHOLD:
            regime_state = "high_volatility"
        
        if trend_strength > 0.7:
            if price_momentum > 0:
                regime_state = "bullish"
            else:
                regime_state = "bearish"
        
        regime_confidence = min(trend_strength, 1.0)
        
        logger.info(f"🌊 Market regime for {ticker}: {regime_state} (confidence: {regime_confidence:.2f})")
        
        # === 7. POSITION SIZING CALCULATION ===
        base_position_size = config.INITIAL_CAPITAL * config.MAX_POSITION_SIZE
        
        # Confidence-based scaling
        confidence_score = (meta_pred - 0.5) * 2  # Convert to -1 to 1 scale
        confidence_multiplier = 1.0
        
        if config.POSITION_SCALING_ENABLED and abs(confidence_score) > (config.CONFIDENCE_SCALING_THRESHOLD - 0.5) * 2:
            confidence_multiplier = 1 + abs(confidence_score) * 0.5  # Up to 1.5x scaling
            confidence_multiplier = min(confidence_multiplier, config.MAX_POSITION_SCALING)
        
        # Volume confirmation scaling
        volume_multiplier = 1.0
        if volume_confirmation:
            volume_multiplier = 1 + min((volume_ratio - 2.0) * 0.1, 0.3)
        
        # Sentiment scaling
        sentiment_multiplier = 1.0
        if sentiment_strength > 0.3:
            sentiment_multiplier = 1 + sentiment_strength * 0.2
        
        # Calculate final position size
        scaled_position_size = base_position_size * confidence_multiplier * volume_multiplier * sentiment_multiplier
        scaled_position_size = min(scaled_position_size, config.INITIAL_CAPITAL * config.MAX_POSITION_SIZE * config.MAX_POSITION_SCALING)
        
        # Convert to shares
        shares_to_buy = int(scaled_position_size / current_price)
        
        # === 8. RISK CHECKS ===
        # Check if we already have a position
        if ticker in trading_state.open_positions:
            logger.info(f"⚠️ Already have position in {ticker}")
            return False
        
        # Check position limits
        if len(trading_state.open_positions) >= config.MAX_POSITIONS:
            logger.warning(f"⚠️ Maximum positions reached: {len(trading_state.open_positions)}")
            return False
        
        # Check sector concentration
        sector = trading_state.get_ticker_sector(ticker)
        current_sector_allocation = trading_state.sector_allocations.get(sector, 0.0)
        new_position_value = shares_to_buy * current_price
        new_sector_allocation = (current_sector_allocation * trading_state.current_equity + new_position_value) / trading_state.current_equity
        
        if new_sector_allocation > config.MAX_SECTOR_ALLOCATION:
            logger.warning(f"⚠️ Sector concentration limit for {sector}: {new_sector_allocation:.2%}")
            return False
        
        # Check minimum position size
        if shares_to_buy * current_price < config.INITIAL_CAPITAL * config.MIN_POSITION_SIZE:
            logger.info(f"⚠️ Position too small for {ticker}: ${shares_to_buy * current_price:.2f}")
            return False
        
        # === 9. TRADING DECISION LOGIC ===
        buy_signal = False
        sell_signal = False
        
        # Collect all signals
        signals = {
            'short_pred': short_pred > config.SHORT_BUY_THRESHOLD,
            'medium_pred': medium_pred > config.MEDIUM_BUY_THRESHOLD,
            'long_pred': long_pred > config.LONG_BUY_THRESHOLD,
            'meta_pred': meta_pred > 0.6,
            'momentum_filter': momentum_filter,
            'volume_spike': volume_spike,
            'vwap_filter': vwap_filter,
            'sentiment_positive': sentiment_score > 0.1,
            'sentiment_not_negative': sentiment_score > config.SENTIMENT_HOLD_OVERRIDE,
            'rsi_not_overbought': rsi_14 < config.RSI_OVERBOUGHT,
            'macd_bullish': macd > macd_signal,
            'bb_not_overbought': bb_position < 0.8,
            'trend_positive': trend_strength > 0.5 and price_momentum > 0,
            'regime_favorable': regime_state in ['bullish', 'neutral']
        }
        
        # Count positive signals
        positive_signals = sum(signals.values())
        total_signals = len(signals)
        signal_strength = positive_signals / total_signals
        
        logger.info(f"📊 Signal analysis for {ticker}: {positive_signals}/{total_signals} ({signal_strength:.2%})")
        
        # BUY DECISION
        if signal_strength >= 0.7:  # 70% of signals must be positive
            # Additional confirmations for high-confidence trades
            confirmations = 0
            
            if volume_confirmation:
                confirmations += 1
            if sentiment_strength > 0.3:
                confirmations += 1
            if regime_confidence > 0.7:
                confirmations += 1
            if meta_pred > 0.7:
                confirmations += 1
            
            if confirmations >= 2:  # Need at least 2 confirmations
                buy_signal = True
                logger.info(f"✅ BUY signal for {ticker} with {confirmations} confirmations")
        
        # === 10. EXECUTE TRADE ===
        if buy_signal and shares_to_buy > 0:
            try:
                # Calculate stop loss and take profit
                stop_loss_price = current_price * (1 - atr_14 / current_price * 2.0)  # 2x ATR stop
                take_profit_price = current_price * (1 + atr_14 / current_price * 3.0)  # 3x ATR target
                
                # Add position to state
                success = trading_state.add_position(
                    ticker=ticker,
                    quantity=shares_to_buy,
                    entry_price=current_price,
                    confidence=meta_pred,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price
                )
                
                if success:
                    # Log trade details
                    trade_details = {
                        'ticker': ticker,
                        'action': 'BUY',
                        'quantity': shares_to_buy,
                        'price': current_price,
                        'value': shares_to_buy * current_price,
                        'confidence': meta_pred,
                        'signal_strength': signal_strength,
                        'confirmations': confirmations,
                        'sentiment': sentiment_score,
                        'regime': regime_state,
                        'stop_loss': stop_loss_price,
                        'take_profit': take_profit_price
                    }
                    
                    logger.trade(f"📈 BUY EXECUTED: {json.dumps(trade_details, default=str)}")
                    
                    # Send Discord alert
                    alert_message = f"📈 BUY: {ticker} - {shares_to_buy} shares @ ${current_price:.2f}"
                    alert_message += f"\n💰 Value: ${shares_to_buy * current_price:,.2f}"
                    alert_message += f"\n🎯 Confidence: {meta_pred:.2%}"
                    alert_message += f"\n📊 Signals: {positive_signals}/{total_signals}"
                    send_discord_alert(alert_message)
                    
                    # Update risk monitoring
                    risk_monitor.update_comprehensive_risk_metrics()
                    
                    # Record prediction for feedback loop
                    if config.FEEDBACK_LOOP_ENABLED:
                        features_dict = {
                            'short_pred': short_pred,
                            'medium_pred': medium_pred,
                            'long_pred': long_pred,
                            'meta_pred': meta_pred,
                            'sentiment_score': sentiment_score,
                            'volume_ratio': volume_ratio,
                            'price_momentum': price_momentum,
                            'vwap_deviation': vwap_deviation,
                            'rsi_14': rsi_14,
                            'macd': macd,
                            'bb_position': bb_position,
                            'trend_strength': trend_strength,
                            'regime_confidence': regime_confidence
                        }
                        
                        trading_state.pending_feedback[ticker] = {
                            'features': features_dict,
                            'prediction': meta_pred,
                            'entry_time': datetime.now(),
                            'entry_price': current_price
                        }
                    
                    return True
                else:
                    logger.warning(f"⚠️ Failed to add position for {ticker}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Trade execution failed for {ticker}: {e}")
                return False
        
        else:
