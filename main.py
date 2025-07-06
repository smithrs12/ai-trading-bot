#!/usr/bin/env python3
"""
Ultra-Enterprise AI Trading Bot
Advanced algorithmic trading system with enterprise-grade features
"""

import os
import sys
import time
import json
import signal
import argparse
import warnings
import traceback
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import re
import requests
from transformers import pipeline
from brokers import AlpacaBroker, InteractiveBrokersBroker, SimulatedBroker
from backtester import run_backtest
import config

# Core scientific computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import joblib
import pickle

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import optuna

# Deep Learning & AI
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Technical Analysis
import ta
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel

# Trading APIs
from alpaca_trade_api.rest import REST, TimeFrame
import yfinance as yf

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

# NLP & Entity Recognition
import spacy

# Run backtest on top tickers
run_backtest(["AAPL", "MSFT", "NVDA", "TSLA"], days=60)

cooldown_cache = {}

# Reinforcement Learning
try:
    import gym
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Visualization & Dashboard
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Web & API
import requests
from flask import Flask, jsonify, render_template_string
import uvicorn
from fastapi import FastAPI

# Utilities
import pytz
from dotenv import load_dotenv
import schedule
import redis
import sqlite3
import psutil

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis not installed. Feature caching will be disabled.")

try:
    import ccxt
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("⚠️ CCXT not installed. Crypto trading features disabled.")

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

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
    
    # Interactive Brokers
    IB_HOST: str = os.getenv("IB_HOST", "127.0.0.1")
    IB_PORT: int = int(os.getenv("IB_PORT", "7497"))
    IB_CLIENT_ID: int = int(os.getenv("IB_CLIENT_ID", "1"))
    
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
    
    # === NEW ULTRA-ENTERPRISE FEATURES ===
    TRANSFER_LEARNING_ENABLED: bool = True
    HEDGE_OVERLAY_ENABLED: bool = True
    OPTIONS_FLOW_ENABLED: bool = True
    NER_CATALYST_FILTERING: bool = True
    DYNAMIC_META_WEIGHTING: bool = True
    SLIPPAGE_ADJUSTED_REWARDS: bool = True
    BROKER_FAILOVER_ENABLED: bool = True
    GPU_ACCELERATION_ENABLED: bool = True
    
    # Options Flow
    UNUSUAL_WHALES_API_KEY: str = os.getenv("UNUSUAL_WHALES_API_KEY", "")
    
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

ner_model = pipeline("ner", grouped_entities=True)

from datetime import datetime, timedelta

def is_on_cooldown(ticker, cooldown_minutes=10):
    """Check if a ticker is on cooldown"""
    now = datetime.utcnow()
    last_trade_time = cooldown_cache.get(ticker)
    if last_trade_time:
        return (now - last_trade_time).total_seconds() < cooldown_minutes * 60
    return False

# Voting ensemble training
def train_voting_classifier(X, y):
    model1 = LogisticRegression(max_iter=1000)
    model2 = RandomForestClassifier(n_estimators=100)
    model3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    ensemble = VotingClassifier(
        estimators=[('lr', model1), ('rf', model2), ('xgb', model3)],
        voting='soft'
    )
    ensemble.fit(X, y)
    return ensemble

def calculate_kelly_fraction(confidence, reward_risk_ratio=2.0, cap=0.2):
    """
    Calculates Kelly Criterion fraction based on model confidence and reward-risk ratio.
    Clamps to avoid over-leveraging.
    """
    win_rate = confidence
    kelly = (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio
    return max(0.0, min(kelly, cap))

# === Broker Manager ===
class BrokerManager:
    def __init__(self):
        self.brokers = {
            "alpaca": AlpacaBroker(),
            "ib": InteractiveBrokersBroker(),
            "sim": SimulatedBroker()
        }
        self.active_broker = None

    def initialize(self):
        # Try Alpaca first
        if self.brokers["alpaca"].is_available():
            self.active_broker = self.brokers["alpaca"]
        elif self.brokers["ib"].is_available():
            self.active_broker = self.brokers["ib"]
        else:
            self.active_broker = self.brokers["sim"]

    def get_active_broker(self):
        return self.active_broker

def detect_news_catalyst(ticker, max_articles=5):
    try:
        headlines = []
        url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&pageSize={max_articles}&apiKey={os.getenv('NEWS_API_KEY')}"
        resp = requests.get(url)
        data = resp.json()

        for article in data.get("articles", []):
            headline = article["title"]
            if any(keyword in headline.lower() for keyword in ["earnings", "fed", "fda", "merger", "approval", "downgrade", "upgrade"]):
                return True, headline
            ner_tags = ner_model(headline)
            for tag in ner_tags:
                if tag["entity_group"] in ["ORG", "MISC"] and "event" in tag["word"].lower():
                    return True, headline
        return False, None
    except Exception as e:
        print(f"❌ Error in catalyst detection: {e}")
        return False, None

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

# === DEVICE DETECTION FOR GPU ACCELERATION ===
def detect_device():
    """Detect and configure optimal compute device"""
    if config.GPU_ACCELERATION_ENABLED and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ GPU acceleration enabled: {torch.cuda.get_device_name()}")
        return device
    else:
        device = torch.device("cpu")
        logger.info("💻 Using CPU for computations")
        return device

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
device = detect_device()

# === MODULAR BROKER MANAGER ===
class BrokerStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class ModularBrokerManager:
    """Enterprise broker management with failover support"""
    
    def __init__(self):
        self.brokers = {}
        self.primary_broker = None
        self.fallback_broker = None
        self.broker_health = {}
        self.initialize_brokers()
    
    def initialize_brokers(self):
        """Initialize all available brokers"""
        try:
            # Alpaca (Primary)
            if config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY:
                self.brokers['alpaca'] = REST(
                    config.ALPACA_API_KEY,
                    config.ALPACA_SECRET_KEY,
                    base_url=config.ALPACA_BASE_URL
                )
                self.primary_broker = 'alpaca'
                self.broker_health['alpaca'] = BrokerStatus.HEALTHY
                logger.info("✅ Alpaca broker initialized")
            
            # Interactive Brokers (Fallback)
            if config.BROKER_FAILOVER_ENABLED:
                try:
                    from ib_insync import IB
                    self.brokers['ib'] = IB()
                    self.fallback_broker = 'ib'
                    self.broker_health['ib'] = BrokerStatus.HEALTHY
                    logger.info("✅ Interactive Brokers fallback initialized")
                except ImportError:
                    logger.warning("⚠️ IB-Insync not available for fallback")
                except Exception as e:
                    logger.warning(f"⚠️ IB initialization failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Broker initialization failed: {e}")
    
    def get_active_broker(self):
        """Get currently active broker"""
        if self.primary_broker and self.broker_health.get(self.primary_broker) == BrokerStatus.HEALTHY:
            return self.brokers[self.primary_broker]
        elif self.fallback_broker and self.broker_health.get(self.fallback_broker) == BrokerStatus.HEALTHY:
            logger.warning("⚠️ Switching to fallback broker")
            return self.brokers[self.fallback_broker]
        else:
            logger.error("❌ No healthy brokers available")
            return None
    
    def check_broker_health(self, broker_name: str) -> BrokerStatus:
        """Check health of specific broker"""
        try:
            broker = self.brokers.get(broker_name)
            if not broker:
                return BrokerStatus.FAILED
            
            if broker_name == 'alpaca':
                account = broker.get_account()
                if account:
                    return BrokerStatus.HEALTHY
            elif broker_name == 'ib':
                if broker.isConnected():
                    return BrokerStatus.HEALTHY
            
            return BrokerStatus.FAILED
            
        except Exception as e:
            logger.error(f"❌ Broker health check failed for {broker_name}: {e}")
            return BrokerStatus.FAILED
    
    def update_broker_health(self):
        """Update health status for all brokers"""
        for broker_name in self.brokers.keys():
            self.broker_health[broker_name] = self.check_broker_health(broker_name)

broker_manager = ModularBrokerManager()

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

# === SLIPPAGE MODELING ===
class SlippageModel:
    """Advanced slippage modeling and tracking"""
    
    def __init__(self):
        self.historical_slippage = deque(maxlen=1000)
        self.slippage_by_symbol = defaultdict(list)
        self.market_impact_model = None
    
    def estimate_slippage(self, symbol: str, quantity: int, current_price: float, 
                         market_session: str = "regular") -> float:
        """Estimate slippage for a trade"""
        try:
            # Base slippage (bps)
            base_slippage_bps = 4.5
            
            # Adjust for market session
            session_multiplier = {
                'premarket': 2.0,
                'regular': 1.0,
                'afterhours': 1.5,
                'closed': 3.0
            }.get(market_session, 1.0)
            
            # Adjust for position size
            position_value = quantity * current_price
            size_multiplier = 1.0 + (position_value / 100000) * 0.1  # 0.1 per $100k
            
            # Historical adjustment
            if symbol in self.slippage_by_symbol and self.slippage_by_symbol[symbol]:
                historical_avg = np.mean(self.slippage_by_symbol[symbol][-50:])
                historical_multiplier = historical_avg / base_slippage_bps
            else:
                historical_multiplier = 1.0
            
            # Calculate final slippage
            estimated_slippage_bps = (base_slippage_bps * session_multiplier * 
                                    size_multiplier * historical_multiplier)
            
            # Convert to dollar amount
            slippage_amount = (estimated_slippage_bps / 10000) * position_value
            
            return slippage_amount
            
        except Exception as e:
            logger.error(f"❌ Slippage estimation failed: {e}")
            return position_value * 0.0005  # Default 5 bps
    
    def record_actual_slippage(self, symbol: str, expected_price: float, 
                              actual_price: float, quantity: int):
        """Record actual slippage for model improvement"""
        try:
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
            
            self.historical_slippage.append(slippage_bps)
            self.slippage_by_symbol[symbol].append(slippage_bps)
            
            # Keep only recent data
            if len(self.slippage_by_symbol[symbol]) > 100:
                self.slippage_by_symbol[symbol] = self.slippage_by_symbol[symbol][-100:]
                
        except Exception as e:
            logger.error(f"❌ Slippage recording failed: {e}")

slippage_model = SlippageModel()

# === NER-BASED CATALYST FILTERING ===
class CatalystFilter:
    """NER-based event catalyst detection and filtering"""
    
    def __init__(self):
        self.nlp = None
        self.catalyst_keywords = {
            'earnings': ['earnings', 'eps', 'quarterly', 'guidance', 'revenue'],
            'fda': ['fda', 'approval', 'clinical', 'trial', 'drug'],
            'merger': ['merger', 'acquisition', 'buyout', 'takeover'],
            'partnership': ['partnership', 'collaboration', 'joint venture'],
            'product': ['launch', 'product', 'release', 'announcement'],
            'legal': ['lawsuit', 'settlement', 'court', 'litigation'],
            'regulatory': ['regulation', 'compliance', 'investigation']
        }
        self.initialize_nlp()
    
    def initialize_nlp(self):
        """Initialize spaCy NLP model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ NLP model initialized for catalyst filtering")
        except Exception as e:
            logger.warning(f"⚠️ NLP model initialization failed: {e}")
    
    def analyze_news_catalysts(self, ticker: str, news_articles: List[Dict]) -> Dict:
        """Analyze news for potential catalysts"""
        try:
            if not self.nlp or not news_articles:
                return {'catalysts': [], 'multiplier': 1.0}
            
            detected_catalysts = []
            catalyst_scores = defaultdict(float)
            
            for article in news_articles[:10]:  # Analyze top 10 articles
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if not text.strip():
                    continue
                
                # Process with spaCy
                doc = self.nlp(text.lower())
                
                # Extract entities
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                
                # Check for catalyst keywords
                for catalyst_type, keywords in self.catalyst_keywords.items():
                    for keyword in keywords:
                        if keyword in text.lower():
                            catalyst_scores[catalyst_type] += 1
                            if catalyst_type not in detected_catalysts:
                                detected_catalysts.append(catalyst_type)
            
            # Calculate position size multiplier
            multiplier = 1.0
            if detected_catalysts:
                # High-impact catalysts
                if any(cat in ['earnings', 'fda', 'merger'] for cat in detected_catalysts):
                    multiplier = 1.3
                # Medium-impact catalysts
                elif any(cat in ['partnership', 'product'] for cat in detected_catalysts):
                    multiplier = 1.15
                # Low-impact catalysts
                else:
                    multiplier = 1.05
            
            return {
                'catalysts': detected_catalysts,
                'multiplier': multiplier,
                'scores': dict(catalyst_scores)
            }
            
        except Exception as e:
            logger.error(f"❌ Catalyst analysis failed for {ticker}: {e}")
            return {'catalysts': [], 'multiplier': 1.0}

catalyst_filter = CatalystFilter()

# === OPTIONS FLOW ANALYZER ===
class OptionsFlowAnalyzer:
    """Options flow analysis for sentiment and positioning"""
    
    def __init__(self):
        self.unusual_whales_api = config.UNUSUAL_WHALES_API_KEY
        self.options_cache = {}
    
    def get_options_flow(self, symbol: str) -> Dict:
        """Get options flow data for symbol"""
        try:
            if not config.OPTIONS_FLOW_ENABLED or not self.unusual_whales_api:
                return self._generate_mock_options_data(symbol)
            
            # Cache key
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self.options_cache:
                return self.options_cache[cache_key]
            
            # Fetch from Unusual Whales API (mock implementation)
            options_data = self._fetch_unusual_whales_data(symbol)
            
            # Cache results
            self.options_cache[cache_key] = options_data
            
            return options_data
            
        except Exception as e:
            logger.error(f"❌ Options flow analysis failed for {symbol}: {e}")
            return self._generate_mock_options_data(symbol)
    
    def _fetch_unusual_whales_data(self, symbol: str) -> Dict:
        """Fetch data from Unusual Whales API"""
        try:
            # This would be the actual API call
            # For now, return mock data
            return self._generate_mock_options_data(symbol)
            
        except Exception as e:
            logger.error(f"❌ Unusual Whales API failed: {e}")
            return self._generate_mock_options_data(symbol)
    
    def _generate_mock_options_data(self, symbol: str) -> Dict:
        """Generate mock options flow data"""
        try:
            # Simulate options flow based on recent price action
            recent_data = yf.download(symbol, period="5d", interval="1d")
            
            if recent_data.empty:
                return {'call_volume': 0, 'put_volume': 0, 'sentiment': 0.0}
            
            recent_return = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1)
            
            # Simulate based on recent performance
            if recent_return > 0.02:  # Strong positive
                call_volume = np.random.randint(1000, 5000)
                put_volume = np.random.randint(200, 1000)
                sentiment = 0.7
            elif recent_return < -0.02:  # Strong negative
                call_volume = np.random.randint(200, 1000)
                put_volume = np.random.randint(1000, 5000)
                sentiment = -0.7
            else:  # Neutral
                call_volume = np.random.randint(500, 2000)
                put_volume = np.random.randint(500, 2000)
                sentiment = 0.0
            
            return {
                'call_volume': call_volume,
                'put_volume': put_volume,
                'call_put_ratio': call_volume / max(put_volume, 1),
                'sentiment': sentiment,
                'unusual_activity': call_volume + put_volume > 3000
            }
            
        except Exception as e:
            logger.error(f"❌ Mock options data generation failed: {e}")
            return {'call_volume': 0, 'put_volume': 0, 'sentiment': 0.0}

options_analyzer = OptionsFlowAnalyzer()

# === HEDGE OVERLAY MANAGER ===
class HedgeOverlayManager:
    """SPY/VIX hedge overlay system"""
    
    def __init__(self):
        self.hedge_active = False
        self.spy_position = 0
        self.vix_threshold = 25.0
        self.drawdown_threshold = 0.05
        self.hedge_ratio = 0.3
    
    def should_activate_hedge(self, current_drawdown: float, vix_level: float) -> bool:
        """Determine if hedge should be activated"""
        try:
            if not config.HEDGE_OVERLAY_ENABLED:
                return False
            
            # Activate hedge if:
            # 1. Drawdown exceeds threshold, OR
            # 2. VIX exceeds threshold
            should_hedge = (current_drawdown > self.drawdown_threshold or 
                          vix_level > self.vix_threshold)
            
            return should_hedge
            
        except Exception as e:
            logger.error(f"❌ Hedge activation check failed: {e}")
            return False
    
    def calculate_hedge_size(self, portfolio_value: float, vix_level: float) -> int:
        """Calculate appropriate hedge size"""
        try:
            # Base hedge ratio
            base_ratio = self.hedge_ratio
            
            # Adjust based on VIX level
            vix_multiplier = min(vix_level / 20.0, 2.0)  # Cap at 2x
            
            # Calculate hedge value
            hedge_value = portfolio_value * base_ratio * vix_multiplier
            
            # Get SPY price
            spy_data = yf.download("SPY", period="1d", interval="1m")
            if not spy_data.empty:
                spy_price = spy_data['Close'].iloc[-1]
                hedge_shares = int(hedge_value / spy_price)
                return -hedge_shares  # Negative for short position
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Hedge size calculation failed: {e}")
            return 0
    
    def get_vix_level(self) -> float:
        """Get current VIX level"""
        try:
            vix_data = yf.download("^VIX", period="1d", interval="1m")
            if not vix_data.empty:
                return float(vix_data['Close'].iloc[-1])
            return 20.0  # Default VIX level
            
        except Exception as e:
            logger.error(f"❌ VIX level fetch failed: {e}")
            return 20.0

hedge_manager = HedgeOverlayManager()

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
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
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
                pos['quantity'] * pos['current_price'] 
                for pos in self.open_positions.values() 
                if self.get_ticker_sector(pos['ticker']) == sector
            ])
            
            position_value = abs(quantity) * entry_price
            new_sector_allocation = (sector_value + position_value) / self.current_equity
            
            if new_sector_allocation > config.MAX_SECTOR_ALLOCATION:
                logger.warning(f"⚠️ Sector concentration limit exceeded for {sector}")
                return False
            
            # Add position
            self.open_positions[ticker] = {
                'ticker': ticker,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': entry_price,
                'unrealized_pnl': 0.0,
                'sector': sector,
                'confidence': confidence,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            # Update tracking
            self.position_entry_times[ticker] = datetime.now()
            self.position_stop_losses[ticker] = stop_loss
            self.position_take_profits[ticker] = take_profit
            self.position_sectors[ticker] = sector
            self.position_confidence[ticker] = confidence
            
            logger.info(f"✅ Added position: {ticker} {quantity} shares @ ${entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add position for {ticker}: {e}")
            return False
    
    def close_position(self, ticker: str, exit_price: float, reason: str = "signal") -> bool:
        """Close position with comprehensive tracking"""
        try:
            if ticker not in self.open_positions:
                logger.warning(f"⚠️ No open position for {ticker}")
                return False
            
            position = self.open_positions[ticker]
            
            # Calculate P&L
            pnl = (exit_price - position['entry_price']) * position['quantity']
            
            # Record trade
            trade_record = {
                'ticker': ticker,
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'hold_duration': datetime.now() - position['entry_time'],
                'sector': position['sector'],
                'confidence': position['confidence'],
                'exit_reason': reason
            }
            
            self.trade_history.append(trade_record)
            
            # Update statistics
            self.total_trades += 1
            self.total_pnl += pnl
            self.current_equity += pnl
            
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Update performance metrics
            self.update_performance_metrics()
            
            # Remove position
            del self.open_positions[ticker]
            
            # Clean up tracking
            if ticker in self.position_entry_times:
                del self.position_entry_times[ticker]
            if ticker in self.position_stop_losses:
                del self.position_stop_losses[ticker]
            if ticker in self.position_take_profits:
                del self.position_take_profits[ticker]
            if ticker in self.position_sectors:
                del self.position_sectors[ticker]
            if ticker in self.position_confidence:
                del self.position_confidence[ticker]
            
            logger.info(f"✅ Closed position: {ticker} P&L: ${pnl:.2f} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to close position for {ticker}: {e}")
            return False
    
    def update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            if self.total_trades > 0:
                self.win_rate = self.winning_trades / self.total_trades
                
                # Calculate average win/loss
                wins = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
                losses = [trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
                
                self.avg_win = np.mean(wins) if wins else 0
                self.avg_loss = np.mean(losses) if losses else 0
                
                # Profit factor
                total_wins = sum(wins) if wins else 0
                total_losses = abs(sum(losses)) if losses else 1
                self.profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Drawdown calculation
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Risk metrics (simplified)
            if len(self.trade_history) >= 30:
                returns = [trade['pnl'] / config.INITIAL_CAPITAL for trade in self.trade_history[-30:]]
                
                if returns:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    # Sharpe ratio (annualized)
                    if std_return > 0:
                        self.sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
                    
                    # VaR calculation
                    self.var_95 = np.percentile(returns, 5) * self.current_equity
                    
        except Exception as e:
            logger.error(f"❌ Performance metrics update failed: {e}")
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get current sector allocation"""
        try:
            sector_values = defaultdict(float)
            total_value = 0
            
            for position in self.open_positions.values():
                sector = position['sector']
                value = abs(position['quantity']) * position['current_price']
                sector_values[sector] += value
                total_value += value
            
            if total_value > 0:
                return {sector: value / total_value for sector, value in sector_values.items()}
            else:
                return {}
                
        except Exception as e:
            logger.error(f"❌ Sector allocation calculation failed: {e}")
            return {}
    
    def save_state(self):
        """Save trading state to disk"""
        try:
            state_data = {
                'open_positions': self.open_positions,
                'trade_history': self.trade_history[-1000:],  # Keep last 1000 trades
                'current_equity': self.current_equity,
                'total_pnl': self.total_pnl,
                'max_drawdown': self.max_drawdown,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'models_trained': self.models_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs("data", exist_ok=True)
            with open("data/trading_state.json", "w") as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")
    
    def load_state(self):
        """Load trading state from disk"""
        try:
            if os.path.exists("data/trading_state.json"):
                with open("data/trading_state.json", "r") as f:
                    state_data = json.load(f)
                
                self.open_positions = state_data.get('open_positions', {})
                self.trade_history = state_data.get('trade_history', [])
                self.current_equity = state_data.get('current_equity', config.INITIAL_CAPITAL)
                self.total_pnl = state_data.get('total_pnl', 0.0)
                self.max_drawdown = state_data.get('max_drawdown', 0.0)
                self.total_trades = state_data.get('total_trades', 0)
                self.winning_trades = state_data.get('winning_trades', 0)
                self.models_trained = state_data.get('models_trained', False)
                
                logger.info(f"✅ State loaded - Equity: ${self.current_equity:,.2f}, Positions: {len(self.open_positions)}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load state: {e}")

trading_state = UltraAdvancedTradingState()

# === ULTRA-ADVANCED DATA MANAGER ===
class UltraAdvancedDataManager:
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = config.CACHE_TTL
        self.redis_client = None
        self.initialize_cache()
    
    def initialize_cache(self):
        """Initialize caching system"""
        try:
            if config.REDIS_ENABLED and REDIS_AVAILABLE:
                self.redis_client = redis.Redis(
                    host=config.REDIS_HOST,
                    port=config.REDIS_PORT,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization failed, using memory cache: {e}")
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            
            # Fallback to memory cache
            if key in self.cache:
                timestamp = self.cache_timestamps.get(key, 0)
                if time.time() - timestamp < self.cache_ttl:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.cache_timestamps[key]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache get failed for {key}: {e}")
            return None
    
    def set_cached_data(self, key: str, data: Any, ttl: Optional[int] = None):
        """Set data in cache"""
        try:
            ttl = ttl or self.cache_ttl
            
            # Try Redis first
            if self.redis_client:
                self.redis_client.setex(key, ttl, json.dumps(data, default=str))
            
            # Always store in memory cache as backup
            self.cache[key] = data
            self.cache_timestamps[key] = time.time()
            
        except Exception as e:
            logger.error(f"❌ Cache set failed for {key}: {e}")
    
    def get_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Get stock data with caching"""
        cache_key = f"stock_data_{ticker}_{period}_{interval}"
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data is not None:
            try:
                return pd.DataFrame(cached_data)
            except Exception:
                pass
        
        try:
            # Fetch from yfinance
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if not data.empty:
                # Cache the data
                self.set_cached_data(cache_key, data.to_dict(), ttl=300)  # 5 minutes
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {ticker}: {e}")
            return None
    
    def get_multiple_stock_data(self, tickers: List[str], period: str = "1y", 
                               interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks efficiently"""
        results = {}
        uncached_tickers = []
        
        # Check cache for each ticker
        for ticker in tickers:
            cache_key = f"stock_data_{ticker}_{period}_{interval}"
            cached_data = self.get_cached_data(cache_key)
            
            if cached_data is not None:
                try:
                    results[ticker] = pd.DataFrame(cached_data)
                except Exception:
                    uncached_tickers.append(ticker)
            else:
                uncached_tickers.append(ticker)
        
        # Fetch uncached data in batch
        if uncached_tickers:
            try:
                # Batch download
                batch_data = yf.download(uncached_tickers, period=period, 
                                       interval=interval, progress=False, group_by='ticker')
                
                if not batch_data.empty:
                    for ticker in uncached_tickers:
                        try:
                            if len(uncached_tickers) == 1:
                                ticker_data = batch_data
                            else:
                                ticker_data = batch_data[ticker]
                            
                            if not ticker_data.empty:
                                results[ticker] = ticker_data
                                
                                # Cache the data
                                cache_key = f"stock_data_{ticker}_{period}_{interval}"
                                self.set_cached_data(cache_key, ticker_data.to_dict(), ttl=300)
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to process data for {ticker}: {e}")
                            
            except Exception as e:
                logger.error(f"❌ Batch data fetch failed: {e}")
        
        return results
    
    def get_real_time_quote(self, ticker: str) -> Optional[Dict]:
        """Get real-time quote with SIP data integration"""
        try:
            # Try SIP data first
            if config.SIP_DATA_ENABLED:
                quote = sip_data_manager.get_real_time_quote(ticker)
                if quote:
                    return quote
            
            # Fallback to yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                return {
                    'symbol': ticker,
                    'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'volume': info.get('volume', 0),
                    'timestamp': time.time(),
                    'provider': 'yfinance'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Real-time quote failed for {ticker}: {e}")
            return None

data_manager = UltraAdvancedDataManager()

# === ULTRA-ADVANCED ENSEMBLE MODEL ===
class UltraAdvancedEnsembleModel:
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.model_weights = {}
        self.performance_history = defaultdict(list)
        self.device = device
        
        # Transfer learning components
        self.pretrained_features = None
        self.transfer_model = None
        
        # Dynamic weighting
        self.dynamic_weights = {}
        self.weight_decay = 0.95
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize ensemble models with GPU support"""
        try:
            # XGBoost with GPU support
            if config.GPU_ACCELERATION_ENABLED and torch.cuda.is_available():
                xgb_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            else:
                xgb_params = {'tree_method': 'hist'}
            
            self.models = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    **xgb_params
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    device='gpu' if config.GPU_ACCELERATION_ENABLED and torch.cuda.is_available() else 'cpu'
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
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
            
            # Initialize dynamic weights
            self.dynamic_weights = {name: 1.0 for name in self.models.keys()}
            
            # Neural network model
            if config.GPU_ACCELERATION_ENABLED:
                self.models['neural_network'] = self._create_neural_network()
                self.dynamic_weights['neural_network'] = 1.0
            
            logger.info(f"✅ Initialized {len(self.models)} ensemble models")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
    
    def _create_neural_network(self):
        """Create neural network model"""
        class TradingNN(nn.Module):
            def __init__(self, input_size):
                super(TradingNN, self).__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 3)  # 3 classes: sell, hold, buy
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.softmax(self.fc4(x))
                return x
        
        return TradingNN
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for training"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['returns_1d'] = data['Close'].pct_change()
            features['returns_5d'] = data['Close'].pct_change(5)
            features['returns_10d'] = data['Close'].pct_change(10)
            features['returns_20d'] = data['Close'].pct_change(20)
            
            # Volatility features
            features['volatility_5d'] = features['returns_1d'].rolling(5).std()
            features['volatility_20d'] = features['returns_1d'].rolling(20).std()
            
            # Technical indicators
            if TALIB_AVAILABLE:
                # TA-Lib indicators
                features['rsi'] = talib.RSI(data['Close'].values)
                features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(data['Close'].values)
                features['bb_upper'], features['bb_middle'], features['bb_lower'] = talib.BBANDS(data['Close'].values)
                features['atr'] = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values)
            else:
                # Fallback to ta library
                features['rsi'] = RSIIndicator(data['Close']).rsi()
                macd = MACD(data['Close'])
                features['macd'] = macd.macd()
                features['macd_signal'] = macd.macd_signal()
                features['macd_hist'] = macd.macd_diff()
                
                bb = BollingerBands(data['Close'])
                features['bb_upper'] = bb.bollinger_hband()
                features['bb_lower'] = bb.bollinger_lband()
                features['bb_middle'] = bb.bollinger_mavg()
                
                features['atr'] = AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = data['Close'].rolling(period).mean()
                features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
                features[f'price_to_sma_{period}'] = data['Close'] / features[f'sma_{period}']
            
            # Volume features
            features['volume_sma_20'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
            features['price_volume'] = data['Close'] * data['Volume']
            
            # Advanced features
            features['high_low_ratio'] = data['High'] / data['Low']
            features['close_to_high'] = data['Close'] / data['High']
            features['close_to_low'] = data['Close'] / data['Low']
            
            # Momentum features
            features['momentum_5'] = data['Close'] / data['Close'].shift(5)
            features['momentum_10'] = data['Close'] / data['Close'].shift(10)
            
            # Support/Resistance levels
            features['support_20'] = data['Low'].rolling(20).min()
            features['resistance_20'] = data['High'].rolling(20).max()
            features['support_distance'] = (data['Close'] - features['support_20']) / data['Close']
            features['resistance_distance'] = (features['resistance_20'] - data['Close']) / data['Close']
            
            # Market structure
            features['higher_high'] = (data['High'] > data['High'].shift(1)).astype(int)
            features['lower_low'] = (data['Low'] < data['Low'].shift(1)).astype(int)
            
            # Time-based features
            features['day_of_week'] = pd.to_datetime(data.index).dayofweek
            features['month'] = pd.to_datetime(data.index).month
            features['quarter'] = pd.to_datetime(data.index).quarter
            
            # Drop NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def create_labels(self, data: pd.DataFrame, lookahead_days: int = 5) -> pd.Series:
        """Create trading labels based on future returns"""
        try:
            future_returns = data['Close'].shift(-lookahead_days) / data['Close'] - 1
            
            # Create labels: 0=sell, 1=hold, 2=buy
            labels = pd.Series(1, index=data.index)  # Default to hold
            
            # Buy signals (top 30% of returns)
            buy_threshold = future_returns.quantile(0.7)
            labels[future_returns > buy_threshold] = 2
            
            # Sell signals (bottom 30% of returns)
            sell_threshold = future_returns.quantile(0.3)
            labels[future_returns < sell_threshold] = 0
            
            return labels.dropna()
            
        except Exception as e:
            logger.error(f"❌ Label creation failed: {e}")
            return pd.Series()
    
    def train_models(self, training_data: Dict[str, pd.DataFrame]) -> bool:
        """Train ensemble models with transfer learning"""
        try:
            if len(training_data) < config.MIN_TICKERS_FOR_TRAINING:
                logger.warning(f"⚠️ Insufficient training data: {len(training_data)} tickers")
                return False
            
            logger.info(f"🚀 Training models on {len(training_data)} tickers...")
            
            # Combine all training data
            all_features = []
            all_labels = []
            
            for ticker, data in training_data.items():
                if data.empty:
                    continue
                
                features = self.prepare_features(data)
                labels = self.create_labels(data)
                
                if features.empty or labels.empty:
                    continue
                
                # Align features and labels
                common_index = features.index.intersection(labels.index)
                if len(common_index) < 50:  # Minimum samples per ticker
                    continue
                
                ticker_features = features.loc[common_index]
                ticker_labels = labels.loc[common_index]
                
                all_features.append(ticker_features)
                all_labels.append(ticker_labels)
            
            if not all_features:
                logger.error("❌ No valid training data prepared")
                return False
            
            # Combine all data
            X = pd.concat(all_features, ignore_index=True)
            y = pd.concat(all_labels, ignore_index=True)
            
            logger.info(f"📊 Training data shape: {X.shape}, Labels: {len(y)}")
            
            # Feature selection
            if config.FEATURE_SELECTION_ENABLED:
                self.feature_selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
                X = self.feature_selector.fit_transform(X, y)
                logger.info(f"✅ Selected {X.shape[1]} features")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # === Add Voting Ensemble ===
            try:
                voting_clf = VotingClassifier(
                    estimators=[
                        ('lr', LogisticRegression(max_iter=1000)),
                        ('rf', RandomForestClassifier(n_estimators=100)),
                        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
                    ],
                    voting='soft',
                    weights=[1, 1, 2]  # Optional: give XGB more influence
                )
                voting_clf.fit(X_scaled, y)
                self.models['voting_ensemble'] = voting_clf
                score = voting_clf.score(X_scaled, y)
                model_scores = {}  # initialize here since we're scoring
                model_scores['voting_ensemble'] = score
                self.performance_history['voting_ensemble'].append(score)
                logger.info(f"✅ voting_ensemble: {score:.4f}")
            except Exception as e:
                logger.error(f"❌ Training failed for voting_ensemble: {e}")
            
            # Train individual models
            model_scores = {}
            
            for name, model in self.models.items():
                try:
                    if name == 'neural_network':
                        score = self._train_neural_network(X_scaled, y)
                    else:
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                        score = cv_scores.mean()
                        
                        # Train on full dataset
                        model.fit(X_scaled, y)
                    
                    model_scores[name] = score
                    self.performance_history[name].append(score)
                    
                    logger.info(f"✅ {name}: {score:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Training failed for {name}: {e}")
                    model_scores[name] = 0.0
            
            # Update dynamic weights based on performance
            self._update_dynamic_weights(model_scores)
            
            # Train meta-model
            if len(model_scores) > 1:
                self._train_meta_model(X_scaled, y)
            
            self.is_trained = True
            trading_state.models_trained = True
            
            logger.info("✅ Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train neural network model"""
        try:
            if 'neural_network' not in self.models:
                return 0.0
            
            # Create model instance
            input_size = X.shape[1]
            model = self.models['neural_network'](input_size).to(self.device)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y.values).to(self.device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            
            # Store trained model
            self.models['neural_network'] = model
            
            return accuracy
            
        except Exception as e:
            logger.error(f"❌ Neural network training failed: {e}")
            return 0.0
    
    def _update_dynamic_weights(self, model_scores: Dict[str, float]):
        """Update dynamic model weights based on performance"""
        try:
            if not model_scores:
                return
            
            # Normalize scores
            total_score = sum(model_scores.values())
            if total_score > 0:
                for name, score in model_scores.items():
                    # Exponential weighting based on performance
                    self.dynamic_weights[name] = (score / total_score) ** 2
            
            # Apply decay to previous weights
            for name in self.dynamic_weights:
                if name not in model_scores:
                    self.dynamic_weights[name] *= self.weight_decay
            
            # Normalize weights
            total_weight = sum(self.dynamic_weights.values())
            if total_weight > 0:
                for name in self.dynamic_weights:
                    self.dynamic_weights[name] /= total_weight
            
            logger.info(f"📊 Updated dynamic weights: {self.dynamic_weights}")
            
        except Exception as e:
            logger.error(f"❌ Dynamic weight update failed: {e}")
    
    def _train_meta_model(self, X: np.ndarray, y: np.ndarray):
        """Train meta-model for ensemble combination"""
        try:
            # Get predictions from all models
            meta_features = []
            
            for name, model in self.models.items():
                if name == 'neural_network':
                    continue  # Skip neural network for meta-model
                
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        meta_features.append(proba)
                    else:
                        pred = model.predict(X).reshape(-1, 1)
                        meta_features.append(pred)
                except Exception:
                    continue
            
            if meta_features:
                X_meta = np.hstack(meta_features)
                
                # Train meta-model
                self.meta_model = LogisticRegression(random_state=42)
                self.meta_model.fit(X_meta, y)
                
                logger.info("✅ Meta-model trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Meta-model training failed: {e}")
    
    def predict(self, features: pd.DataFrame) -> Dict[str, float]:
        """Make ensemble predictions with dynamic weighting"""
        try:
            if not self.is_trained or features.empty:
                return {'buy_prob': 0.33, 'hold_prob': 0.34, 'sell_prob': 0.33}
            
            # Prepare features
            if self.feature_selector:
                X = self.feature_selector.transform(features)
            else:
                X = features.values
            
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                try:
                    if name == 'neural_network':
                        pred_proba = self._predict_neural_network(X_scaled)
                    else:
                        if hasattr(model, 'predict_proba'):
                            pred_proba = model.predict_proba(X_scaled)
                        else:
                            pred = model.predict(X_scaled)
                            # Convert to probabilities
                            pred_proba = np.zeros((len(pred), 3))
                            for i, p in enumerate(pred):
                                pred_proba[i, int(p)] = 1.0
                    
                    if pred_proba is not None and len(pred_proba) > 0:
                        probabilities[name] = pred_proba[-1]  # Last prediction
                        
                except Exception as e:
                    logger.warning(f"⚠️ Prediction failed for {name}: {e}")
                    continue
            
            if not probabilities:
                return {'buy_prob': 0.33, 'hold_prob': 0.34, 'sell_prob': 0.33}
            
            # Weighted ensemble
            ensemble_proba = np.zeros(3)
            total_weight = 0
            
            for name, proba in probabilities.items():
                weight = self.dynamic_weights.get(name, 1.0)
                ensemble_proba += weight * proba
                total_weight += weight
            
            if total_weight > 0:
                ensemble_proba /= total_weight
            
            return {
                'sell_prob': float(ensemble_proba[0]),
                'hold_prob': float(ensemble_proba[1]),
                'buy_prob': float(ensemble_proba[2])
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            return {'buy_prob': 0.33, 'hold_prob': 0.34, 'sell_prob': 0.33}
    
    def _predict_neural_network(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make prediction with neural network"""
        try:
            if 'neural_network' not in self.models:
                return None
            
            model = self.models['neural_network']
            if not isinstance(model, nn.Module):
                return None
            
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = model(X_tensor)
                return outputs.cpu().numpy()
            
        except Exception as e:
            logger.error(f"❌ Neural network prediction failed: {e}")
            return None

ensemble_model = UltraAdvancedEnsembleModel()

# === SENTIMENT ANALYZER ===
class UltraAdvancedSentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.news_api = api_manager.news_api
        self.sentiment_cache = {}
        self.transformer_pipeline = None
        self.initialize_transformer()
    
    def initialize_transformer(self):
        """Initialize transformer-based sentiment analysis"""
        try:
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if config.GPU_ACCELERATION_ENABLED and torch.cuda.is_available() else -1
            )
            logger.info("✅ FinBERT sentiment analyzer initialized")
        except Exception as e:
            logger.warning(f"⚠️ Transformer sentiment initialization failed: {e}")
    
    def get_news_sentiment(self, ticker: str, days_back: int = 3) -> Dict[str, float]:
        """Get comprehensive news sentiment for ticker"""
        try:
            cache_key = f"news_sentiment_{ticker}_{days_back}"
            
            # Check cache
            cached_result = data_manager.get_cached_data(cache_key)
            if cached_result:
                return cached_result
            
            # Get news articles
            articles = self._fetch_news_articles(ticker, days_back)
            
            if not articles:
                return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            
            # Analyze sentiment
            sentiments = []
            confidences = []
            
            for article in articles[:20]:  # Limit to 20 articles
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if not text.strip():
                    continue
                
                # VADER sentiment
                vader_scores = self.vader_analyzer.polarity_scores(text)
                vader_sentiment = vader_scores['compound']
                
                # Transformer sentiment (if available)
                transformer_sentiment = 0.0
                transformer_confidence = 0.0
                
                if self.transformer_pipeline:
                    try:
                        result = self.transformer_pipeline(text[:512])  # Limit text length
                        if result:
                            label = result[0]['label']
                            score = result[0]['score']
                            
                            if label == 'POSITIVE':
                                transformer_sentiment = score
                            elif label == 'NEGATIVE':
                                transformer_sentiment = -score
                            
                            transformer_confidence = score
                    except Exception:
                        pass
                
                # Combine sentiments
                if transformer_confidence > 0.7:
                    final_sentiment = transformer_sentiment
                    final_confidence = transformer_confidence
                else:
                    final_sentiment = vader_sentiment
                    final_confidence = abs(vader_sentiment)
                
                sentiments.append(final_sentiment)
                confidences.append(final_confidence)
            
            if not sentiments:
                result = {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            else:
                # Weight by confidence
                weighted_sentiment = np.average(sentiments, weights=confidences) if confidences else np.mean(sentiments)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                result = {
                    'sentiment': float(weighted_sentiment),
                    'confidence': float(avg_confidence),
                    'article_count': len(sentiments)
                }
            
            # Cache result
            data_manager.set_cached_data(cache_key, result, ttl=1800)  # 30 minutes
            
            return result
            
        except Exception as e:
            logger.error(f"❌ News sentiment analysis failed for {ticker}: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
    
    def _fetch_news_articles(self, ticker: str, days_back: int) -> List[Dict]:
        """Fetch news articles for ticker"""
        try:
            if not self.news_api:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Search for news
            articles = self.news_api.get_everything(
                q=f"{ticker} OR {self._get_company_name(ticker)}",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=50
            )
            
            return articles.get('articles', [])
            
        except Exception as e:
            logger.error(f"❌ News fetch failed for {ticker}: {e}")
            return []
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for ticker"""
        company_names = {
            'AAPL': 'Apple Inc',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc',
            'AMZN': 'Amazon.com Inc',
            'TSLA': 'Tesla Inc',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc',
            'NFLX': 'Netflix Inc'
        }
        return company_names.get(ticker, ticker)

sentiment_analyzer = UltraAdvancedSentimentAnalyzer()

# === RISK MONITOR ===
class UltraAdvancedRiskMonitor:
    def __init__(self):
        self.risk_metrics = {}
        self.alerts = []
        self.position_risks = {}
        self.portfolio_var = 0.0
        self.correlation_matrix = None
        self.risk_limits = {
            'max_position_size': config.MAX_POSITION_SIZE,
            'max_sector_allocation': config.MAX_SECTOR_ALLOCATION,
            'max_daily_drawdown': config.MAX_DAILY_DRAWDOWN,
            'max_total_drawdown': config.MAX_TOTAL_DRAWDOWN,
            'max_correlation': config.CORRELATION_LIMIT,
            'max_var': config.VAR_LIMIT
        }
    
    def check_pre_trade_risk(self, ticker: str, quantity: int, price: float) -> Dict[str, Any]:
        """Comprehensive pre-trade risk check"""
        try:
            risk_check = {
                'approved': True,
                'warnings': [],
                'rejections': [],
                'risk_score': 0.0
            }
            
            position_value = abs(quantity) * price
            portfolio_value = trading_state.current_equity
            
            # Position size check
            position_pct = position_value / portfolio_value
            if position_pct > self.risk_limits['max_position_size']:
                risk_check['approved'] = False
                risk_check['rejections'].append(f"Position size {position_pct:.2%} exceeds limit {self.risk_limits['max_position_size']:.2%}")
            elif position_pct > self.risk_limits['max_position_size'] * 0.8:
                risk_check['warnings'].append(f"Position size {position_pct:.2%} approaching limit")
            
            # Sector concentration check
            sector = trading_state.get_ticker_sector(ticker)
            current_sector_allocation = trading_state.get_sector_allocation().get(sector, 0.0)
            new_sector_allocation = current_sector_allocation + position_pct
            
            if new_sector_allocation > self.risk_limits['max_sector_allocation']:
                risk_check['approved'] = False
                risk_check['rejections'].append(f"Sector allocation {new_sector_allocation:.2%} exceeds limit")
            
            # Correlation check
            correlation_risk = self._check_correlation_risk(ticker, quantity)
            if correlation_risk['high_correlation']:
                risk_check['warnings'].append(f"High correlation with existing positions: {correlation_risk['max_correlation']:.2f}")
            
            # Drawdown check
            if trading_state.current_drawdown > self.risk_limits['max_daily_drawdown']:
                risk_check['approved'] = False
                risk_check['rejections'].append(f"Current drawdown {trading_state.current_drawdown:.2%} exceeds daily limit")
            
            # Calculate risk score
            risk_score = (
                position_pct * 2 +
                new_sector_allocation * 1.5 +
                correlation_risk['max_correlation'] * 1.0 +
                trading_state.current_drawdown * 3.0
            )
            risk_check['risk_score'] = risk_score
            
            return risk_check
            
        except Exception as e:
            logger.error(f"❌ Pre-trade risk check failed: {e}")
            return {'approved': False, 'warnings': [], 'rejections': ['Risk check failed'], 'risk_score': 1.0}
    
    def _check_correlation_risk(self, ticker: str, quantity: int) -> Dict[str, Any]:
        """Check correlation risk with existing positions"""
        try:
            if not trading_state.open_positions:
                return {'high_correlation': False, 'max_correlation': 0.0}
            
            # Get recent price data for correlation calculation
            tickers = [ticker] + list(trading_state.open_positions.keys())
            price_data = data_manager.get_multiple_stock_data(tickers, period="3mo", interval="1d")
            
            if len(price_data) < 2:
                return {'high_correlation': False, 'max_correlation': 0.0}
            
            # Calculate returns
            returns_data = {}
            for t, data in price_data.items():
                if not data.empty:
                    returns_data[t] = data['Close'].pct_change().dropna()
            
            if len(returns_data) < 2:
                return {'high_correlation': False, 'max_correlation': 0.0}
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if returns_df.empty or ticker not in returns_df.columns:
                return {'high_correlation': False, 'max_correlation': 0.0}
            
            # Calculate correlations
            correlations = returns_df.corr()[ticker].drop(ticker)
            max_correlation = correlations.abs().max() if not correlations.empty else 0.0
            
            high_correlation = max_correlation > self.risk_limits['max_correlation']
            
            return {
                'high_correlation': high_correlation,
                'max_correlation': float(max_correlation)
            }
            
        except Exception as e:
            logger.error(f"❌ Correlation risk check failed: {e}")
            return {'high_correlation': False, 'max_correlation': 0.0}
    
    def monitor_portfolio_risk(self) -> Dict[str, Any]:
        """Monitor overall portfolio risk"""
        try:
            risk_status = {
                'status': 'healthy',
                'alerts': [],
                'metrics': {}
            }
            
            # Current drawdown
            current_dd = trading_state.current_drawdown
            risk_status['metrics']['current_drawdown'] = current_dd
            
            if current_dd > self.risk_limits['max_total_drawdown']:
                risk_status['status'] = 'critical'
                risk_status['alerts'].append(f"Drawdown {current_dd:.2%} exceeds maximum limit")
            elif current_dd > self.risk_limits['max_daily_drawdown']:
                risk_status['status'] = 'warning'
                risk_status['alerts'].append(f"Drawdown {current_dd:.2%} exceeds daily limit")
            
            # Position concentration
            if trading_state.open_positions:
                position_values = [
                    abs(pos['quantity']) * pos['current_price'] 
                    for pos in trading_state.open_positions.values()
                ]
                max_position_pct = max(position_values) / trading_state.current_equity
                risk_status['metrics']['max_position_concentration'] = max_position_pct
                
                if max_position_pct > self.risk_limits['max_position_size']:
                    risk_status['status'] = 'warning'
                    risk_status['alerts'].append(f"Position concentration {max_position_pct:.2%} exceeds limit")
            
            # Sector concentration
            sector_allocations = trading_state.get_sector_allocation()
            if sector_allocations:
                max_sector_pct = max(sector_allocations.values())
                risk_status['metrics']['max_sector_concentration'] = max_sector_pct
                
                if max_sector_pct > self.risk_limits['max_sector_allocation']:
                    risk_status['status'] = 'warning'
                    risk_status['alerts'].append(f"Sector concentration {max_sector_pct:.2%} exceeds limit")
            
            # Number of positions
            num_positions = len(trading_state.open_positions)
            risk_status['metrics']['num_positions'] = num_positions
            
            if num_positions >= config.MAX_POSITIONS:
                risk_status['alerts'].append(f"At maximum position limit ({num_positions})")
            
            return risk_status
            
        except Exception as e:
            logger.error(f"❌ Portfolio risk monitoring failed: {e}")
            return {'status': 'error', 'alerts': ['Risk monitoring failed'], 'metrics': {}}

risk_monitor = UltraAdvancedRiskMonitor()

# === TRADING LOGIC ===
class UltraAdvancedTradingLogic:
    def __init__(self):
        self.signal_history = defaultdict(list)
        self.last_signal_time = {}
        self.position_entry_signals = {}
        self.confidence_threshold = 0.6
        self.min_signal_strength = 0.55
        
        # Multi-timeframe settings
        self.timeframes = config.TIMEFRAMES if config.MULTI_TIMEFRAME_ENABLED else ['1d']
        self.mtf_weights = {
            '1min': config.MTF_WEIGHT_1MIN,
            '5min': config.MTF_WEIGHT_5MIN,
            '15min': config.MTF_WEIGHT_15MIN,
            '1day': config.MTF_WEIGHT_DAILY
        }

    def generate_trading_signals(self, ticker: str) -> Dict[str, Any]:
        """Generate comprehensive trading signals using voting ensemble"""
        try:
            # Get multi-timeframe data
            signals = {}

            for timeframe in self.timeframes:
                tf_signal = self._generate_timeframe_signal(ticker, timeframe)
                signals[timeframe] = tf_signal

            # Aggregate multi-timeframe signals
            if config.MULTI_TIMEFRAME_ENABLED and len(signals) > 1:
                final_signal = self._aggregate_mtf_signals(signals)
            else:
                final_signal = signals.get('1day', signals.get(list(signals.keys())[0]))

            # === Replace prediction with Voting Ensemble if available ===
            if 'voting_ensemble' in self.models:
                recent_data = data_manager.get_stock_data(ticker, period='5d', interval='1d')
                if recent_data is not None and not recent_data.empty:
                    features = self.prepare_features(recent_data)
                    if not features.empty:
                        latest_features = features.tail(1)
                        if config.FEATURE_SELECTION_ENABLED and self.feature_selector:
                            latest_features = self.feature_selector.transform(latest_features)
                        scaled = self.scaler.transform(latest_features)

                        model = self.models['voting_ensemble']
                        proba = model.predict_proba(scaled)[0]
                        prediction = model.predict(scaled)[0]
                        confidence = max(proba)

                        final_signal['predictions'] = proba.tolist()
                        final_signal['confidence'] = confidence
                        final_signal['action'] = 'buy' if prediction == 1 else 'sell'

            # Add meta-information
            final_signal['ticker'] = ticker
            final_signal['timestamp'] = datetime.now()
            final_signal['timeframes_analyzed'] = list(signals.keys())

            # Store signal history
            self.signal_history[ticker].append(final_signal)
            if len(self.signal_history[ticker]) > 100:
                self.signal_history[ticker] = self.signal_history[ticker][-100:]

            return final_signal

        except Exception as e:
            logger.error(f"❌ Signal generation failed for {ticker}: {e}")
            return self._get_neutral_signal(ticker)

    def _generate_timeframe_signal(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Generate signal for specific timeframe"""
        try:
            period_map = {
                '1min': '5d',
                '5min': '1mo',
                '15min': '3mo',
                '1day': '1y'
            }

            period = period_map.get(timeframe, '1y')
            interval = timeframe if timeframe != '1day' else '1d'

            data = data_manager.get_stock_data(ticker, period=period, interval=interval)
            if data is None or data.empty:
                return self._get_neutral_signal(ticker)

            features = ensemble_model.prepare_features(data)
            if features.empty:
                return self._get_neutral_signal(ticker)

            predictions = ensemble_model.predict(features.tail(1))
            sentiment_data = sentiment_analyzer.get_news_sentiment(ticker)
            options_data = options_analyzer.get_options_flow(ticker)
            news_articles = sentiment_analyzer._fetch_news_articles(ticker, 2)
            catalyst_data = catalyst_filter.analyze_news_catalysts(ticker, news_articles)
            technical_signals = self._calculate_technical_signals(data)

            signal_strength = self._calculate_signal_strength(
                predictions, sentiment_data, options_data,
                catalyst_data, technical_signals
            )

            action = self._determine_action(signal_strength, predictions)

            return {
                'action': action,
                'confidence': signal_strength['confidence'],
                'strength': signal_strength['strength'],
                'predictions': predictions,
                'sentiment': sentiment_data,
                'options_flow': options_data,
                'catalysts': catalyst_data,
                'technical': technical_signals,
                'timeframe': timeframe
            }

        except Exception as e:
            logger.error(f"❌ Timeframe signal generation failed for {ticker} ({timeframe}): {e}")
            return self._get_neutral_signal(ticker)
    
    def _calculate_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical analysis signals"""
        try:
            if data.empty or len(data) < 20:
                return {'rsi': 50, 'macd_signal': 0, 'bb_position': 0.5, 'volume_signal': 0}
            
            # RSI
            rsi = RSIIndicator(data['Close']).rsi().iloc[-1]
            
            # MACD
            macd_indicator = MACD(data['Close'])
            macd_line = macd_indicator.macd().iloc[-1]
            macd_signal = macd_indicator.macd_signal().iloc[-1]
            macd_signal_strength = 1 if macd_line > macd_signal else -1
            
            # Bollinger Bands
            bb = BollingerBands(data['Close'])
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            current_price = data['Close'].iloc[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            # Volume analysis
            volume_sma = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_signal = 1 if current_volume > volume_sma * 1.5 else 0
            
            # Moving average signals
            sma_20 = data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            ma_signal = 1 if sma_20 > sma_50 and current_price > sma_20 else -1
            
            return {
                'rsi': float(rsi) if not np.isnan(rsi) else 50,
                'macd_signal': macd_signal_strength,
                'bb_position': float(bb_position) if not np.isnan(bb_position) else 0.5,
                'volume_signal': volume_signal,
                'ma_signal': ma_signal,
                'price': float(current_price)
            }
            
        except Exception as e:
            logger.error(f"❌ Technical signals calculation failed: {e}")
            return {'rsi': 50, 'macd_signal': 0, 'bb_position': 0.5, 'volume_signal': 0}
    
    def _calculate_signal_strength(self, predictions: Dict, sentiment_data: Dict, 
                                 options_data: Dict, catalyst_data: Dict, 
                                 technical_signals: Dict) -> Dict[str, float]:
        """Calculate overall signal strength"""
        try:
            # Model prediction strength
            buy_prob = predictions.get('buy_prob', 0.33)
            sell_prob = predictions.get('sell_prob', 0.33)
            hold_prob = predictions.get('hold_prob', 0.34)
            
            model_strength = max(buy_prob, sell_prob, hold_prob) - 0.33
            
            # Sentiment strength
            sentiment_score = sentiment_data.get('sentiment', 0.0)
            sentiment_confidence = sentiment_data.get('confidence', 0.0)
            sentiment_strength = abs(sentiment_score) * sentiment_confidence
            
            # Options flow strength
            options_sentiment = options_data.get('sentiment', 0.0)
            options_strength = abs(options_sentiment) * 0.5
            
            # Catalyst strength
            catalyst_multiplier = catalyst_data.get('multiplier', 1.0)
            catalyst_strength = (catalyst_multiplier - 1.0) * 0.5
            
            # Technical strength
            rsi = technical_signals.get('rsi', 50)
            rsi_strength = abs(rsi - 50) / 50  # 0 to 1
            
            bb_position = technical_signals.get('bb_position', 0.5)
            bb_strength = abs(bb_position - 0.5) * 2  # 0 to 1
            
            technical_strength = (rsi_strength + bb_strength) / 2
            
            # Combine strengths
            total_strength = (
                model_strength * 0.4 +
                sentiment_strength * 0.2 +
                options_strength * 0.15 +
                catalyst_strength * 0.1 +
                technical_strength * 0.15
            )
            
            # Calculate confidence
            confidence = min(total_strength * 2, 1.0)  # Scale to 0-1
            
            return {
                'strength': float(total_strength),
                'confidence': float(confidence),
                'components': {
                    'model': model_strength,
                    'sentiment': sentiment_strength,
                    'options': options_strength,
                    'catalyst': catalyst_strength,
                    'technical': technical_strength
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Signal strength calculation failed: {e}")
            return {'strength': 0.0, 'confidence': 0.0, 'components': {}}
    
    def _determine_action(self, signal_strength: Dict, predictions: Dict) -> str:
        """Determine trading action based on signals"""
        try:
            confidence = signal_strength.get('confidence', 0.0)
            strength = signal_strength.get('strength', 0.0)
            
            # Minimum thresholds
            if confidence < self.confidence_threshold or strength < self.min_signal_strength:
                return 'hold'
            
            # Get strongest prediction
            buy_prob = predictions.get('buy_prob', 0.33)
            sell_prob = predictions.get('sell_prob', 0.33)
            hold_prob = predictions.get('hold_prob', 0.34)
            
            max_prob = max(buy_prob, sell_prob, hold_prob)
            
            if max_prob == buy_prob and buy_prob > 0.6:
                return 'buy'
            elif max_prob == sell_prob and sell_prob > 0.6:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"❌ Action determination failed: {e}")
            return 'hold'
    
    def _aggregate_mtf_signals(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate multi-timeframe signals"""
        try:
            # Weight signals by timeframe
            weighted_confidence = 0.0
            weighted_strength = 0.0
            action_votes = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            
            total_weight = 0.0
            
            for timeframe, signal in signals.items():
                weight = self.mtf_weights.get(timeframe, 0.25)
                confidence = signal.get('confidence', 0.0)
                strength = signal.get('strength', 0.0)
                action = signal.get('action', 'hold')
                
                weighted_confidence += confidence * weight
                weighted_strength += strength * weight
                action_votes[action] += weight
                total_weight += weight
            
            # Normalize
            if total_weight > 0:
                weighted_confidence /= total_weight
                weighted_strength /= total_weight
            
            # Determine final action
            final_action = max(action_votes, key=action_votes.get)
            
            # Require confirmation from multiple timeframes
            if config.MTF_CONFIRMATION_THRESHOLD > 0:
                action_strength = action_votes[final_action] / total_weight
                if action_strength < config.MTF_CONFIRMATION_THRESHOLD:
                    final_action = 'hold'
            
            # Combine other data from primary timeframe (daily)
            primary_signal = signals.get('1day', signals.get(list(signals.keys())[0]))
            
            return {
                'action': final_action,
                'confidence': weighted_confidence,
                'strength': weighted_strength,
                'mtf_votes': action_votes,
                'predictions': primary_signal.get('predictions', {}),
                'sentiment': primary_signal.get('sentiment', {}),
                'options_flow': primary_signal.get('options_flow', {}),
                'catalysts': primary_signal.get('catalysts', {}),
                'technical': primary_signal.get('technical', {})
            }
            
        except Exception as e:
            logger.error(f"❌ MTF signal aggregation failed: {e}")
            return self._get_neutral_signal("")
    
    def _get_neutral_signal(self, ticker: str) -> Dict[str, Any]:
        """Get neutral signal when analysis fails"""
        return {
            'action': 'hold',
            'confidence': 0.0,
            'strength': 0.0,
            'predictions': {'buy_prob': 0.33, 'hold_prob': 0.34, 'sell_prob': 0.33},
            'sentiment': {'sentiment': 0.0, 'confidence': 0.0},
            'options_flow': {'sentiment': 0.0},
            'catalysts': {'multiplier': 1.0},
            'technical': {'rsi': 50, 'macd_signal': 0},
            'ticker': ticker,
            'timestamp': datetime.now()
        }
    
def should_exit_position(self, ticker: str, current_price: float) -> Dict[str, Any]:
    """Determine if position should be exited"""
    try:
        if ticker not in trading_state.open_positions:
            return {'should_exit': False, 'reason': 'no_position'}
        
        position = trading_state.open_positions[ticker]
        entry_price = position['entry_price']
        quantity = position['quantity']
        entry_time = position['entry_time']

        # Calculate current P&L
        current_pnl = (current_price - entry_price) * quantity
        pnl_pct = current_pnl / (abs(quantity) * entry_price)

        # Stop loss check
        stop_loss = position.get('stop_loss')
        if stop_loss:
            if (quantity > 0 and current_price <= stop_loss) or \
               (quantity < 0 and current_price >= stop_loss):
                return {'should_exit': True, 'reason': 'stop_loss', 'exit_price': current_price}

        # Take profit check
        take_profit = position.get('take_profit')
        if take_profit:
            if (quantity > 0 and current_price >= take_profit) or \
               (quantity < 0 and current_price <= take_profit):
                return {'should_exit': True, 'reason': 'take_profit', 'exit_price': current_price}

        # Time-based exit (hold for max 5 days)
        hold_duration = datetime.now() - entry_time
        if hold_duration.days >= 5:
            return {'should_exit': True, 'reason': 'time_limit', 'exit_price': current_price}

        # === Voting Ensemble Prediction (if available) ===
        predictions = None
        if 'voting_ensemble' in self.models:
            latest_data = data_manager.get_stock_data(ticker, period='5d', interval='1d')
            if latest_data is not None and not latest_data.empty:
                features = self.prepare_features(latest_data)
                if not features.empty:
                    latest_features = features.tail(1)
                    if config.FEATURE_SELECTION_ENABLED and self.feature_selector:
                        latest_features = self.feature_selector.transform(latest_features)
                    scaled = self.scaler.transform(latest_features)
                    model = self.models['voting_ensemble']
                    proba = model.predict_proba(scaled)[0]
                    prediction = model.predict(scaled)[0]
                    confidence = max(proba)

                    # Exit based on voting ensemble signal
                    if quantity > 0 and prediction == 0 and confidence > 0.7:  # Exit long on strong sell
                        return {'should_exit': True, 'reason': 'voting_sell', 'exit_price': current_price}
                    elif quantity < 0 and prediction == 1 and confidence > 0.7:  # Exit short on strong buy
                        return {'should_exit': True, 'reason': 'voting_buy', 'exit_price': current_price}

        # === Legacy fallback ===
        current_signal = self.generate_trading_signals(ticker)
        if quantity > 0 and current_signal['action'] == 'sell' and current_signal['confidence'] > 0.7:
            return {'should_exit': True, 'reason': 'sell_signal', 'exit_price': current_price}
        if quantity < 0 and current_signal['action'] == 'buy' and current_signal['confidence'] > 0.7:
            return {'should_exit': True, 'reason': 'buy_signal', 'exit_price': current_price}

        # Risk-based exit (large loss)
        if pnl_pct < -0.08:
            return {'should_exit': True, 'reason': 'risk_management', 'exit_price': current_price}

        return {'should_exit': False, 'reason': 'hold'}

    except Exception as e:
        logger.error(f"❌ Exit decision failed for {ticker}: {e}")
        return {'should_exit': False, 'reason': 'error'}
trading_logic = UltraAdvancedTradingLogic()

# === MAIN TRADING LOOP ===
class UltraAdvancedMainLoop:
    def __init__(self):
        self.running = False
        self.loop_count = 0
        self.last_model_retrain = datetime.now()
        self.last_watchlist_update = datetime.now()
        self.last_portfolio_rebalance = datetime.now()
        self.performance_tracker = defaultdict(list)
        
        # Feedback loop
        self.pending_feedback = {}
        self.feedback_queue = deque(maxlen=1000)
        
        # Health monitoring
        self.health_status = {
            'status': 'healthy',
            'last_update': datetime.now(),
            'errors': [],
            'warnings': []
        }
    
    def start(self):
        """Start the main trading loop"""
        try:
            logger.info("🚀 Starting Ultra-Advanced Trading Bot...")
            
            # Initialize components
            self._initialize_components()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.running = True
            
            # Main loop
            while self.running:
                try:
                    self._execute_trading_cycle()
                    self._update_health_status()
                    
                    # Sleep between cycles
                    time.sleep(30)  # 30 seconds between cycles
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Received shutdown signal")
                    break
                except Exception as e:
                    logger.error(f"❌ Trading cycle error: {e}")
                    self.health_status['errors'].append(str(e))
                    time.sleep(60)  # Wait longer on error
            
            self._shutdown()
            
        except Exception as e:
            logger.error(f"❌ Main loop failed: {e}")
            self._shutdown()
    
    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            logger.info("🔧 Initializing components...")
            
            # Check market status
            if not market_status.is_market_open(include_extended=True):
                time_to_open = market_status.time_to_market_open()
                if time_to_open:
                    logger.info(f"⏰ Market closed. Opens in: {time_to_open}")
            
            # Load trading state
            trading_state.load_state()
            
            # Train models if needed
            if not trading_state.models_trained:
                self._train_models()
            
            # Update watchlist
            self._update_watchlist()
            
            logger.info("✅ Components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        try:
            # Start dashboard if enabled
            if config.DASHBOARD_ENABLED:
                dashboard_thread = threading.Thread(target=self._run_dashboard, daemon=True)
                dashboard_thread.start()
                logger.info(f"📊 Dashboard started on port {config.DASHBOARD_PORT}")
            
            # Start health check server
            if config.HEALTH_CHECK_ENABLED:
                health_thread = threading.Thread(target=self._run_health_server, daemon=True)
                health_thread.start()
                logger.info(f"🏥 Health check server started on port {config.HEALTH_CHECK_PORT}")
            
        except Exception as e:
            logger.error(f"❌ Background task startup failed: {e}")
    
    def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            self.loop_count += 1
            cycle_start = time.time()
            
            logger.info(f"🔄 Trading cycle #{self.loop_count} started")
            
            # Check if market is open
            if not market_status.is_market_open(include_extended=True):
                logger.info("💤 Market closed - skipping trading cycle")
                return
            
            # Update broker health
            broker_manager.update_broker_health()
            
            # Process pending feedback
            self._process_feedback_loop()
            
            # Update existing positions
            self._update_positions()
            
            # Check for exit signals
            self._check_exit_signals()
            
            # Generate new signals
            self._generate_new_signals()
            
            # Execute trades
            self._execute_trades()
            
            # Risk monitoring
            self._monitor_risks()
            
            # Periodic tasks
            self._handle_periodic_tasks()
            
            # Update performance metrics
            trading_state.update_performance_metrics()
            
            # Save state
            trading_state.save_state()
            
            cycle_time = time.time() - cycle_start
            logger.info(f"✅ Trading cycle #{self.loop_count} completed in {cycle_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
            self.health_status['errors'].append(f"Cycle {self.loop_count}: {str(e)}")
    
    def _process_feedback_loop(self):
        """Process feedback from completed trades"""
        try:
            if not config.FEEDBACK_LOOP_ENABLED:
                return
            
            # Check for completed trades that need feedback
            for trade in trading_state.trade_history[-10:]:  # Last 10 trades
                trade_id = f"{trade['ticker']}_{trade['entry_time']}"
                
                if trade_id not in trading_state.pending_feedback:
                    # Calculate feedback metrics
                    feedback = self._calculate_trade_feedback(trade)
                    
                    # Store feedback
                    trading_state.feedback_history.append(feedback)
                    trading_state.pending_feedback[trade_id] = feedback
                    
                    # Update model performance tracking
                    self._update_model_performance(trade, feedback)
            
            # Retrain meta-model if enough feedback
            if len(trading_state.feedback_history) >= config.META_MODEL_RETRAIN_FREQUENCY:
                self._retrain_meta_model()
                
        except Exception as e:
            logger.error(f"❌ Feedback loop processing failed: {e}")
    
    def _calculate_trade_feedback(self, trade: Dict) -> Dict:
        """Calculate feedback metrics for a trade"""
        try:
            # Basic metrics
            pnl_pct = trade['pnl'] / (abs(trade['quantity']) * trade['entry_price'])
            hold_duration_hours = trade['hold_duration'].total_seconds() / 3600
            
            # Success metrics
            was_profitable = trade['pnl'] > 0
            exceeded_expectations = pnl_pct > 0.02  # 2% threshold
            
            # Risk-adjusted return
            position_size = abs(trade['quantity']) * trade['entry_price']
            risk_adjusted_return = trade['pnl'] / position_size
            
            # Confidence vs outcome
            confidence = trade.get('confidence', 0.5)
            confidence_accuracy = 1.0 if (was_profitable and confidence > 0.6) or (not was_profitable and confidence < 0.4) else 0.0
            
            return {
                'trade_id': f"{trade['ticker']}_{trade['entry_time']}",
                'ticker': trade['ticker'],
                'pnl_pct': pnl_pct,
                'was_profitable': was_profitable,
                'exceeded_expectations': exceeded_expectations,
                'risk_adjusted_return': risk_adjusted_return,
                'confidence_accuracy': confidence_accuracy,
                'hold_duration_hours': hold_duration_hours,
                'sector': trade['sector'],
                'exit_reason': trade['exit_reason'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Trade feedback calculation failed: {e}")
            return {}
    
    def _update_model_performance(self, trade: Dict, feedback: Dict):
        """Update model performance tracking"""
        try:
            # Track performance by model predictions
            # This would be expanded to track individual model contributions
            
            sector = trade['sector']
            was_profitable = feedback['was_profitable']
            
            if sector not in trading_state.meta_model_performance:
                trading_state.meta_model_performance[sector] = {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0
                }
            
            perf = trading_state.meta_model_performance[sector]
            perf['total_trades'] += 1
            
            if was_profitable:
                perf['profitable_trades'] += 1
            
            perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']
            
            # Update average return (simplified)
            perf['avg_return'] = (perf['avg_return'] * (perf['total_trades'] - 1) + 
                                feedback['pnl_pct']) / perf['total_trades']
            
        except Exception as e:
            logger.error(f"❌ Model performance update failed: {e}")
    
    def _retrain_meta_model(self):
        """Retrain meta-model based on feedback"""
        try:
            logger.info("🔄 Retraining meta-model based on feedback...")
            
            # This would implement online learning updates
            # For now, just log the action
            
            feedback_data = list(trading_state.feedback_history)
            
            if len(feedback_data) >= 50:
                # Calculate performance metrics
                win_rate = sum(1 for f in feedback_data if f.get('was_profitable', False)) / len(feedback_data)
                avg_return = np.mean([f.get('pnl_pct', 0) for f in feedback_data])
                
                logger.info(f"📊 Meta-model feedback: Win Rate: {win_rate:.2%}, Avg Return: {avg_return:.2%}")
                
                # Clear old feedback
                trading_state.feedback_history.clear()
            
        except Exception as e:
            logger.error(f"❌ Meta-model retraining failed: {e}")
    
    def _update_positions(self):
        """Update current position prices and P&L"""
        try:
            if not trading_state.open_positions:
                return
            
            for ticker in list(trading_state.open_positions.keys()):
                try:
                    # Get current price
                    quote = data_manager.get_real_time_quote(ticker)
                    
                    if quote and quote.get('price', 0) > 0:
                        current_price = quote['price']
                        
                        # Update position
                        position = trading_state.open_positions[ticker]
                        position['current_price'] = current_price
                        
                        # Calculate unrealized P&L
                        entry_price = position['entry_price']
                        quantity = position['quantity']
                        unrealized_pnl = (current_price - entry_price) * quantity
                        position['unrealized_pnl'] = unrealized_pnl
                        
                        # Update equity
                        trading_state.current_equity = (
                            config.INITIAL_CAPITAL + 
                            trading_state.total_pnl + 
                            sum(pos['unrealized_pnl'] for pos in trading_state.open_positions.values())
                        )
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update position for {ticker}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Position update failed: {e}")
    
    def _check_exit_signals(self):
        """Check for exit signals on open positions"""
        try:
            positions_to_close = []
            
            for ticker, position in trading_state.open_positions.items():
                try:
                    current_price = position['current_price']
                    
                    # Check exit conditions
                    exit_decision = trading_logic.should_exit_position(ticker, current_price)
                    
                    if exit_decision['should_exit']:
                        positions_to_close.append({
                            'ticker': ticker,
                            'exit_price': exit_decision['exit_price'],
                            'reason': exit_decision['reason']
                        })
                        
                except Exception as e:
                    logger.warning(f"⚠️ Exit check failed for {ticker}: {e}")
                    continue
            
            # Execute exits
            for exit_order in positions_to_close:
                self._execute_exit_order(exit_order)
                
        except Exception as e:
            logger.error(f"❌ Exit signal check failed: {e}")
    
    def _execute_exit_order(self, exit_order: Dict):
        """Execute exit order"""
        try:
            ticker = exit_order['ticker']
            exit_price = exit_order['exit_price']
            reason = exit_order['reason']
            
            if ticker not in trading_state.open_positions:
                return
            
            position = trading_state.open_positions[ticker]
            quantity = position['quantity']
            
            # Execute order through broker
            broker = broker_manager.get_active_broker()
            if not broker:
                logger.error("❌ No active broker for exit order")
                return
            
            # For paper trading, simulate the order
            if config.PAPER_TRADING_MODE:
                success = self._simulate_order(ticker, -quantity, exit_price, "exit")
            else:
                success = self._execute_real_order(ticker, -quantity, exit_price, "exit")
            
            if success:
                # Close position in state
                trading_state.close_position(ticker, exit_price, reason)
                logger.info(f"✅ Exited position: {ticker} @ ${exit_price:.2f} ({reason})")
            else:
                logger.error(f"❌ Failed to exit position: {ticker}")
                
        except Exception as e:
            logger.error(f"❌ Exit order execution failed: {e}")
    
def _generate_new_signals(self):
    """Generate signals for watchlist tickers"""
    try:
        # Check if we can take new positions
        if len(trading_state.open_positions) >= config.MAX_POSITIONS:
            return

        # Get qualified watchlist (tickers not in positions)
        available_tickers = [
            ticker for ticker in trading_state.qualified_watchlist 
            if ticker not in trading_state.open_positions
        ]

        if not available_tickers:
            return

        # Generate signals for available tickers
        signals = []

        for ticker in available_tickers[:20]:  # Limit to 20 tickers per cycle
            try:
                if is_on_cooldown(ticker):
                    logger.info(f"⏳ {ticker} is on cooldown. Skipping signal generation.")
                    continue

                # === Catalyst Check ===
                has_catalyst, catalyst_headline = detect_news_catalyst(ticker)
                if has_catalyst:
                    logger.info(f"🧨 Catalyst found for {ticker}: {catalyst_headline}")
                else:
                    logger.info(f"⏭️ No catalyst for {ticker}, skipping.")
                    continue

                signal = trading_logic.generate_trading_signals(ticker)

                if signal['action'] in ['buy', 'sell'] and signal['confidence'] > 0.6:
                    signals.append(signal)

            except Exception as e:
                logger.warning(f"⚠️ Signal generation failed for {ticker}: {e}")
                continue

        # Sort by confidence and strength
        signals.sort(key=lambda x: x['confidence'] * x['strength'], reverse=True)

        # Store top signals for execution
        trading_state.pending_signals = signals[:config.MAX_PENDING_SIGNALS]

    except Exception as e:
        logger.error(f"❌ Failed to generate new signals: {e}")
    
def _execute_trades(self):
    """Execute pending trades"""
    try:
        if not hasattr(self, 'pending_signals') or not self.pending_signals:
            return
        
        for signal in self.pending_signals:
            try:
                if len(trading_state.open_positions) >= config.MAX_POSITIONS:
                    break
                
                ticker = signal['ticker']
                action = signal['action']
                confidence = signal['confidence']

                # ✅ FIXED: Proper indentation
                if is_on_cooldown(ticker):
                    logger.info(f"⏸️ {ticker} is on cooldown. Skipping trade.")
                    continue

                logger.info(f"🚀 Executing trade for {ticker} — action: {action}, confidence: {confidence:.2f}")

                # Get current price
                quote = data_manager.get_real_time_quote(ticker)
                if not quote or quote.get('price', 0) <= 0:
                    continue

                current_price = quote['price']

                # === Kelly Criterion Position Sizing ===
                if config.POSITION_SIZE_KELLY_ENABLED:
                    kelly_fraction = calculate_kelly_fraction(confidence, cap=config.KELLY_FRACTION_CAP)
                    position_size = trading_state.current_equity * kelly_fraction
                    logger.info(f"📈 Kelly sizing for {ticker}: fraction={kelly_fraction:.4f}, size=${position_size:.2f}")
                else:
                    position_size = self._calculate_position_size(ticker, confidence, current_price)

                if position_size == 0:
                    continue

                # Determine quantity (positive for buy, negative for sell)
                quantity = position_size if action == 'buy' else -position_size
                
                # Risk check
                risk_check = risk_monitor.check_pre_trade_risk(ticker, quantity, current_price)
                if not risk_check['approved']:
                    logger.warning(f"⚠️ Trade rejected for {ticker}: {risk_check['rejections']}")
                    continue
                
                # Execute order
                if config.PAPER_TRADING_MODE:
                    success = self._simulate_order(ticker, quantity, current_price, "entry")
                else:
                    success = self._execute_real_order(ticker, quantity, current_price, "entry")
                
                if success:
                    # Set cooldown timestamp
                    cooldown_cache[ticker] = datetime.utcnow()

                    # Add position to state
                    stop_loss = self._calculate_stop_loss(current_price, action)
                    take_profit = self._calculate_take_profit(current_price, action)
                    trading_state.add_position(
                        ticker, quantity, current_price, confidence,
                        stop_loss, take_profit
                    )
                    
                    logger.info(f"✅ Opened position: {ticker} {quantity} @ ${current_price:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Trade execution failed for {signal.get('ticker', 'unknown')}: {e}")
                continue

        # Clear pending signals
        self.pending_signals = []

    except Exception as e:
        logger.error(f"❌ Trade execution failed: {e}")
            
            # Clear pending signals
            self.pending_signals = []
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
    
    def _calculate_position_size(self, ticker: str, confidence: float, price: float) -> int:
        """Calculate position size based on confidence and risk"""
        try:
            # Base position size
            base_allocation = config.MAX_POSITION_SIZE * 0.5  # 50% of max
            
            # Adjust for confidence
            confidence_multiplier = confidence ** 2  # Square for more aggressive scaling
            
            # Adjust for catalysts
            news_articles = sentiment_analyzer._fetch_news_articles(ticker, 1)
            catalyst_data = catalyst_filter.analyze_news_catalysts(ticker, news_articles)
            catalyst_multiplier = catalyst_data.get('multiplier', 1.0)
            
            # Calculate position value
            position_allocation = base_allocation * confidence_multiplier * catalyst_multiplier
            position_allocation = min(position_allocation, config.MAX_POSITION_SIZE)
            
            position_value = trading_state.current_equity * position_allocation
            
            # Convert to shares
            shares = int(position_value / price)
            
            # Minimum position size
            min_value = trading_state.current_equity * config.MIN_POSITION_SIZE
            min_shares = int(min_value / price)
            
            return max(shares, min_shares)
            
        except Exception as e:
            logger.error(f"❌ Position size calculation failed for {ticker}: {e}")
            return 0
    
    def _calculate_stop_loss(self, entry_price: float, action: str) -> float:
        """Calculate stop loss price"""
        try:
            stop_loss_pct = 0.05  # 5% stop loss
            
            if action == 'buy':
                return entry_price * (1 - stop_loss_pct)
            else:  # sell
                return entry_price * (1 + stop_loss_pct)
                
        except Exception:
            return None
    
    def _calculate_take_profit(self, entry_price: float, action: str) -> float:
        """Calculate take profit price"""
        try:
            take_profit_pct = 0.10  # 10% take profit
            
            if action == 'buy':
                return entry_price * (1 + take_profit_pct)
            else:  # sell
                return entry_price * (1 - take_profit_pct)
                
        except Exception:
            return None
    
    def _simulate_order(self, ticker: str, quantity: int, price: float, order_type: str) -> bool:
        """Simulate order execution for paper trading"""
        try:
            # Add slippage
            estimated_slippage = slippage_model.estimate_slippage(
                ticker, quantity, price, market_status.get_market_session()
            )
            
            # Apply slippage
            if quantity > 0:  # Buy order
                execution_price = price + (estimated_slippage / abs(quantity))
            else:  # Sell order
                execution_price = price - (estimated_slippage / abs(quantity))
            
            # Record slippage
            slippage_model.record_actual_slippage(ticker, price, execution_price, quantity)
            
            # Log trade
            logger.trade(f"SIMULATED {order_type.upper()}: {ticker} {quantity} @ ${execution_price:.2f} (slippage: ${estimated_slippage:.2f})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Order simulation failed: {e}")
            return False
    
    def _execute_real_order(self, ticker: str, quantity: int, price: float, order_type: str) -> bool:
        """Execute real order through broker"""
        try:
            broker = broker_manager.get_active_broker()
            if not broker:
                return False
            
            # Determine order side
            side = 'buy' if quantity > 0 else 'sell'
            abs_quantity = abs(quantity)
            
            # Submit order
            order = api_manager.safe_api_call(
                broker.submit_order,
                symbol=ticker,
                qty=abs_quantity,
                side=side,
                type=config.ORDER_TYPE,
                time_in_force=config.TIME_IN_FORCE
            )
            
            if order:
                logger.trade(f"REAL {order_type.upper()}: {ticker} {quantity} @ market (Order ID: {order.id})")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Real order execution failed: {e}")
            return False
    
    def _monitor_risks(self):
        """Monitor portfolio risks"""
        try:
            risk_status = risk_monitor.monitor_portfolio_risk()
            
            if risk_status['status'] == 'critical':
                logger.error(f"🚨 CRITICAL RISK: {risk_status['alerts']}")
                
                # Halt trading if critical
                trading_state.trading_halted = True
                trading_state.halt_reason = "Critical risk level"
                
            elif risk_status['status'] == 'warning':
                logger.warning(f"⚠️ Risk Warning: {risk_status['alerts']}")
            
            # Check hedge overlay
            if config.HEDGE_OVERLAY_ENABLED:
                vix_level = hedge_manager.get_vix_level()
                should_hedge = hedge_manager.should_activate_hedge(
                    trading_state.current_drawdown, vix_level
                )
                
                if should_hedge and not hedge_manager.hedge_active:
                    self._activate_hedge(vix_level)
                elif not should_hedge and hedge_manager.hedge_active:
                    self._deactivate_hedge()
            
        except Exception as e:
            logger.error(f"❌ Risk monitoring failed: {e}")
    
    def _activate_hedge(self, vix_level: float):
        """Activate hedge overlay"""
        try:
            hedge_size = hedge_manager.calculate_hedge_size(trading_state.current_equity, vix_level)
            
            if hedge_size != 0:
                # Execute hedge order (short SPY)
                if config.PAPER_TRADING_MODE:
                    success = self._simulate_order("SPY", hedge_size, 0, "hedge")
                else:
                    success = self._execute_real_order("SPY", hedge_size, 0, "hedge")
                
                if success:
                    hedge_manager.hedge_active = True
                    hedge_manager.spy_position = hedge_size
                    logger.info(f"✅ Hedge activated: SPY {hedge_size} shares")
            
        except Exception as e:
            logger.error(f"❌ Hedge activation failed: {e}")
    
    def _deactivate_hedge(self):
        """Deactivate hedge overlay"""
        try:
            if hedge_manager.spy_position != 0:
                # Close hedge position
                close_quantity = -hedge_manager.spy_position
                
                if config.PAPER_TRADING_MODE:
                    success = self._simulate_order("SPY", close_quantity, 0, "hedge_close")
                else:
                    success = self._execute_real_order("SPY", close_quantity, 0, "hedge_close")
                
                if success:
                    hedge_manager.hedge_active = False
                    hedge_manager.spy_position = 0
                    logger.info("✅ Hedge deactivated")
            
        except Exception as e:
            logger.error(f"❌ Hedge deactivation failed: {e}")
    
    def _handle_periodic_tasks(self):
        """Handle periodic maintenance tasks"""
        try:
            current_time = datetime.now()
            
            # Model retraining
            if (current_time - self.last_model_retrain).total_seconds() > config.MODEL_RETRAIN_FREQUENCY * 3600:
                self._retrain_models()
                self.last_model_retrain = current_time
            
            # Watchlist update
            if (current_time - self.last_watchlist_update).total_seconds() > config.WATCHLIST_REFRESH_HOURS * 3600:
                self._update_watchlist()
                self.last_watchlist_update = current_time
            
            # Portfolio rebalancing
            if config.SECTOR_ROTATION_ENABLED and \
               (current_time - self.last_portfolio_rebalance).total_seconds() > config.PORTFOLIO_OPTIMIZATION_FREQUENCY * 3600:
                self._rebalance_portfolio()
                self.last_portfolio_rebalance = current_time
            
        except Exception as e:
            logger.error(f"❌ Periodic tasks failed: {e}")
    
    def _train_models(self):
        """Train ensemble models"""
        try:
            logger.info("🤖 Training ensemble models...")
            
            # Get training data
            training_data = data_manager.get_multiple_stock_data(
                trading_state.current_watchlist[:50],  # Limit for training
                period="1y",
                interval="1d"
            )
            
            # Train models
            success = ensemble_model.train_models(training_data)
            
            if success:
                trading_state.models_trained = True
                logger.info("✅ Model training completed")
            else:
                logger.error("❌ Model training failed")
                
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
    
    def _retrain_models(self):
        """Retrain models with recent data"""
        try:
            logger.info("🔄 Retraining models with recent data...")
            self._train_models()
            
        except Exception as e:
            logger.error(f"❌ Model retraining failed: {e}")
    
    def _update_watchlist(self):
        """Update and qualify watchlist"""
        try:
            logger.info("📋 Updating watchlist...")
            
            # Filter watchlist based on criteria
            qualified_tickers = []
            
            for ticker in trading_state.current_watchlist:
                try:
                    # Get basic info
                    stock_info = yf.Ticker(ticker).info
                    
                    if not stock_info:
                        continue
                    
                    # Apply filters
                    market_cap = stock_info.get('marketCap', 0)
                    price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))
                    volume = stock_info.get('averageVolume', 0)
                    
                    if (market_cap >= config.MIN_MARKET_CAP and
                        config.MIN_PRICE <= price <= config.MAX_PRICE and
                        volume >= config.MIN_DAILY_VOLUME):
                        qualified_tickers.append(ticker)
                        
                except Exception:
                    continue
            
            trading_state.qualified_watchlist = qualified_tickers
            logger.info(f"✅ Qualified watchlist updated: {len(qualified_tickers)} tickers")
            
        except Exception as e:
            logger.error(f"❌ Watchlist update failed: {e}")
    
    def _rebalance_portfolio(self):
        """Rebalance portfolio based on sector rotation"""
        try:
            if not config.SECTOR_ROTATION_ENABLED or not trading_state.open_positions:
                return
            
            logger.info("⚖️ Rebalancing portfolio...")
            
            # Get current sector allocations
            current_allocations = trading_state.get_sector_allocation()
            
            # Determine if rebalancing is needed
            needs_rebalancing = False
            for sector, allocation in current_allocations.items():
                if allocation > config.MAX_SECTOR_ALLOCATION or allocation < config.MIN_SECTOR_ALLOCATION:
                    needs_rebalancing = True
                    break
            
            if needs_rebalancing:
                logger.info("🔄 Portfolio rebalancing needed")
                # Implement rebalancing logic here
                # This would involve closing overweight positions and opening underweight ones
            else:
                logger.info("✅ Portfolio allocation within targets")
                
        except Exception as e:
            logger.error(f"❌ Portfolio rebalancing failed: {e}")
    
    def _update_health_status(self):
        """Update system health status"""
        try:
            self.health_status['last_update'] = datetime.now()
            
            # Check component health
            components_healthy = True
            
            # Check broker health
            broker = broker_manager.get_active_broker()
            if not broker:
                components_healthy = False
                self.health_status['warnings'].append("No active broker")
            
            # Check model status
            if not trading_state.models_trained:
                components_healthy = False
                self.health_status['warnings'].append("Models not trained")
            
            # Check recent errors
            if len(self.health_status['errors']) > 10:
                components_healthy = False
            
            # Update status
            if components_healthy and not self.health_status['errors']:
                self.health_status['status'] = 'healthy'
            elif components_healthy:
                self.health_status['status'] = 'warning'
            else:
                self.health_status['status'] = 'unhealthy'
            
            # Clean old errors/warnings
            if len(self.health_status['errors']) > 50:
                self.health_status['errors'] = self.health_status['errors'][-25:]
            if len(self.health_status['warnings']) > 50:
                self.health_status['warnings'] = self.health_status['warnings'][-25:]
                
        except Exception as e:
            logger.error(f"❌ Health status update failed: {e}")
    
    def _run_dashboard(self):
        """Run Streamlit dashboard"""
        try:
            import subprocess
            import sys
            
            # Run dashboard in subprocess
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", "dashboard.py",
                "--server.port", str(config.DASHBOARD_PORT),
                "--server.headless", "true"
            ])
            
        except Exception as e:
            logger.error(f"❌ Dashboard startup failed: {e}")
    
    def _run_health_server(self):
        """Run health check server"""
        try:
            from flask import Flask, jsonify
            
            app = Flask(__name__)
            
            @app.route('/health')
            def health_check():
                return jsonify(self.health_status)
            
            @app.route('/metrics')
            def metrics():
                return jsonify({
                    'equity': trading_state.current_equity,
                    'positions': len(trading_state.open_positions),
                    'total_trades': trading_state.total_trades,
                    'win_rate': trading_state.win_rate,
                    'drawdown': trading_state.current_drawdown
                })
            
            app.run(host='0.0.0.0', port=config.HEALTH_CHECK_PORT, debug=False)
            
        except Exception as e:
            logger.error(f"❌ Health server startup failed: {e}")
    
    def _shutdown(self):
        """Graceful shutdown"""
        try:
            logger.info("🛑 Shutting down trading bot...")
            
            self.running = False
            
            # Save final state
            trading_state.save_state()
            
            # Close all positions if configured
            if config.PAPER_TRADING_MODE:
                logger.info("💼 Closing all positions for shutdown...")
                for ticker in list(trading_state.open_positions.keys()):
                    position = trading_state.open_positions[ticker]
                    current_price = position['current_price']
                    trading_state.close_position(ticker, current_price, "shutdown")
            
            logger.info("✅ Shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Shutdown failed: {e}")

# === DASHBOARD ===
def create_dashboard():
    """Create Streamlit dashboard"""
    try:
        st.set_page_config(
            page_title="Ultra Trading Bot Dashboard",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 Ultra-Advanced AI Trading Bot Dashboard")
        
        # Sidebar
        st.sidebar.header("📊 System Status")
        
        # Load current state
        try:
            with open("data/trading_state.json", "r") as f:
                state_data = json.load(f)
        except:
            state_data = {}
        
        # Key metrics
        equity = state_data.get('current_equity', config.INITIAL_CAPITAL)
        total_pnl = state_data.get('total_pnl', 0)
        positions = len(state_data.get('open_positions', {}))
        total_trades = state_data.get('total_trades', 0)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Current Equity", f"${equity:,.2f}", f"${total_pnl:,.2f}")
        
        with col2:
            st.metric("📈 Open Positions", positions)
        
        with col3:
            st.metric("🔄 Total Trades", total_trades)
        
        with col4:
            win_rate = state_data.get('winning_trades', 0) / max(total_trades, 1)
            st.metric("🎯 Win Rate", f"{win_rate:.1%}")
        
        # Charts
        st.header("📊 Performance Charts")
        
        # Create sample performance chart
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        performance = np.cumsum(np.random.randn(100) * 0.01) + 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=performance, mode='lines', name='Portfolio Value'))
        fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Value')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Positions table
        if state_data.get('open_positions'):
            st.header("📋 Current Positions")
            positions_df = pd.DataFrame.from_dict(state_data['open_positions'], orient='index')
            st.dataframe(positions_df)
        
        # Recent trades
        if state_data.get('trade_history'):
            st.header("📜 Recent Trades")
            trades_df = pd.DataFrame(state_data['trade_history'][-10:])
            st.dataframe(trades_df)
        
    except Exception as e:
        st.error(f"Dashboard error: {e}")

# === Initialize Brokers ===
broker_manager = BrokerManager()
broker_manager.initialize()

# === MAIN EXECUTION ===
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Ultra-Advanced AI Trading Bot")
    parser.add_argument("--mode", choices=["trade", "backtest", "train", "dashboard"], 
                       default="trade", help="Execution mode")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    
    args = parser.parse_args()
    
    # Optional: only run when in backtest mode
    if args.mode == "backtest":
        run_backtest(["AAPL", "MSFT", "NVDA", "TSLA"], days=60)
        return  # 🔁 Exit after backtest
    
    # Override paper trading if specified
    if args.paper:
        config.PAPER_TRADING_MODE = True
    
    try:
        if args.mode == "trade":
            # Start main trading loop
            main_loop = UltraAdvancedMainLoop()
            
            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info("🛑 Received shutdown signal")
                main_loop.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start trading
            main_loop.start()
            
        elif args.mode == "backtest":
            logger.info("📊 Starting backtesting mode...")
            # Implement backtesting logic here
            
        elif args.mode == "train":
            logger.info("🤖 Starting training mode...")
            # Train models only
            training_data = data_manager.get_multiple_stock_data(
                trading_state.current_watchlist[:20],
                period="1y",
                interval="1d"
            )
            ensemble_model.train_models(training_data)
            
        elif args.mode == "dashboard":
            logger.info("📊 Starting dashboard mode...")
            create_dashboard()
            
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        traceback.print_exc()
    finally:
        logger.info("👋 Ultra-Advanced Trading Bot shutdown complete")

def run_backtest(tickers, days=30):
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    print("🚀 Running Backtest...")
    results = []
    equity = config.INITIAL_CAPITAL
    equity_curve = []

    for ticker in tickers:
        df = get_data(ticker, days=days)
        if df is None or len(df) < 50:
            print(f"⚠️ Skipping {ticker} — insufficient data")
            continue

        position = None
        entry_price = 0
        entry_index = None

        for i in range(50, len(df)):
            sample = df.iloc[:i]
            latest = df.iloc[i]

            pred_short, pred_medium = predict(ticker, sample)
            if pred_short is None:
                continue

            signal = "HOLD"
            if pred_short > 0.6 and pred_medium > 0.55:
                signal = "BUY"
            elif pred_short < 0.45 and pred_medium < 0.45:
                signal = "SELL"

            if position is None and signal == "BUY":
                position = latest["Close"]
                entry_price = latest["Close"]
                entry_index = i

            elif position is not None and signal == "SELL":
                pnl = latest["Close"] - entry_price
                results.append({
                    "ticker": ticker,
                    "entry": entry_price,
                    "exit": latest["Close"],
                    "pnl": pnl,
                    "entry_index": entry_index,
                    "exit_index": i,
                    "duration": i - entry_index,
                    "confidence_short": pred_short,
                    "confidence_medium": pred_medium
                })
                equity += pnl
                position = None
                entry_index = None

            equity_curve.append(equity)

    # === Summary Statistics ===
    total_trades = len(results)
    profitable = sum(1 for r in results if r["pnl"] > 0)
    win_rate = profitable / total_trades if total_trades > 0 else 0
    returns = pd.Series([r["pnl"] / r["entry"] for r in results if r["entry"] > 0])
    sharpe_ratio = returns.mean() / returns.std() * (252**0.5) if returns.std() > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    durations = [r["duration"] for r in results if "duration" in r]
    avg_duration = np.mean(durations) if durations else 0

    print(f"\n📊 Backtest Summary:")
    print(f"Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Average Trade Duration: {avg_duration:.1f} bars")
    print(f"Final Equity: ${equity:,.2f}")
    print(f"Return: {(equity - config.INITIAL_CAPITAL)/config.INITIAL_CAPITAL:.2%}")

    # === Plot Equity Curve ===
    try:
        plt.plot(equity_curve)
        plt.title("Backtest Equity Curve")
        plt.xlabel("Trade")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"⚠️ Plot failed: {e}")

    # === Save to CSV ===
    try:
        df_results = pd.DataFrame(results)
        df_results.to_csv("backtest_results.csv", index=False)
        print(f"📁 Saved trade log to backtest_results.csv")
    except Exception as e:
        print(f"❌ Failed to save backtest results: {e}")

if __name__ == "__main__":
    import sys

    mode = "main"
    if "--mode" in sys.argv:
        mode_index = sys.argv.index("--mode") + 1
        if mode_index < len(sys.argv):
            mode = sys.argv[mode_index]

    if mode == "backtest":
        run_backtest(["AAPL", "MSFT", "NVDA", "TSLA"], days=60)
    else:
        main()
