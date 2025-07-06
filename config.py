"""
Configuration settings for the AI Trading Bot
"""
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    
    # Account Settings
    MAX_PORTFOLIO_RISK: float = 0.02  # 2% max risk per trade
    MAX_DAILY_TRADES: int = 10
    MAX_POSITIONS: int = 5
    MIN_ACCOUNT_VALUE: float = 1000.0

    AUTO_LIQUIDATE_ENABLED: bool = True
    LIQUIDATION_HOUR: int = 15  # 3 PM
    LIQUIDATION_MINUTE: int = 55
    TIMEZONE: str = 'US/Eastern'
    
    # Position Sizing
    KELLY_MULTIPLIER: float = 0.25  # Conservative Kelly fraction
    MIN_POSITION_SIZE: float = 100.0  # Minimum $100 per position
    MAX_POSITION_SIZE: float = 2000.0  # Maximum $2000 per position
    
    # Risk Management
    STOP_LOSS_MULTIPLIER: float = 2.0  # 2x ATR for stop loss
    PROFIT_TARGET_MULTIPLIER: float = 3.0  # 3x ATR for profit target
    TRAILING_STOP_THRESHOLD: float = 0.03  # 3% profit before trailing
    PROFIT_DECAY_THRESHOLD: float = 0.5  # 50% profit decay triggers exit
    
    # Model Thresholds
    MIN_PREDICTION_CONFIDENCE: float = 0.6
    META_MODEL_THRESHOLD: float = 0.5
    SENTIMENT_OVERRIDE_THRESHOLD: float = -0.5
    
    # Market Filters
    MIN_VOLUME_MULTIPLIER: float = 1.2  # 1.2x average volume
    MIN_MOMENTUM_THRESHOLD: float = 0.005  # 0.5% momentum
    MAX_VOLATILITY_THRESHOLD: float = 0.05  # 5% max volatility
    
    # Sector Limits
    MAX_SECTOR_ALLOCATION: float = 0.30  # 30% max per sector
    MAX_POSITIONS_PER_SECTOR: int = 3
    
    # Timing
    MARKET_OPEN_DELAY: int = 30  # Wait 30 minutes after market open
    MARKET_CLOSE_BUFFER: int = 15  # Stop trading 15 minutes before close
    EOD_LIQUIDATION_TIME: str = "15:45"  # Eastern Time
    
    # Model Retraining
    SHORT_RETRAIN_INTERVAL: int = 5  # Minutes
    MEDIUM_RETRAIN_INTERVAL: int = 60  # Minutes
    META_RETRAIN_INTERVAL: int = 1440  # Daily (minutes)
    MIN_TRAINING_SAMPLES: int = 100

@dataclass
class APIConfig:
    """API configuration settings"""
    
    # Alpaca
    ALPACA_API_KEY: str = os.getenv("APCA_API_KEY_ID", "")
    ALPACA_SECRET_KEY: str = os.getenv("APCA_API_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Discord
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    
    # News API
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    
    # Google Sheets
    GSHEET_ID: str = os.getenv("GSHEET_ID", "")
    GSPREAD_JSON_PATH: str = os.getenv("GSPREAD_JSON_PATH", "")

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    
    # Feature Engineering
    TECHNICAL_INDICATORS: List[str] = None
    LOOKBACK_PERIODS: List[int] = None
    
    # Model Parameters
    XGBOOST_PARAMS: Dict[str, Any] = None
    RANDOM_FOREST_PARAMS: Dict[str, Any] = None
    LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = None
    
    # Q-Learning
    Q_LEARNING_PARAMS: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.TECHNICAL_INDICATORS is None:
            self.TECHNICAL_INDICATORS = [
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12',
                'atr', 'volume_sma', 'price_change', 'volatility'
            ]
        
        if self.LOOKBACK_PERIODS is None:
            self.LOOKBACK_PERIODS = [5, 10, 20, 50]
        
        if self.XGBOOST_PARAMS is None:
            self.XGBOOST_PARAMS = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.RANDOM_FOREST_PARAMS is None:
            self.RANDOM_FOREST_PARAMS = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        
        if self.LOGISTIC_REGRESSION_PARAMS is None:
            self.LOGISTIC_REGRESSION_PARAMS = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        
        if self.Q_LEARNING_PARAMS is None:
            self.Q_LEARNING_PARAMS = {
                'learning_rate': 0.001,
                'epsilon': 0.1,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'memory_size': 10000,
                'batch_size': 32,
                'target_update_freq': 100
            }

@dataclass
class WatchlistConfig:
    """Watchlist configuration"""
    
    # Default watchlist
    DEFAULT_SYMBOLS: List[str] = None
    
    # Dynamic watchlist
    ENABLE_DYNAMIC_WATCHLIST: bool = True
    WATCHLIST_SIZE: int = 20
    WATCHLIST_UPDATE_INTERVAL: int = 60  # Minutes
    
    # Screening criteria
    MIN_MARKET_CAP: float = 1e9  # $1B minimum market cap
    MIN_AVERAGE_VOLUME: int = 1000000  # 1M shares daily volume
    MAX_PRICE: float = 500.0  # Maximum stock price
    MIN_PRICE: float = 5.0  # Minimum stock price
    
    def __post_init__(self):
        if self.DEFAULT_SYMBOLS is None:
            self.DEFAULT_SYMBOLS = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
                'SHOP', 'SQ', 'ROKU', 'ZM', 'DOCU', 'SNOW', 'PLTR', 'COIN'
            ]

# Global configuration instances
TRADING_CONFIG = TradingConfig()
API_CONFIG = APIConfig()
MODEL_CONFIG = ModelConfig()
WATCHLIST_CONFIG = WatchlistConfig()

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required API keys
    if not API_CONFIG.ALPACA_API_KEY:
        errors.append("APCA_API_KEY_ID environment variable is required")
    
    if not API_CONFIG.ALPACA_SECRET_KEY:
        errors.append("APCA_API_SECRET_KEY environment variable is required")
    
    # Check trading parameters
    if TRADING_CONFIG.MAX_PORTFOLIO_RISK <= 0 or TRADING_CONFIG.MAX_PORTFOLIO_RISK > 0.1:
        errors.append("MAX_PORTFOLIO_RISK must be between 0 and 0.1 (10%)")
    
    if TRADING_CONFIG.MIN_POSITION_SIZE >= TRADING_CONFIG.MAX_POSITION_SIZE:
        errors.append("MIN_POSITION_SIZE must be less than MAX_POSITION_SIZE")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return True

# Environment-specific settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"
LOG_LEVEL = "DEBUG" if DEBUG else "INFO"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PERFORMANCE_DIR = os.path.join(BASE_DIR, "performance")

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR, PERFORMANCE_DIR]:
    os.makedirs(directory, exist_ok=True)
    for subdir in ['short', 'medium', 'meta', 'q_learning']:
        os.makedirs(os.path.join(MODELS_DIR, subdir), exist_ok=True)

# config.py
INITIAL_CAPITAL = 100000  # or whatever starting balance you prefer
