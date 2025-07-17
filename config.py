# config.py
import os
import redis
import json
from urllib.parse import urlparse

# Unique user identifier (used for Redis key namespacing)
USER_ID = os.getenv("USER_SESSION_ID", "default_user")

class Config:
    def __init__(self):
        self.USER_ID = USER_ID
        self.REDIS_URL = self._get("REDIS_URL", required=False)
        self.PAPER_TRADING_MODE = True  # default

        # Try Redis override
        self._load_trading_mode_from_redis()

        # Then load correct key set
        self._load_keys()

        # === Additional config values ===
        self.EOD_LIQUIDATION_ENABLED = self._get("EOD_LIQUIDATION_ENABLED", default="True").lower() == "true"
        self.MULTI_INSTANCE_MONITORING = self._get("MULTI_INSTANCE_MONITORING", default="False").lower() == "true"
        self.DEBUG = self._get("DEBUG", default="False").lower() == "true"
        self.Q_LEARNING_EPSILON = float(self._get("Q_LEARNING_EPSILON", default="0.1"))
        self.Q_LEARNING_GAMMA = float(self._get("Q_LEARNING_GAMMA", default="0.95"))
        self.Q_LEARNING_LR = float(self._get("Q_LEARNING_LR", default="0.001"))
        self.Q_LEARNING_EPSILON_DECAY = float(self._get("Q_LEARNING_EPSILON_DECAY", default="0.995"))
        self.Q_LEARNING_MIN_EPSILON = float(self._get("Q_LEARNING_MIN_EPSILON", default="0.01"))
        self.REGIME_DETECTION_WINDOW = int(self._get("REGIME_DETECTION_WINDOW", default="20"))
        self.BULL_MARKET_THRESHOLD = float(self._get("BULL_MARKET_THRESHOLD", default="0.02"))
        self.META_MODEL_MIN_ACCURACY = float(self._get("META_MODEL_MIN_ACCURACY", default="0.60"))
        self.META_MODEL_MIN_TRADES = int(self._get("META_MODEL_MIN_TRADES", default="100"))
        self.DYNAMIC_WATCHLIST_REFRESH_HOURS = int(self._get("DYNAMIC_WATCHLIST_REFRESH_HOURS", default="6"))
        self.MAX_PER_SECTOR_WATCHLIST = int(self._get("MAX_PER_SECTOR_WATCHLIST", default="3"))
        self.WATCHLIST_LIMIT = int(self._get("WATCHLIST_LIMIT", default="20"))
        self.BACKTEST_MODE = self._get("BACKTEST_MODE", default="False").lower() == "true"
        self.MARKET_OPEN_WAIT_MINUTES = int(self._get("MARKET_OPEN_WAIT_MINUTES", default="15"))
        self.EOD_LIQUIDATION_TIME = self._get("EOD_LIQUIDATION_TIME", default="15:45")
        self.FINBERT_MODEL_NAME = self._get("FINBERT_MODEL_NAME", default="ProsusAI/finbert")
        self.MAX_POSITIONS = int(self._get("MAX_POSITIONS", default="10"))
        self.TRADE_COOLDOWN_MINUTES = int(self._get("TRADE_COOLDOWN_MINUTES", default="30"))
        self.ENSEMBLE_CONFIDENCE_THRESHOLD = float(self._get("ENSEMBLE_CONFIDENCE_THRESHOLD", default="0.65"))
        self.POSITION_SIZING_MODE = self._get("POSITION_SIZING_MODE", default="Kelly Criterion")
        self.FIXED_TRADE_AMOUNT = int(self._get("FIXED_TRADE_AMOUNT", default="1000"))
        self.MAX_PORTFOLIO_RISK = float(self._get("MAX_PORTFOLIO_RISK", default="0.25"))
        self.KELLY_MULTIPLIER = float(self._get("KELLY_MULTIPLIER", default="0.5"))
        self.RSI_MIN = int(self._get("RSI_MIN", default="30"))
        self.RSI_MAX = int(self._get("RSI_MAX", default="70"))

    def _load_trading_mode_from_redis(self):
        """Attempt to load PAPER_TRADING_MODE from Redis if available"""
        if not self.REDIS_URL:
            self._fallback_trading_mode()
            return

        try:
            parsed = urlparse(self.REDIS_URL)
            redis_client = redis.Redis(
                host=parsed.hostname,
                port=parsed.port,
                password=parsed.password,
                ssl=parsed.scheme == "rediss",
                decode_responses=True
            )
            key = f"{self.USER_ID}:mode"
            raw = redis_client.get(key)
            if raw:
                mode_data = json.loads(raw)
                self.PAPER_TRADING_MODE = mode_data.get("paper", True)
                return
        except Exception as e:
            print(f"⚠️ Redis mode load failed: {e}")
        self._fallback_trading_mode()

    def _fallback_trading_mode(self):
        """Fallback if Redis fails"""
        mode_env = os.getenv("TRADING_MODE", "paper").lower()
        if mode_env == "live":
            self.PAPER_TRADING_MODE = False
        else:
            self.PAPER_TRADING_MODE = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true"

    def _load_keys(self):
        """Load correct key set depending on mode"""
        if self.PAPER_TRADING_MODE:
            self.ALPACA_API_KEY = os.getenv("ALPACA_PAPER_API_KEY")
            self.ALPACA_SECRET_KEY = os.getenv("ALPACA_PAPER_SECRET_KEY")
            self.ALPACA_BASE_URL = os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
        else:
            self.ALPACA_API_KEY = os.getenv("ALPACA_LIVE_API_KEY")
            self.ALPACA_SECRET_KEY = os.getenv("ALPACA_LIVE_SECRET_KEY")
            self.ALPACA_BASE_URL = os.getenv("ALPACA_LIVE_BASE_URL", "https://api.alpaca.markets")

    def _get(self, var, default=None, required=True):
        value = os.getenv(var, default)
        if required and value is None:
            raise EnvironmentError(f"Missing required environment variable: {var}")
        return value

class TradingConfig:
    PAPER_TRADING_MODE = True
    MAX_POSITIONS = 10
    TRADE_COOLDOWN_MINUTES = 30
    ENSEMBLE_CONFIDENCE_THRESHOLD = 0.65

config = Config()
