# market_regime.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from api_manager import safe_api_call, api
from redis_cache import redis_cache, redis_key
from config import config
import logger

class MarketRegimeDetector:
    def __init__(self, symbol="SPY", short_window=50, long_window=200):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window

    def detect_market_regime(self) -> str:
        """Detects market regime using MA crossover on SPY."""
        cache_key = redis_key("MARKET_REGIME", self.symbol)
        cached = redis_cache.get(cache_key)
        if cached:
            return cached.decode()

        try:
            end = datetime.utcnow()
            start = end - timedelta(days=300)

            bars = safe_api_call(lambda: api.get_bars(
                symbol=self.symbol,
                timeframe="1Day",
                start=start.isoformat() + "Z",
                end=end.isoformat() + "Z",
                adjustment='raw'
            ))

            if not bars or len(bars) < self.long_window:
                logger.logger.warning("⚠️ Not enough data to detect market regime.")
                return "neutral"

            df = pd.DataFrame([{
                "timestamp": bar.t,
                "close": bar.c
            } for bar in bars])

            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            df["SMA_short"] = df["close"].rolling(window=self.short_window).mean()
            df["SMA_long"] = df["close"].rolling(window=self.long_window).mean()

            latest = df.iloc[-1]
            regime = "neutral"

            if latest["SMA_short"] > latest["SMA_long"]:
                regime = "bullish"
            elif latest["SMA_short"] < latest["SMA_long"]:
                regime = "bearish"

            redis_cache.set(cache_key, regime, ttl_seconds=3600)
            return regime

        except Exception as e:
            logger.logger.error(f"❌ Failed to detect market regime: {e}")
            return "neutral"


# === Singleton Instance ===
regime_detector = MarketRegimeDetector()
