# market_regime.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from api_manager import safe_api_call, api
from redis_cache import redis_cache, redis_key
from config import config
import logger

class MarketRegimeDetector:
    def __init__(self, symbol="SPY", short_window=50, long_window=200, ma_type="sma"):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.lower()
        self.last_regime = "neutral"
        self.last_timestamp = None

    def detect_market_regime(self, force_refresh: bool = False) -> str:
        """Detects market regime using MA crossover on SPY."""
        cache_key = redis_key("MARKET_REGIME", self.symbol)
        cached = redis_cache.get(cache_key)
        if cached and not force_refresh:
            logger.deduped_log("debug", f"✅ Using cached regime: {cached}")
            return cached

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
                return self.last_regime

            df = pd.DataFrame([{
                "timestamp": bar.t,
                "close": bar.c
            } for bar in bars])

            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            if self.ma_type == "ema":
                df["SMA_short"] = df["close"].ewm(span=self.short_window).mean()
                df["SMA_long"] = df["close"].ewm(span=self.long_window).mean()
            else:
                df["SMA_short"] = df["close"].rolling(window=self.short_window).mean()
                df["SMA_long"] = df["close"].rolling(window=self.long_window).mean()

            df.dropna(subset=["SMA_short", "SMA_long"], inplace=True)

            if df.empty:
                logger.logger.warning("⚠️ SMA columns contain insufficient data.")
                return self.last_regime

            latest = df.iloc[-1]
            regime = "neutral"

            if latest["SMA_short"] > latest["SMA_long"]:
                regime = "bullish"
            elif latest["SMA_short"] < latest["SMA_long"]:
                regime = "bearish"

            regime = regime.lower()
            redis_cache.set(cache_key, regime, ttl_seconds=3600)
            self.last_regime = regime
            self.last_timestamp = end

            logger.deduped_log("info", f"📊 Market Regime Detected: {regime.upper()}")
            return regime

        except Exception as e:
            logger.logger.error(f"❌ Failed to detect market regime: {e}")
            return self.last_regime

    def get_last_regime(self):
        return self.last_regime

    def get_last_detection_time(self):
        return self.last_timestamp

    def get_regime_label(self) -> str:
        return f"📊 Market Regime: {self.last_regime.upper()} ({self.symbol})"

# === Singleton Instance ===
regime_detector = MarketRegimeDetector()
