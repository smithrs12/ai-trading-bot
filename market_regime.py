import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from api_manager import safe_api_call, api
import redis
import os
from urllib.parse import urlparse
from config import config
from logger import logger
from trading_state import trading_state
import json
import hashlib

# Redis setup
def get_redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        parsed_url = urlparse(redis_url)
        client = redis.Redis(
            host=parsed_url.hostname,
            port=parsed_url.port,
            password=parsed_url.password,
            ssl=parsed_url.scheme == 'rediss',
            decode_responses=True
        )
        client.ping()
        return client
    except Exception:
        return None

client = get_redis_client()

# === RedisCache ===
class RedisCache:
    def __init__(self, redis_client):
        self.client = redis_client
        self.enabled = redis_client is not None

    def make_key(self, prefix, payload, user_id=None):
        raw = json.dumps(payload, sort_keys=True)
        uid = user_id or getattr(config, "USER_ID", "global")
        return f"{uid}:{prefix}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, key):
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            if value:
                print(f"📥 Redis HIT: {key}")
            else:
                print(f"📭 Redis MISS: {key}")
            return json.loads(value) if value else None
        except Exception as e:
            print(f"❌ Redis get failed: {e}")
            return None

    def set(self, key, value, ttl_seconds=3600):
        if not self.enabled:
            return
        try:
            self.client.setex(key, ttl_seconds, json.dumps(value))
            print(f"📤 Redis SET: {key} (TTL: {ttl_seconds}s)")
        except Exception as e:
            print(f"❌ Redis set failed: {e}")

redis_cache = RedisCache(client)

class MarketRegimeDetector:
    def __init__(self, symbol="SPY", short_window=50, long_window=200, ma_type="sma"):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.lower()
        self.last_regime = "neutral"
        self.last_timestamp = None

    def detect_market_regime(self, user_id=None, force_refresh: bool = False):
        """Detects market regime using MA crossover on SPY."""
        cache_key = f"MARKET_REGIME:{self.symbol}"
        cached = redis_cache.get(cache_key)
        if cached and not force_refresh:
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

            # Handle explicit SIP subscription failure
            if bars == "SIP_SUBSCRIPTION_ERROR":
                logger.warning("⚠️ SIP subscription error detected. Regime detection skipped.")
                trading_state.trading_disabled = True
                trading_state.disabled_reason = "SIP subscription required"
                return self.last_regime

            # Handle None or too-short response
            if not bars or isinstance(bars, str) or len(bars) < self.long_window:
                logger.warning("⚠️ Not enough data or invalid format for regime detection.")
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
                logger.warning("⚠️ SMA values missing; skipping regime detection.")
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
            if "subscription" in str(e).lower():
                logger.warning("⚠️ Exception indicates SIP subscription missing. Disabling trading.")
                trading_state.trading_disabled = True
                trading_state.disabled_reason = "SIP subscription required"
            else:
                logger.error(f"❌ Exception in regime detection: {e}")
            return self.last_regime

    def get_last_regime(self):
        return self.last_regime

    def get_last_detection_time(self):
        return self.last_timestamp

    def get_regime_label(self) -> str:
        return f"📊 Market Regime: {self.last_regime.upper()} ({self.symbol})"

# === Singleton Instance ===
regime_detector = MarketRegimeDetector()
