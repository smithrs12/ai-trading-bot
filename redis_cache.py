# redis_cache.py

import os
import redis
import json
import hashlib
import pandas as pd
from urllib.parse import urlparse
from typing import Optional
from config import config
from api_manager import api
from logger import logger

# === Redis URL ===
REDIS_URL = os.getenv("REDIS_URL")

# === Redis Client ===
def get_redis_client():
    REDIS_URL = os.getenv("REDIS_URL")
    if not REDIS_URL:
        print("⚠️ REDIS_URL not set.")
        return None
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        print("✅ Redis connected.")
        return client
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
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

def get_enhanced_data(ticker: str, limit: int = 390, timeframe: str = "5Min", user_id: str = "default_user") -> Optional[pd.DataFrame]:
    """Fetch historical price data with Redis caching"""
    try:
        key = redis_cache.make_key(user_id, f"ENHANCED_DATA:{ticker}:{timeframe}")
        cached = redis_cache.get(key)
        if cached:
            return pd.DataFrame(cached)

        bars = api.get_bars(ticker, timeframe, limit=limit)
        if bars == "SIP_SUBSCRIPTION_ERROR":
            logger.warning("⚠️ SIP subscription error detected. Using cached data if available.")
            from trading_state import trading_state
            trading_state.trading_disabled = True
            trading_state.disabled_reason = "SIP subscription required"
            return None
        if bars is None or len(bars) == 0:
            return None

        df = pd.DataFrame([{
            "timestamp": bar.t,
            "open": bar.o,
            "high": bar.h,
            "low": bar.l,
            "close": bar.c,
            "volume": bar.v
        } for bar in bars])

        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        redis_cache.set(key, df.to_dict(orient="records"), ttl_seconds=3600)
        return df

    except Exception as e:
        logger.error(f"❌ Failed to fetch enhanced data for {ticker}: {e}")
        return None

# === Instance ===
redis_cache = RedisCache(client)
