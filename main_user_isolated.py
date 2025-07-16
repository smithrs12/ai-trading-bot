# main_user_isolated.py

import os
import redis
import json
import hashlib
import uuid
import logging
from urllib.parse import urlparse
from config import config
import api_manager
import pytz
from datetime import datetime, timedelta

logger = logging.getLogger("MarketStatus")

# === Session-Based User ID Setup ===
def get_user_id():
    return os.getenv("USER_SESSION_ID", str(uuid.uuid4()))
config.USER_ID = get_user_id()

# === Redis Setup ===
REDIS_URL = config.REDIS_URL
REDIS_AVAILABLE = False
redis_client = None

try:
    if REDIS_URL:
        parsed_url = urlparse(REDIS_URL)
        redis_client = redis.Redis(
            host=parsed_url.hostname,
            port=parsed_url.port,
            password=parsed_url.password,
            ssl=parsed_url.scheme == 'rediss',
            decode_responses=True
        )
        redis_client.ping()
        REDIS_AVAILABLE = True
        print("✅ Redis connected successfully.")
    else:
        print("⚠️ REDIS_URL not set. Redis disabled.")
except Exception as e:
    print(f"❌ Redis error: {e}")

# === Redis Namespacing ===
def redis_key(*parts):
    return f"{config.USER_ID}:" + ":".join(parts)

# === RedisCache Helper ===
class RedisCache:
    def __init__(self, redis_url):
        self.enabled = bool(redis_url)
        self.client = redis.Redis.from_url(redis_url) if self.enabled else None

    def make_key(self, prefix, payload):
        raw = json.dumps(payload, sort_keys=True)
        return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, key):
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
        except Exception:
            return None

    def set(self, key, value, ttl_seconds=3600):
        if not self.enabled:
            return
        try:
            self.client.setex(key, ttl_seconds, json.dumps(value))
        except Exception:
            pass

redis_cache = RedisCache(REDIS_URL)

# === Market Status Manager ===
class MarketStatusManager:
    def __init__(self):
        self.market_timezone = pytz.timezone("US/Eastern")
        self.last_market_check = None
        self.cached_market_status = False

    def is_market_open(self) -> bool:
        try:
            now = datetime.now(self.market_timezone)
            if self.last_market_check and (now - self.last_market_check).total_seconds() < 60:
                return self.cached_market_status
            if api_manager.api:
                clock = api_manager.safe_api_call(api_manager.api.get_clock)
                if clock:
                    self.cached_market_status = clock.is_open
                    self.last_market_check = now
                    return clock.is_open
            if now.weekday() >= 5:
                self.cached_market_status = False
                self.last_market_check = now
                return False
            market_open = now.replace(hour=9, minute=30)
            market_close = now.replace(hour=16, minute=0)
            self.cached_market_status = market_open <= now <= market_close
            self.last_market_check = now
            return self.cached_market_status
        except Exception as e:
            logger.error(f"Market open check failed: {e}")
            return False

    def is_in_trading_window(self) -> bool:
        try:
            if not self.is_market_open():
                return False
            now = datetime.now(self.market_timezone)
            start = now.replace(hour=10, minute=0)
            end = now.replace(hour=15, minute=45)
            return start <= now <= end
        except Exception as e:
            logger.error(f"Trading window check failed: {e}")
            return False

    def is_near_eod(self) -> bool:
        try:
            now = datetime.now(self.market_timezone)
            close = now.replace(hour=16, minute=0)
            return self.is_market_open() and (close - now <= timedelta(minutes=15))
        except Exception as e:
            logger.error(f"EOD check failed: {e}")
            return False

    def get_time_until_market_open(self) -> timedelta:
        try:
            now = datetime.now(self.market_timezone)
            if self.is_market_open():
                return timedelta(0)
            next_open = now.replace(hour=9, minute=30)
            if now >= next_open:
                next_open += timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            return next_open - now
        except Exception as e:
            logger.error(f"Market open timing failed: {e}")
            return timedelta(hours=1)

market_status = MarketStatusManager()
__all__ = ["market_status", "redis_cache", "config"]
