import pytz
from datetime import datetime, timedelta
from logger import logger
import api_manager
from config import config
import redis
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    redis_cache = redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    redis_cache = None

def redis_key(user_id, suffix):
    return f"user:{user_id}:{suffix}"

class MarketStatusManager:
    def __init__(self):
        self.market_timezone = pytz.timezone("US/Eastern")

    def is_market_open(self) -> bool:
        try:
            cache_key = redis_key("MARKET_OPEN", config.USER_ID)
            cached = redis_cache.get(cache_key) if redis_cache else None
            if cached is not None:
                return cached

            now = datetime.now(self.market_timezone)

            if api_manager.api:
                clock = api_manager.safe_api_call(api_manager.api.get_clock)
                if clock:
                    is_open = clock.is_open
                    if redis_cache:
                        redis_cache.set(cache_key, is_open, ttl_seconds=60)
                    return is_open

            # Fallback: assume closed on weekends
            if now.weekday() >= 5:
                if redis_cache:
                    redis_cache.set(cache_key, False, ttl_seconds=60)
                return False

            market_open = now.replace(hour=9, minute=30)
            market_close = now.replace(hour=16, minute=0)
            is_open = market_open <= now <= market_close
            if redis_cache:
                redis_cache.set(cache_key, is_open, ttl_seconds=60)
            return is_open

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
                next_open = timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open = timedelta(days=1)
            return next_open - now
        except Exception as e:
            logger.error(f"Market open timing failed: {e}")
            return timedelta(hours=1)
import pytz
from datetime import datetime, timedelta
from logger import logger
import api_manager
from config import config
import redis
import os
from urllib.parse import urlparse

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

redis_cache = get_redis_client()

def redis_key(prefix, user_id):
    return f"user:{user_id}:{prefix}"

class MarketStatusManager:
    def __init__(self):
        self.market_timezone = pytz.timezone("US/Eastern")

    def is_market_open(self) -> bool:
        try:
            cache_key = redis_key("MARKET_OPEN", config.USER_ID)
            cached = None
            if redis_cache:
                try:
                    cached_value = redis_cache.get(cache_key)
                    if cached_value is not None:
                        cached = cached_value.lower() == 'true'
                        return cached
                except Exception:
                    pass

            now = datetime.now(self.market_timezone)

            if api_manager.api:
                clock = api_manager.safe_api_call(api_manager.api.get_clock)
                if clock:
                    is_open = clock.is_open
                    if redis_cache:
                        try:
                            redis_cache.setex(cache_key, 60, str(is_open))
                        except Exception:
                            pass
                    return is_open

            # Fallback: assume closed on weekends
            if now.weekday() >= 5:
                if redis_cache:
                    try:
                        redis_cache.setex(cache_key, 60, 'False')
                    except Exception:
                        pass
                return False

            market_open = now.replace(hour=9, minute=30)
            market_close = now.replace(hour=16, minute=0)
            is_open = market_open <= now <= market_close
            if redis_cache:
                try:
                    redis_cache.setex(cache_key, 60, str(is_open))
                except Exception:
                    pass
            return is_open

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
                next_open = timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open = timedelta(days=1)
            return next_open - now
        except Exception as e:
            logger.error(f"Market open timing failed: {e}")
            return timedelta(hours=1)
