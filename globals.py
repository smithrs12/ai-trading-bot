# globals.py
import os
import redis
from urllib.parse import urlparse
from redis_cache import RedisCache
from key_utils import redis_key

# --- Redis Client ---
def get_redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        parsed = urlparse(redis_url)
        client = redis.Redis(
            host=parsed.hostname,
            port=parsed.port,
            password=parsed.password,
            ssl=parsed.scheme == 'rediss',
            decode_responses=True
        )
        client.ping()
        return client
    except Exception:
        return None

client = get_redis_client()
redis_cache = RedisCache(client)

# --- Lazy Imports ---
def get_config():
    from config import config
    return config

def get_trading_state():
    from trading_state import trading_state
    return trading_state

def get_market_status():
    from market_status_manager import MarketStatusManager
    return MarketStatusManager()

# --- Singletons ---
config = get_config()
trading_state = get_trading_state()
market_status = get_market_status()
