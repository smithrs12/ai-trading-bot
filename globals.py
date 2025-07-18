# globals.py

from config import config
from logger import logger
from redis_cache import redis_cache, redis_key
from trading_state import trading_state
from market_status_manager import MarketStatusManager

# Singleton market status instance
market_status = MarketStatusManager()
