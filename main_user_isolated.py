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
from execution_manager import ultra_advanced_trading_logic, perform_eod_liquidation, get_price
from trading_state import trading_state
from regime_detection import detect_market_regime

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

from execution_manager import ultra_advanced_trading_logic, perform_eod_liquidation, get_price
from trading_state import trading_state
from datetime import datetime, timedelta
from regime_detection import detect_market_regime
import time

def main_loop(user_id=None):
    """Ultra-Advanced 24/7 AI Trading Loop"""
    last_heartbeat = datetime.min
    last_watchlist_refresh = datetime.min
    last_model_retrain = datetime.min
    last_equity_log = datetime.min
    last_regime_check = datetime.min
    loop_count = 0

    while True:
        try:
            now = datetime.now()
            loop_count += 1

            # === Heartbeat (every 60s) ===
            if (now - last_heartbeat).total_seconds() > 60:
                print(f"❤️ Heartbeat {loop_count} @ {now.strftime('%H:%M:%S')}")
                last_heartbeat = now

            market_open = market_status.is_market_open()
            trading_window = market_status.is_in_trading_window()
            near_eod = market_status.is_near_eod()

            print(f"📊 Market Open={market_open}, Window={trading_window}, Near EOD={near_eod}")

            # === Regime Detection (every 10m) ===
            if (now - last_regime_check).total_seconds() > 600:
                trading_state.market_regime = detect_market_regime()
                print(f"🔍 Market Regime: {trading_state.market_regime}")
                last_regime_check = now

            # === Dynamic Watchlist Refresh ===
            if (now - last_watchlist_refresh) > timedelta(hours=config.DYNAMIC_WATCHLIST_REFRESH_HOURS):
                from watchlist_optimizer import optimize_watchlist
                optimize_watchlist()
                last_watchlist_refresh = now
                print("✅ Watchlist refreshed")

            # === Model Retraining ===
            if (now - last_model_retrain).total_seconds() > 3600:
                from model_training import train_all_models
                train_all_models()
                last_model_retrain = now
                print("🧠 Models retrained")

            # === Real-Time Equity Logging ===
            if (now - last_equity_log).total_seconds() > 300:
                try:
                    account = api_manager.safe_api_call(api_manager.api.get_account)
                    if account and hasattr(trading_state, "equity_curve"):
                        trading_state.equity_curve.append({
                            "time": now,
                            "equity": float(account.equity)
                        })
                        print(f"💹 Equity logged: ${float(account.equity):,.2f}")
                    last_equity_log = now
                except Exception as e:
                    logger.warning(f"⚠️ Equity logging failed: {e}")

            # === Trade Execution ===
            if market_open and trading_window:
                candidates = trading_state.qualified_watchlist or trading_state.current_watchlist
                for ticker in candidates[:config.WATCHLIST_LIMIT]:
                    ultra_advanced_trading_logic(ticker)

            # === End of Day Liquidation ===
            if market_open and near_eod and not trading_state.eod_liquidation_triggered:
                perform_eod_liquidation()

            time.sleep(10)

        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")
            time.sleep(5)
