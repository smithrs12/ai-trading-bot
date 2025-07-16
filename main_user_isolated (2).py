# === main_user_isolated.py ===

import os
import json
import time
import uuid
import redis
import hashlib
import logging
import random
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from urllib.parse import urlparse

from config import config
import api_manager

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

# === Redis Key Namespacing ===
def redis_key(*parts):
    return f"{config.USER_ID}:" + ":".join(parts)

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
            current_time = datetime.now(self.market_timezone)

            if self.last_market_check and (current_time - self.last_market_check).total_seconds() < 60:
                return self.cached_market_status

            if api_manager.api:
                clock = api_manager.safe_api_call(api_manager.api.get_clock)
                if clock:
                    self.cached_market_status = clock.is_open
                    self.last_market_check = current_time
                    return clock.is_open

            if current_time.weekday() >= 5:
                self.cached_market_status = False
                self.last_market_check = current_time
                return False

            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            is_open = market_open <= current_time <= market_close
            self.cached_market_status = is_open
            self.last_market_check = current_time
            return is_open

        except Exception as e:
            logger.error(f"Market status check failed: {e}")
            return False

    def is_in_trading_window(self) -> bool:
        try:
            if not self.is_market_open():
                return False
            now = datetime.now(self.market_timezone)
            trade_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
            trade_end = now.replace(hour=15, minute=45, second=0, microsecond=0)
            return trade_start <= now <= trade_end
        except Exception as e:
            logger.error(f"Trading window check failed: {e}")
            return False

    def is_near_eod(self) -> bool:
        try:
            now = datetime.now(self.market_timezone)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            return self.is_market_open() and (market_close - now) <= timedelta(minutes=15)
        except Exception as e:
            logger.error(f"EOD check failed: {e}")
            return False

    def should_hold_overnight(self, position_data: Dict) -> bool:
        try:
            if not config.EOD_LIQUIDATION_ENABLED:
                return True
            entry_price = position_data.get("entry_price", 0)
            current_price = position_data.get("current_price", entry_price)
            if entry_price > 0:
                profit_pct = (current_price - entry_price) / entry_price
                if profit_pct >= config.OVERNIGHT_HOLD_PROFIT_THRESHOLD:
                    logger.info(f"Holding {position_data.get('ticker')} overnight - Profit: {profit_pct:.2%}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Overnight hold logic failed: {e}")
            return False

    def get_time_until_market_open(self) -> timedelta:
        try:
            current_time = datetime.now(self.market_timezone)
            if self.is_market_open():
                return timedelta(0)
            next_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            if current_time >= next_open:
                next_open += timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            return next_open - current_time
        except Exception as e:
            logger.error(f"Time until market open calculation failed: {e}")
            return timedelta(hours=1)

def main_loop(user_id):
    """Main 24/7 trading loop with market awareness"""
    last_heartbeat = datetime.min
    last_watchlist_optimization = datetime.min
    last_model_retrain = datetime.min
    last_eod_check = datetime.now().date()
    loop_count = 0

    while True:
        try:
            current_time = datetime.now()
            loop_count += 1

            # Send heartbeat
            if (current_time - last_heartbeat).total_seconds() >= 60:
                heartbeat_monitor.send_heartbeat()
                last_heartbeat = current_time

            # Check market status
            market_open = market_status.is_market_open()
            trading_window = market_status.is_in_trading_window()
            near_eod = market_status.is_near_eod()

            logger.deduped_log(f"📊 Market Status: Open={market_open}, Trading Window={trading_window}, Near EOD={near_eod}")

            if market_open:
                if near_eod and not trading_state.eod_liquidation_triggered:
                    perform_eod_liquidation()

                elif trading_window:
                    logger.deduped_log("info", "📈 In trading window - executing trading logic")

                    if (current_time - last_watchlist_optimization).total_seconds() >= config.DYNAMIC_WATCHLIST_REFRESH_HOURS * 3600:
                        optimized_watchlist = watchlist_optimizer.optimize_watchlist()
                        last_watchlist_optimization = current_time
                        logger.deduped_log("info", f"✅ Watchlist optimized: {len(optimized_watchlist)} tickers")

                    current_watchlist = trading_state.qualified_watchlist or trading_state.current_watchlist
                    meta_approval_system.evaluate_model_performance()

                    trades_executed = 0
                    for ticker in current_watchlist:
                        try:
                            if ultra_advanced_trading_logic(ticker):
                                trades_executed += 1
                            time.sleep(1)
                        except Exception as e:
                            logger.error(f"❌ Error processing {ticker}: {e}")
                            continue

                    logger.deduped_log("info", f"📊 Trading cycle complete: {trades_executed} trades executed")

                    if (current_time - last_model_retrain).total_seconds() >= config.MODEL_RETRAIN_FREQUENCY_HOURS * 3600:
                        logger.deduped_log("info", "🔄 Periodic model retraining...")
                        qualified_tickers = trading_state.qualified_watchlist or trading_state.current_watchlist[:config.MIN_TICKERS_FOR_TRAINING]
                        ensemble_model.train_dual_horizon_ensemble(qualified_tickers)
                        trading_state.models_trained = True
                        last_model_retrain = current_time

                else:
                    time_until_trading = market_status.get_time_until_market_open()
                    if time_until_trading.total_seconds() > 0:
                        wait_minutes = min(30 - (current_time.hour * 60 + current_time.minute - 9 * 60 - 30), 30)
                        logger.deduped_log("info", f"⏰ Waiting {wait_minutes} minutes for trading window to open")

                    logger.deduped_log("info", "🔧 Performing maintenance tasks...")
                    trading_state.update_ultra_advanced_risk_metrics()

                    current_time_ts = current_time.timestamp()
                    for ticker in list(trading_state.sentiment_cache.keys()):
                        cache_entry = trading_state.sentiment_cache[ticker]
                        if isinstance(cache_entry, dict) and 'timestamp' in cache_entry:
                            if (current_time_ts - cache_entry['timestamp'].timestamp()) > 3600:
                                del trading_state.sentiment_cache[ticker]

            else:
                logger.deduped_log("info", "🌙 Market closed - performing maintenance and monitoring")

                if current_time.date() != last_eod_check:
                    logger.deduped_log("info", "🌅 New trading day - resetting daily state")
                    trading_state.reset_daily()
                    last_eod_check = current_time.date()

                if loop_count % 10 == 0:
                    logger.deduped_log("info", "🔧 Performing extended maintenance...")

                    try:
                        backup_data = {
                            'trade_outcomes': trading_state.trade_outcomes[-100:],
                            'risk_metrics': trading_state.risk_metrics,
                            'model_accuracy': trading_state.model_accuracy,
                            'watchlist_performance': trading_state.watchlist_performance
                        }

                        os.makedirs('backups', exist_ok=True)
                        backup_filename = f"backups/backup_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                        with open(backup_filename, 'w') as f:
                            json.dump(backup_data, f, indent=2, default=str)

                        logger.deduped_log("info", f"💾 Data backup created: {backup_filename}")

                    except Exception as e:
                        logger.error(f"❌ Backup failed: {e}")

                    try:
                        qualified = trading_state.qualified_watchlist or trading_state.current_watchlist
                        if len(qualified) >= config.MIN_TICKERS_FOR_TRAINING:
                            ensemble_model.retrain_meta_model()
                            ensemble_model.train_dual_horizon_ensemble(qualified[:config.MIN_TICKERS_FOR_TRAINING])
                            logger.deduped_log("info", "✅ Daily model retraining completed.")
                        else:
                            logger.warning(f"⚠️ Skipping retraining - only {len(qualified)} tickers available")
                    except Exception as e:
                        logger.error(f"❌ Model retraining failed: {e}")

                time_until_open = market_status.get_time_until_market_open()
                if time_until_open.total_seconds() > 3600:
                    logger.deduped_log("info", f"⏰ Market opens in {time_until_open}. Sleeping for 5 minutes...")
                    time.sleep(300)
                    continue

            time.sleep(30)

        except KeyboardInterrupt:
            logger.deduped_log("info", "🛑 Received shutdown signal")
            break

        except Exception as e:
            logger.error(f"❌ Main loop error: {e}")

# === Exported Symbols ===
market_status = MarketStatusManager()
__all__ = ["main_loop", "market_status", "config"]
