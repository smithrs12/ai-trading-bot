# main_user_isolated.py

import os
import json
import hashlib
import uuid
import logging
from urllib.parse import urlparse
from config import config
import api_manager
import pytz
from datetime import datetime, timedelta
from globals import market_status, redis_cache, redis_key
from trading_state import trading_state
from market_regime import detect_market_regime
import redis
if redis_cache.enabled:
    print("✅ Redis is active and ready.")
else:
    print("⚠️ Redis is not enabled.")

dlogger = logging.getLogger("MarketStatus")

# === Session-Based User ID Setup ===
def get_user_id():
    return os.getenv("USER_SESSION_ID", str(uuid.uuid4()))
config.USER_ID = get_user_id()

# === Main Trading Loop ===
def main_loop(user_id=None):
    """Ultra-Advanced 24/7 AI Trading Loop"""
    from execution_manager import ultra_advanced_trading_logic, perform_eod_liquidation
    from watchlist_optimizer import optimize_watchlist
    from model_training import train_all_models

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
                optimize_watchlist()
                last_watchlist_refresh = now
                print("✅ Watchlist refreshed")

            # === Model Retraining ===
            if (now - last_model_retrain).total_seconds() > 3600:
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
                    dlogger.warning(f"⚠️ Equity logging failed: {e}")

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
            dlogger.error(f"❌ Main loop error: {e}")
            time.sleep(5)
