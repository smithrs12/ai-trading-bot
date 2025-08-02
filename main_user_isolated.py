import os
import time
import json
import uuid
import logging
import redis
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from alpaca_trade_api.rest import REST

from config import config
import api_manager
from market_regime import MarketRegimeDetector
from trading_state import trading_state
from logger import logger
from redis_cache import redis_cache

# -----------------------------------------------------------------------------
# Redis helpers
# -----------------------------------------------------------------------------

def get_redis_client():
    """Create a Redis client that supports redis:// and rediss:// with decode_responses."""
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        parsed = urlparse(url)
        use_ssl = parsed.scheme == "rediss"
        rc = redis.Redis.from_url(url, decode_responses=True, ssl=use_ssl)
        rc.ping()
        print("✅ Redis connected (worker)")
        return rc
    except Exception as e:
        print(f"⚠️ Redis not available in worker: {e}")
        return None

redis_client = get_redis_client()

# -----------------------------------------------------------------------------
# Settings & Mode management (from Redis)
# -----------------------------------------------------------------------------

def _read_settings_from_redis(user_id: str) -> dict:
    """Return user settings hash or defaults."""
    defaults = {
        "trading_mode": "Paper Trading",
        "max_positions": 10,
        "confidence_threshold": 0.65,
        "max_trade_amount": 10000,
        "rsi_min": 30,
        "rsi_max": 70,
    }
    if not (redis_client and user_id):
        return defaults
    try:
        raw = redis_client.hgetall(f"user:{user_id}:settings") or {}
        # Coerce
        def _to_int(v, d): 
            try: return int(float(v))
            except Exception: return d
        def _to_float(v, d):
            try: return float(v)
            except Exception: return d
        return {
            "trading_mode": raw.get("trading_mode", defaults["trading_mode"]),
            "max_positions": _to_int(raw.get("max_positions"), defaults["max_positions"]),
            "confidence_threshold": _to_float(raw.get("confidence_threshold"), defaults["confidence_threshold"]),
            "max_trade_amount": _to_int(raw.get("max_trade_amount"), defaults["max_trade_amount"]),
            "rsi_min": _to_int(raw.get("rsi_min"), defaults["rsi_min"]),
            "rsi_max": _to_int(raw.get("rsi_max"), defaults["rsi_max"]),
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to read settings from Redis: {e}")
        return defaults

def _apply_settings_to_runtime(settings: dict):
    """Apply slider values into the running config (this process only)."""
    try:
        # Guard RSI ordering
        rsi_min = min(int(settings["rsi_min"]), int(settings["rsi_max"]))
        rsi_max = max(int(settings["rsi_min"]), int(settings["rsi_max"]))

        config.MAX_POSITIONS = int(settings["max_positions"])
        config.ENSEMBLE_CONFIDENCE_THRESHOLD = float(settings["confidence_threshold"])
        config.RSI_MIN = rsi_min
        config.RSI_MAX = rsi_max
        # Some codepaths use FIXED_TRADE_AMOUNT vs MAX_TRADE_AMOUNT — set both for safety
        config.FIXED_TRADE_AMOUNT = int(settings["max_trade_amount"])
        config.MAX_TRADE_AMOUNT = int(settings["max_trade_amount"])

        logger.info(
            f"⚙️ Applied settings → "
            f"MODE={settings['trading_mode']}, "
            f"MAX_POS={config.MAX_POSITIONS}, "
            f"CONF={config.ENSEMBLE_CONFIDENCE_THRESHOLD:.2f}, "
            f"RSI=[{config.RSI_MIN},{config.RSI_MAX}], "
            f"MAX_TRADE=${config.MAX_TRADE_AMOUNT:,}"
        )
    except Exception as e:
        logger.warning(f"⚠️ Failed to apply settings to runtime: {e}")

def _read_trading_mode_from_redis(user_id: str) -> str:
    """
    Determine Paper vs Live from:
      1) user:{id}:mode JSON blob: {"paper": true/false}
      2) Fallback to settings hash 'trading_mode'
    Returns one of: "paper" | "live"
    """
    # Preferred JSON blob (what your settings page writes)
    if redis_client:
        try:
            raw = redis_client.get(f"user:{user_id}:mode")
            if raw:
                obj = json.loads(raw)
                return "paper" if obj.get("paper", True) else "live"
        except Exception as e:
            logger.warning(f"⚠️ Could not read user:{user_id}:mode JSON: {e}")

    # Fallback to settings hash
    s = _read_settings_from_redis(user_id).get("trading_mode", "Paper Trading").strip().lower()
    return "paper" if "paper" in s else "live"

def _load_alpaca_creds(user_id: str) -> dict:
    """Read the consolidated creds hash your UI writes: user:{id}:alpaca"""
    if not (redis_client and user_id):
        return {"api_key_paper": "", "secret_key_paper": "", "api_key_live": "", "secret_key_live": ""}
    try:
        h = redis_client.hgetall(f"user:{user_id}:alpaca") or {}
        # Trim strings
        return {k: ("" if v is None else str(v).strip()) for k, v in h.items()}
    except Exception as e:
        logger.warning(f"⚠️ Failed to read Alpaca creds from Redis: {e}")
        return {"api_key_paper": "", "secret_key_paper": "", "api_key_live": "", "secret_key_live": ""}

def _connect_alpaca_for_scope(user_id: str, scope: str) -> bool:
    """
    Connect api_manager.api to Paper or Live using creds from Redis.
    scope ∈ {"paper","live"}
    """
    creds = _load_alpaca_creds(user_id)
    if scope == "paper":
        key = creds.get("api_key_paper", "")
        sec = creds.get("secret_key_paper", "")
        base = "https://paper-api.alpaca.markets"
        config.PAPER_TRADING_MODE = True
    else:
        key = creds.get("api_key_live", "")
        sec = creds.get("secret_key_live", "")
        base = "https://api.alpaca.markets"
        config.PAPER_TRADING_MODE = False

    if not key or not sec:
        logger.warning(f"🚫 Missing {scope.upper()} credentials for user {user_id}.")
        return False

    try:
        client = REST(key_id=key, secret_key=sec, base_url=base, api_version="v2")
        _ = client.get_account()  # auth sanity check
        api_manager.api = client  # set the global used elsewhere
        # Optional: SIP check (non-fatal)
        try:
            client.get_latest_quote("AAPL")
            sip_ok = True
        except Exception as e2:
            sip_ok = False
            if any(s in str(e2).lower() for s in ("403", "forbidden", "permission", "subscription")):
                logger.info("ℹ️ Connected, but SIP (real-time market data) is not enabled for this account.")
        logger.info(f"🔌 Connected Alpaca ({scope.upper()}), SIP_OK={sip_ok}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect Alpaca ({scope}): {e}")
        return False

# -----------------------------------------------------------------------------
# Subscription helper (optional; keeps prior behavior)
# -----------------------------------------------------------------------------

def check_subscription(user_id: str) -> bool:
    if redis_client is None:
        return True
    try:
        return (redis_client.get(f"user:{user_id}:active") == "true")
    except Exception:
        return True  # don't hard-fail worker

# -----------------------------------------------------------------------------
# Runtime refresh coordinator
# -----------------------------------------------------------------------------

class RuntimeController:
    """
    Tracks last-applied settings/mode and refreshes api_manager + config
    when Redis values change.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.scope = None           # "paper" | "live"
        self.settings_sig = None    # signature of last-applied settings

    def _signature(self, s: dict) -> str:
        try:
            # stable signature to detect changes
            return json.dumps(s, sort_keys=True)
        except Exception:
            return str(s)

    def initial_sync(self):
        self.refresh(force=True)

    def refresh(self, force: bool = False):
        # 1) Apply latest settings
        settings = _read_settings_from_redis(self.user_id)
        sig = self._signature(settings)
        if force or sig != self.settings_sig:
            _apply_settings_to_runtime(settings)
            self.settings_sig = sig

        # 2) Ensure Alpaca is connected to the right scope
        desired_scope = _read_trading_mode_from_redis(self.user_id)  # "paper"|"live"
        if force or desired_scope != self.scope or api_manager.api is None:
            if _connect_alpaca_for_scope(self.user_id, desired_scope):
                self.scope = desired_scope
                logger.info(f"✅ Worker is now using {self.scope.upper()} environment")
            else:
                # If connection fails, prevent trading but keep loop alive
                trading_state.trading_disabled = True
                trading_state.disabled_reason = f"No valid {desired_scope} credentials"
                logger.warning(f"🚫 Trading disabled: {trading_state.disabled_reason}")

# -----------------------------------------------------------------------------
# Main trading loop
# -----------------------------------------------------------------------------

from datetime import datetime, timedelta
import time
from config import config
from globals import redis_cache
from logger import logger
from market_regime import MarketRegimeDetector
from trading_state import trading_state
from runtime_controller import RuntimeController
import api_manager
from subscription_checker import check_subscription

def main_loop(user_id: str):
    """
    Ultra-Advanced 24/7 AI Trading Loop
    - Per-user isolation via config.USER_ID
    - Auto-applies new settings and mode without restart
    """
    regime_detector = MarketRegimeDetector()
    ctrl = RuntimeController(user_id)
    ctrl.initial_sync()

    last_heartbeat = datetime.min.replace(tzinfo=None)
    last_watchlist_refresh = datetime.min
    last_model_retrain = datetime.min
    last_equity_log = datetime.min
    last_regime_check = datetime.min
    last_subscription_check = datetime.min
    last_runtime_refresh = datetime.min
    loop_count = 0

    from execution_manager import ultra_advanced_trading_logic, perform_eod_liquidation
    from watchlist_optimizer import optimize_watchlist

    while True:
        try:
            now = datetime.now()

            # === Periodic runtime refresh ===
            if (now - last_runtime_refresh).total_seconds() > 30:
                ctrl.refresh()
                last_runtime_refresh = now

            loop_count += 1

            # === Reload Redis settings ===
            try:
                config.reload_user_settings()
            except Exception as e:
                logger.warning(f"🔁 Failed to reload user settings: {e}")

            # === SIP sanity check ===
            try:
                if api_manager.api:
                    _ = api_manager.api.get_clock()
                else:
                    logger.warning("⚠️ No Alpaca client; trading disabled until credentials are added.")
            except Exception as e:
                if "subscription" in str(e).lower():
                    logger.warning("🚫 SIP subscription required; throttling live data usage.")
                    trading_state.trading_disabled = True
                    trading_state.disabled_reason = "SIP subscription required"

            # === Heartbeat ===
            if (now - last_heartbeat).total_seconds() > 60:
                print(f"❤️ [{config.USER_ID}] Heartbeat {loop_count} @ {now.strftime('%H:%M:%S')}")
                last_heartbeat = now

            # === Market status check ===
            from market_status_manager import MarketStatusManager
            msm = MarketStatusManager()
            market_open = msm.is_market_open()
            trading_window = msm.is_in_trading_window()
            near_eod = msm.is_near_eod()

            print(f"📊 [{config.USER_ID}] Market Open={market_open}, Window={trading_window}, NearEOD={near_eod}")

            # === Regime Detection (every 10m) ===
            if (now - last_regime_check).total_seconds() > 600:
                try:
                    trading_state.market_regime = regime_detector.detect_market_regime()
                    print(f"🔍 Market Regime: {trading_state.market_regime}")
                except Exception as e:
                    logger.warning(f"⚠️ Market regime detection failed: {e}")
                    trading_state.market_regime = "neutral"
                last_regime_check = now

            # === Subscription Check (every 5m) ===
            if (now - last_subscription_check).total_seconds() > 300:
                active = check_subscription(config.USER_ID)
                if not active:
                    trading_state.trading_disabled = True
                    trading_state.disabled_reason = "No active subscription"
                    print(f"🚫 [{config.USER_ID}] Subscription inactive; trading disabled")
                else:
                    print(f"✅ [{config.USER_ID}] Subscription active")
                last_subscription_check = now

            # === Watchlist Refresh (per config interval) ===
            if (now - last_watchlist_refresh) > timedelta(hours=getattr(config, "DYNAMIC_WATCHLIST_REFRESH_HOURS", 6)):
                try:
                    optimized = optimize_watchlist(user_id=config.USER_ID)
                    if optimized:
                        trading_state.qualified_watchlist = optimized
                except Exception as e:
                    logger.warning(f"⚠️ Watchlist optimization failed: {e}")
                last_watchlist_refresh = now

            # === Model Retraining (every hour) ===
            if (now - last_model_retrain).total_seconds() > 3600:
                try:
                    from model_training import ensemble_model
                    tickers = trading_state.qualified_watchlist or trading_state.current_watchlist
                    success = ensemble_model.train_all_models(tickers, config.USER_ID)
                    print(f"🧠 [{config.USER_ID}] Models retrained: {'✅' if success else '⚠️ Failed'}")
                except Exception as e:
                    logger.warning(f"⚠️ Model retraining error: {e}")
                last_model_retrain = now

            # === Equity Logging (every 5 min) ===
            if (now - last_equity_log).total_seconds() > 300:
                try:
                    if api_manager.api and hasattr(trading_state, "equity_curve"):
                        account = api_manager.api.get_account()
                        trading_state.equity_curve.append({
                            "time": now,
                            "equity": float(account.equity)
                        })
                        print(f"💹 [{config.USER_ID}] Equity logged: ${float(account.equity):,.2f}")
                except Exception as e:
                    logger.warning(f"⚠️ Equity logging failed: {e}")
                last_equity_log = now

            # === Live Trading ===
            if market_open and trading_window:
                if getattr(trading_state, "trading_disabled", False):
                    print(f"🚫 [{config.USER_ID}] Trading disabled: {trading_state.disabled_reason}")
                else:
                    candidates = trading_state.qualified_watchlist or trading_state.current_watchlist
                    for ticker in candidates[:getattr(config, "WATCHLIST_LIMIT", 20)]:
                        ultra_advanced_trading_logic(ticker)

            # === End-of-Day Liquidation ===
            if market_open and near_eod and not trading_state.eod_liquidation_triggered:
                perform_eod_liquidation()

            time.sleep(10)

        except KeyboardInterrupt:
            logger.info("🛑 Worker interrupted — shutting down.")
            break
        except Exception as e:
            logger.error(f"❌ Main loop error [{config.USER_ID}]: {e}")
            time.sleep(5)

def _resolve_user_id() -> str:
    # Align with your UI; prefer a stable ID the UI writes (e.g., USER_SESSION_ID)
    return os.getenv("USER_SESSION_ID", "background-worker")

if __name__ == "__main__":
    # Bind the per-user isolation into config
    config.USER_ID = _resolve_user_id()
    print(f"🔒 Worker started for USER_ID={config.USER_ID}")

    # Soft gate on subscription (won't crash the worker)
    if not check_subscription(config.USER_ID):
        trading_state.trading_disabled = True
        trading_state.disabled_reason = "No active subscription"
        print(f"🚫 User {config.USER_ID} is not subscribed. Trading disabled but worker remains active.")
    else:
        trading_state.trading_disabled = False
        trading_state.disabled_reason = None

    main_loop(user_id=config.USER_ID)
