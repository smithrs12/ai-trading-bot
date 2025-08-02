# trading_state.py

from config import config
from datetime import datetime
import api_manager

class TradingState:
    def __init__(self):
        self.reset_all_state()

    def reset_all_state(self):
        self.open_positions = []  # [{ticker, entry_price, qty, etc.}]
        self.qualified_watchlist = []
        self.current_watchlist = [
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "INTC"
        ]
        self.cooldown_map = {}  # {ticker: datetime}
        self.positions_by_ticker = {}
        self.trade_outcomes = []  # Trade logs
        self.sentiment_cache = {}  # {ticker: {score, timestamp}}
        self.model_accuracy = {}  # {ticker: float}
        self.watchlist_performance = {}  # {ticker: float}
        self.risk_metrics = {}  # Sharpe, drawdown, etc.
        self.models_trained = False
        self.eod_liquidation_triggered = False
        self.daily_reset_time = datetime.now().date()
        self.last_watchlist_optimization = datetime.min
        self.last_model_retrain = datetime.min
        self.sector_allocations = {}  # {ticker: sector}
        self.equity_curve = []  # [{time, equity}]
        self.model_confidence_snapshot = {}  # {ticker: {"short_term": x, "medium_term": y}}

        # === SaaS session-aware state ===
        self.market_regime = "neutral"
        self.daily_trade_count = 0
        self.model_predictions = {}
        
        # === Trading control flags ===
        self.trading_disabled = False
        self.disabled_reason = None

        print(f"📂 Initialized trading state for USER_ID = {config.USER_ID}")

    def reset_daily(self):
        print(f"🔁 Resetting daily state for USER_ID = {config.USER_ID}")
        self.open_positions.clear()
        self.cooldown_map.clear()
        self.positions_by_ticker.clear()
        self.trade_outcomes.clear()
        self.sentiment_cache.clear()
        self.eod_liquidation_triggered = False
        self.models_trained = False
        self.daily_trade_count = 0
        self.daily_reset_time = datetime.now().date()

    def update_ultra_advanced_risk_metrics(self):
        print("📊 Updating ultra-advanced risk metrics (placeholder)")
        # TODO: Add drawdown, Sharpe ratio, volatility if needed

    def log_equity_curve(self):
        try:
            account = api_manager.safe_api_call(api_manager.api.get_account)
            if account:
                self.equity_curve.append({
                    "time": datetime.now(),
                    "equity": float(account.equity)
                })
                print(f"📈 Equity curve updated: ${float(account.equity):,.2f}")
        except Exception as e:
            print(f"⚠️ Equity curve logging failed: {e}")

    def disable_due_to_sip(self):
        if not self.trading_disabled:
            print(f"🚫 SIP subscription unavailable. Disabling trading for user {config.USER_ID}")
        self.trading_disabled = True
        self.disabled_reason = "SIP subscription required"
    
    def check_and_enable_trading(self):
        """Check if trading criteria are met and enable trading if they are."""
        try:
            import redis
            import os
            from urllib.parse import urlparse

            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                parsed_url = urlparse(redis_url)
                use_ssl = parsed_url.scheme == "rediss"

                redis_client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    ssl=use_ssl
                )
                active = redis_client.get(f"user:{config.USER_ID}:active")
                if active == "true":
                    if self.trading_disabled:
                        print(f"✅ Trading criteria met. Enabling trading for user {config.USER_ID}")
                        self.trading_disabled = False
                        self.disabled_reason = None
                    return True
                else:
                    if not self.trading_disabled:
                        print(f"🚫 Trading criteria not met. Disabling trading for user {config.USER_ID}")
                        self.trading_disabled = True
                        self.disabled_reason = "No active subscription"
                    return False
            else:
                # If no Redis URL, assume trading is allowed
                if self.trading_disabled:
                    print(f"✅ No Redis available. Enabling trading for user {config.USER_ID}")
                    self.trading_disabled = False
                    self.disabled_reason = None
                return True
        except Exception as e:
            print(f"⚠️ Error checking trading criteria: {e}")
            return False

# === Instantiate session-aware trading state ===
trading_state = TradingState()
meta_model_approved = False
