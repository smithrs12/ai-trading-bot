from config import config
from datetime import datetime
import api_manager

class TradingState:
    def __init__(self):
        self.open_positions = []  # List of dicts: ticker, entry_price, etc.
        self.qualified_watchlist = []  # Filtered/optimized list for trading
        self.current_watchlist = [
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "INTC"
        ]  # Raw candidate list
        self.cooldown_map = {}  # {ticker: datetime}
        self.positions_by_ticker = {}  # Optional: for fast lookup (ticker -> position)
        self.trade_outcomes = []  # Log of trade result data
        self.sentiment_cache = {}  # {ticker: {sentiment_score, timestamp}}
        self.model_accuracy = {}  # {model_name: accuracy}
        self.watchlist_performance = {}  # {ticker: score}
        self.risk_metrics = {}  # Sharpe ratio, drawdown, etc.
        self.models_trained = False
        self.eod_liquidation_triggered = False
        self.daily_reset_time = datetime.now().date()
        self.last_watchlist_optimization = datetime.min
        self.last_model_retrain = datetime.min
        self.sector_allocations = {}  # {ticker: sector}
        self.equity_curve = []

        # 🔧 Added fields
        self.market_regime = "neutral"  # Used on dashboard
        self.daily_trade_count = 0  # Track # of trades per day
        self.model_predictions = {}  # Optional: {ticker: proba}

    def reset_daily(self):
        print("🔁 Resetting daily state")
        self.open_positions = []
        self.cooldown_map = {}
        self.positions_by_ticker = {}
        self.trade_outcomes = []
        self.sentiment_cache = {}
        self.eod_liquidation_triggered = False
        self.models_trained = False
        self.daily_trade_count = 0
        self.daily_reset_time = datetime.now().date()

    def update_ultra_advanced_risk_metrics(self):
        print("📊 Updating ultra advanced risk metrics (placeholder)")
        # Could add drawdown tracking, Sharpe ratio calc, etc.

    def log_equity_curve(self):
        try:
            account = api_manager.safe_api_call(api_manager.api.get_account)
            if account:
                self.equity_curve.append({
                    "time": datetime.now(),
                    "equity": float(account.equity)
                })
        except Exception as e:
            print(f"⚠️ Equity curve logging failed: {e}")

trading_state = TradingState()
