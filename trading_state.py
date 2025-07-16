
# trading_state.py

from config import config
from datetime import datetime

class TradingState:
    def __init__(self):
        self.open_positions = []
        self.qualified_watchlist = []
        self.current_watchlist = []
        self.trade_outcomes = []
        self.risk_metrics = {}
        self.model_accuracy = {}
        self.watchlist_performance = {}
        self.sentiment_cache = {}
        self.models_trained = False
        self.eod_liquidation_triggered = False
        self.daily_reset_time = datetime.now().date()

    def reset_daily(self):
        print("🔁 Resetting daily state")
        self.open_positions = []
        self.trade_outcomes = []
        self.sentiment_cache = {}
        self.eod_liquidation_triggered = False
        self.daily_reset_time = datetime.now().date()

    def update_ultra_advanced_risk_metrics(self):
        print("📊 Updating ultra advanced risk metrics (placeholder)")

trading_state = TradingState()
