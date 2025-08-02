from datetime import datetime, timedelta
from config import config
from trading_state import trading_state
import logger

class RiskManager:
    def __init__(self):
        self.max_drawdown_threshold = 0.20  # 20%
        self.recent_pnl_window = 10  # last 10 trades
        self.max_daily_loss = -0.10  # -10% equity
        self.last_check = datetime.min

    def check_max_drawdown(self) -> bool:
        """
        Evaluates recent trade outcomes for excessive losses.
        Returns True if trading should be paused.
        """
        try:
            outcomes = trading_state.trade_outcomes[-self.recent_pnl_window:]
            if not outcomes:
                return False

            cumulative_pnl = sum([o.get("pnl", 0) for o in outcomes])
            if cumulative_pnl < -self.max_drawdown_threshold:
                logger.warning(f"⚠️ Max drawdown hit: {cumulative_pnl:.2f}")
                return True

        except Exception as e:
            logger.error(f"❌ Drawdown check failed: {e}")
        return False

    def check_daily_loss_limit(self) -> bool:
        """
        Optional: Check daily net loss.
        """
        # You can implement this using timestamps if desired
        return False

    def block_trades_if_risky(self) -> bool:
        """
        Wrapper check for all risk controls.
        Returns True if trade activity should be paused.
        """
        if self.check_max_drawdown():
            return True
        if self.check_daily_loss_limit():
            return True
        return False

    def update_risk_metrics(self):
        """
        Recalculate and store risk metrics for reporting.
        """
        try:
            outcomes = trading_state.trade_outcomes[-self.recent_pnl_window:]
            if not outcomes:
                return

            returns = [o.get("pnl", 0) for o in outcomes]
            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5

            sharpe = (avg_return / volatility) if volatility > 0 else 0

            trading_state.risk_metrics["sharpe_ratio"] = round(sharpe, 3)
            trading_state.risk_metrics["avg_return"] = round(avg_return, 4)
            trading_state.risk_metrics["volatility"] = round(volatility, 4)

        except Exception as e:
            logger.error(f"❌ Risk metric update failed: {e}")

risk_manager = RiskManager()
# risk_management.py

from datetime import datetime, timedelta
from config import config
from trading_state import trading_state
from logger import logger

class RiskManager:
    def __init__(self):
        self.max_drawdown_threshold = 0.20  # 20%
        self.recent_pnl_window = 10  # last 10 trades
        self.max_daily_loss = -0.10  # -10% equity
        self.last_check = datetime.min

    def check_max_drawdown(self) -> bool:
        """
        Evaluates recent trade outcomes for excessive losses.
        Returns True if trading should be paused.
        """
        try:
            outcomes = trading_state.trade_outcomes[-self.recent_pnl_window:]
            if not outcomes:
                return False

            cumulative_pnl = sum([o.get("pnl", 0) for o in outcomes])
            if cumulative_pnl < -self.max_drawdown_threshold:
                logger.warning(f"⚠️ Max drawdown hit: {cumulative_pnl:.2f}")
                return True

        except Exception as e:
            logger.error(f"❌ Drawdown check failed: {e}")
        return False

    def check_daily_loss_limit(self) -> bool:
        """
        Optional: Check daily net loss.
        """
        # You can implement this using timestamps if desired
        return False

    def block_trades_if_risky(self) -> bool:
        """
        Wrapper check for all risk controls.
        Returns True if trade activity should be paused.
        """
        if self.check_max_drawdown():
            return True
        if self.check_daily_loss_limit():
            return True
        return False

    def update_risk_metrics(self):
        """
        Recalculate and store risk metrics for reporting.
        """
        try:
            outcomes = trading_state.trade_outcomes[-self.recent_pnl_window:]
            if not outcomes:
                return

            returns = [o.get("pnl", 0) for o in outcomes]
            avg_return = sum(returns) / len(returns)
            volatility = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5

            sharpe = (avg_return / volatility) if volatility > 0 else 0

            trading_state.risk_metrics["sharpe_ratio"] = round(sharpe, 3)
            trading_state.risk_metrics["avg_return"] = round(avg_return, 4)
            trading_state.risk_metrics["volatility"] = round(volatility, 4)

        except Exception as e:
            logger.error(f"❌ Risk metric update failed: {e}")

risk_manager = RiskManager()
