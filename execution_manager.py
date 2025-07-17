# execution_manager.py

import time
from datetime import datetime

from config import config
from trading_state import trading_state
from main_user_isolated import market_status
from ensemble_model import ensemble_model
from reinforcement import PyTorchQLearningAgent
from technical_indicators import passes_vwap, passes_volume_spike, extract_features
from meta_approval_system import meta_approval_system
from risk_management import risk_manager
from sentiment_analysis import get_sentiment
import api_manager
import logger

# === Reinforcement Learning Agent ===
q_agent = PyTorchQLearningAgent()

# === Decision Mapping ===
ACTION_MAP = {0: 'buy', 1: 'sell', 2: 'hold'}

# === Ultra Advanced Trading Logic ===
def ultra_advanced_trading_logic(ticker: str) -> bool:
    """Performs signal checks, model prediction, and executes a trade if approved."""
    try:
        if not passes_all_filters(ticker):
            return False

        # Get features + confidence from ensemble
        features = extract_features(ticker)
        confidence = ensemble_model.predict_weighted_proba(ticker)
        action_index = q_agent.act(features)
        action = ACTION_MAP[action_index]

        logger.deduped_log("debug", f"📊 {ticker} - RL Decision: {action} | Confidence: {confidence:.2f}")

        if not passes_sector_allocation(ticker):
            return False

        if risk_manager.block_trades_if_risky():
            logger.logger.warning(f"🚫 Trade blocked due to risk controls for {ticker}")
            return False

        if not is_meta_approved(ticker, confidence):
            return False

        if action == 'buy' and can_enter_position(ticker):
            size = calculate_kelly_position_size(ticker, confidence)
            return execute_buy(ticker, features, size)

        elif action == 'sell' and has_open_position(ticker):
            if should_exit_due_to_profit_decay(ticker):
                return execute_sell(ticker, features)

        elif should_add_to_position(ticker, confidence):
            logger.deduped_log("info", f"➕ Adding to {ticker} position (pyramiding)")
            size = calculate_kelly_position_size(ticker, confidence)
            return execute_buy(ticker, features, size)

        return False

    except Exception as e:
        logger.error(f"❌ Trade logic error for {ticker}: {e}")
        return False
        
# === End of Day Liquidation ===
def perform_eod_liquidation():
    """Sell all open positions before market close if EOD liquidation is enabled."""
    if not config.EOD_LIQUIDATION_ENABLED:
        return

    logger.deduped_log("info", "🧯 Performing EOD liquidation")
    for pos in trading_state.open_positions:
        ticker = pos.get("ticker")
        if ticker:
            try:
                execute_sell(ticker)
                logger.deduped_log("info", f"💣 EOD liquidation: {ticker}")
            except Exception as e:
                logger.error(f"❌ Failed to liquidate {ticker}: {e}")
    trading_state.eod_liquidation_triggered = True

# === Helper: Filter Logic ===
def passes_all_filters(ticker: str) -> bool:
    """Combines all filter checks — sentiment, momentum, volume, etc."""
    from sentiment_analysis import get_sentiment
    from technical_indicators import passes_vwap, passes_volume_spike

    if get_sentiment(ticker) < 0:
        return False

    if not passes_vwap(ticker):
        return False

    if not passes_volume_spike(ticker):
        return False

    return True

# === Helper: Buy Conditions ===
def can_enter_position(ticker: str) -> bool:
    """Checks if a new position in ticker can be opened."""
    if len(trading_state.open_positions) >= config.MAX_POSITIONS:
        return False
    if is_on_cooldown(ticker):
        return False
    return True

def is_on_cooldown(ticker: str) -> bool:
    cooldowns = trading_state.cooldown_map if hasattr(trading_state, "cooldown_map") else {}
    cooldown = cooldowns.get(ticker)
    if cooldown and (datetime.now() - cooldown).total_seconds() < config.TRADE_COOLDOWN_MINUTES * 60:
        return True
    return False

# === Helper: Sell Conditions ===
def has_open_position(ticker: str) -> bool:
    for pos in trading_state.open_positions:
        if pos["ticker"] == ticker:
            return True
    return False

# === Execution Logic ===
def execute_buy(ticker: str, features=None, size: int = 1) -> bool:
    """Places a buy order for the ticker."""
    try:
        order = api_manager.safe_api_call(lambda: api_manager.api.submit_order(
            symbol=ticker,
            qty=size,
            side='buy',
            type='market',
            time_in_force='day'
        ))
        if order:
            trading_state.open_positions.append({
                "ticker": ticker,
                "entry_time": datetime.now(),
                "entry_price": get_price(ticker),
                "features": features
            })
            update_cooldown(ticker)
            logger.deduped_log("info", f"🟢 BUY {ticker}")
            return True
    except Exception as e:
        logger.error(f"❌ Buy execution failed for {ticker}: {e}")
    return False

def execute_sell(ticker: str, features=None) -> bool:
    """Closes a position in the ticker."""
    try:
        order = api_manager.safe_api_call(lambda: api_manager.api.submit_order(
            symbol=ticker,
            qty=1,
            side='sell',
            type='market',
            time_in_force='day'
        ))
        if order:
            trading_state.open_positions = [
                pos for pos in trading_state.open_positions if pos["ticker"] != ticker
            ]
            update_cooldown(ticker)
            logger.deduped_log("info", f"🔴 SELL {ticker}")
            return True
    except Exception as e:
        logger.error(f"❌ Sell execution failed for {ticker}: {e}")
    return False

def get_price(ticker: str) -> float:
    """Fetches latest price (stub)."""
    return 100.0  # Stub — you may want to pull live price here

def update_cooldown(ticker: str):
    if not hasattr(trading_state, "cooldown_map"):
        trading_state.cooldown_map = {}
    trading_state.cooldown_map[ticker] = datetime.now()

def is_meta_approved(ticker: str, proba: float) -> bool:
    """
    Rejects trades unless the model meets accuracy and sample thresholds.
    """
    if not meta_approval_system.is_model_approved(ticker, proba):
        logger.deduped_log("warn", f"⚠️ Trade rejected by meta model for {ticker}")
        return False
    return True

def calculate_kelly_position_size(ticker: str, confidence: float) -> int:
    """
    Uses Kelly Criterion and account balance to size trades.
    """
    try:
        if config.POSITION_SIZING_MODE.lower() == "fixed":
            return config.FIXED_TRADE_AMOUNT

        kelly_fraction = ((2 * confidence) - 1) * config.KELLY_MULTIPLIER
        kelly_fraction = max(0.01, min(kelly_fraction, config.MAX_PORTFOLIO_RISK))

        account = api_manager.safe_api_call(api_manager.api.get_account)
        if account:
            equity = float(account.cash)
            return max(1, int(equity * kelly_fraction))

    except Exception as e:
        logger.error(f"❌ Kelly sizing failed for {ticker}: {e}")
    return config.FIXED_TRADE_AMOUNT

def log_trade_outcome(ticker: str, result: dict):
    """
    Save PnL, reward, model inputs for RL training.
    """
    outcome = {
        "ticker": ticker,
        "time": datetime.now().isoformat(),
        **result
    }
    trading_state.trade_outcomes.append(outcome)

def passes_sector_allocation(ticker: str) -> bool:
    """
    Ensures portfolio isn't over-concentrated in one sector.
    Replace with real sector data lookup.
    """
    # Example: limit 2 tech stocks
    tech_tickers = {"AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD"}
    open_tech = [pos for pos in trading_state.open_positions if pos["ticker"] in tech_tickers]

    if ticker in tech_tickers and len(open_tech) >= 2:
        return False
    return True

def should_exit_due_to_profit_decay(ticker: str) -> bool:
    """
    Exit trade if price has stalled or regressed from peak.
    Placeholder version — replace with trailing peak % logic.
    """
    return False  # Implement real trailing stop later

def should_add_to_position(ticker: str, confidence: float) -> bool:
    """
    Allow scaling into position if it's moving in favor and confidence is high.
    """
    if not has_open_position(ticker):
        return False
    if confidence >= 0.9:
        return True
    return False

