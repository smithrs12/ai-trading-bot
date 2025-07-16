# execution_manager.py

import time
from datetime import datetime

from config import config
from trading_state import trading_state
from main_user_isolated import market_status
from ensemble_model import ensemble_model
from reinforcement import PyTorchQLearningAgent
from technical_indicators import passes_vwap, passes_volume_spike, extract_features
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

        features = extract_features(ticker)
        action_index = q_agent.act(features)
        action = ACTION_MAP[action_index]

        logger.deduped_log("debug", f"📊 {ticker} - RL Decision: {action}")

        if action == 'buy' and can_enter_position(ticker):
            return execute_buy(ticker, features)
        elif action == 'sell' and has_open_position(ticker):
            return execute_sell(ticker, features)

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
    # This is where you'd call: passes_vwap(), passes_volume_spike(), passes_sentiment(), etc.
    return True  # Stub: allow all

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
def execute_buy(ticker: str, features=None) -> bool:
    """Places a buy order for the ticker."""
    try:
        order = api_manager.safe_api_call(lambda: api_manager.api.submit_order(
            symbol=ticker,
            qty=1,
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
