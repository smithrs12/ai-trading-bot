# execution_manager.py

import time
from datetime import datetime

from config import config
from trading_state import trading_state
from main_user_isolated import market_status, redis_cache, redis_key
from ensemble_model import ensemble_model
from reinforcement import PyTorchQLearningAgent
from technical_indicators import passes_vwap, passes_volume_spike, extract_features
from meta_approval_system import meta_approval_system
from risk_management import risk_manager
from sentiment_analysis import get_sentiment
from filter_logic import passes_all_filters
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

        elif action == 'hold' and has_open_position(ticker):
            if should_hold_position(ticker, confidence):
                logger.deduped_log("info", f"🤝 HOLD {ticker}")
                return False

        return False

    except Exception as e:
        logger.error(f"❌ Trade logic error for {ticker}: {e}")
        return False

def get_sector_for_ticker(ticker: str) -> str:
    """
    Returns the sector for a given ticker. Replace with live data for production.
    """
    # Example static mapping — replace with API lookup or cached reference
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "NVDA": "Technology",
        "META": "Technology",
        "AMD": "Technology",
        "TSLA": "Consumer Discretionary",
        "AMZN": "Consumer Discretionary",
        "JPM": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
        "WMT": "Consumer Staples",
        "KO": "Consumer Staples"
        # Add more as needed
    }
    return sector_map.get(ticker.upper(), "Other")
        
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
            entry_price = get_price(ticker)

            # Log new position in both trackers
            position = {
                "ticker": ticker,
                "entry_time": datetime.now(),
                "entry_price": entry_price,
                "features": features,
                "size": size,
                "peak_price": entry_price,  # Track peak price
            }
            sector = get_sector_for_ticker(ticker)  # You may already have this
            position["sector"] = sector
            trading_state.sector_allocations[ticker] = sector
            trading_state.open_positions.append(position)
            trading_state.positions_by_ticker[ticker] = position

            update_cooldown(ticker)

            logger.deduped_log("info", f"🟢 BUY {ticker} @ ${entry_price:.2f} x{size}")
            return True

    except Exception as e:
        logger.error(f"❌ Buy execution failed for {ticker}: {e}")
    return False

def execute_sell(ticker: str, features=None) -> bool:
    """Closes a position in the ticker."""
    try:
        entry = trading_state.positions_by_ticker.get(ticker)
        size = entry.get("size", 1) if entry else 1

        order = api_manager.safe_api_call(lambda: api_manager.api.submit_order(
            symbol=ticker,
            qty=size,
            side='sell',
            type='market',
            time_in_force='day'
        ))

        if order:
            exit_price = get_price(ticker)
            pnl = None

            if entry:
                entry_price = entry.get("entry_price", 0)
                pnl = round((exit_price - entry_price) * size, 2)

                outcome = {
                    "ticker": ticker,
                    "entry_time": entry.get("entry_time"),
                    "exit_time": datetime.now(),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "size": size,
                    "hold_duration": datetime.now() - entry.get("entry_time", datetime.now())
                }

                trading_state.trade_outcomes.append(outcome)
                del trading_state.positions_by_ticker[ticker]

            trading_state.open_positions = [
                pos for pos in trading_state.open_positions if pos["ticker"] != ticker
            ]

            update_cooldown(ticker)
            logger.deduped_log("info", f"🔴 SELL {ticker} @ ${exit_price:.2f} | PnL: ${pnl:.2f}" if pnl is not None else f"🔴 SELL {ticker}")
            return True

    except Exception as e:
        logger.error(f"❌ Sell execution failed for {ticker}: {e}")
    return False
    
def get_price(ticker: str) -> float:
    """
    Fetch latest price using Alpaca, with Redis cache fallback.
    """
    try:
        cache_key = redis_key("LAST_PRICE", ticker)
        cached = redis_cache.get(cache_key)
        if cached:
            return cached

        bars = api_manager.safe_api_call(
            lambda: api_manager.api.get_latest_trade(ticker)
        )
        if bars and hasattr(bars, "price"):
            price = float(bars.price)
            redis_cache.set(cache_key, price, ttl_seconds=60)
            return price
    except Exception as e:
        logger.logger.warning(f"⚠️ get_price failed for {ticker}: {e}")
    return 100.0  # fallback

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
    
    try:
        if hasattr(trading_state, "log_equity_curve"):
            trading_state.log_equity_curve()
    except Exception as e:
        logger.warning(f"⚠️ Equity logging failed: {e}")

def passes_sector_allocation(ticker: str) -> bool:
    """
    Ensure no overconcentration in any one sector.
    """
    sector = get_sector_for_ticker(ticker)
    current_allocations = list(trading_state.sector_allocations.values())
    sector_count = current_allocations.count(sector)
    if sector_count >= config.MAX_SECTOR_ALLOCATION:
        logger.deduped_log("warn", f"⚠️ Sector cap reached for {sector}")
        return False
    return True

def should_hold_position(ticker: str, confidence: float) -> bool:
    """
    Decides whether to hold a position based on sentiment or technicals.
    Placeholder: hold if confidence is mid-range.
    """
    if 0.4 < confidence < 0.7:
        return True
    return False

def should_exit_due_to_profit_decay(ticker: str) -> bool:
    position = trading_state.positions_by_ticker.get(ticker)
    if not position:
        return False

    current_price = get_price(ticker)
    peak_price = position.get("peak_price", current_price)
    entry_price = position["entry_price"]

    # Update peak if current is higher
    if current_price > peak_price:
        position["peak_price"] = current_price
        return False

    # Drawdown from peak
    drawdown = (peak_price - current_price) / peak_price
    gain = (current_price - entry_price) / entry_price
    profit_decay_exit = gain > 0.03 and drawdown > 0.01

    # Aging out logic
    max_hold_minutes = getattr(config, "MAX_HOLD_DURATION_MINUTES", 180)
    age_minutes = (datetime.now() - position["entry_time"]).total_seconds() / 60
    age_exit = age_minutes > max_hold_minutes

    if age_exit:
        logger.deduped_log("info", f"⏳ Exiting {ticker} due to max hold duration ({int(age_minutes)} min)")
    if profit_decay_exit:
        logger.deduped_log("info", f"📉 Exiting {ticker} due to profit decay (gain={gain:.2%}, drawdown={drawdown:.2%})")

    return age_exit or profit_decay_exit

def should_add_to_position(ticker: str, confidence: float) -> bool:
    """
    Allow scaling into position if it's moving in favor and confidence is high.
    """
    if not has_open_position(ticker):
        return False
    if confidence >= 0.9:
        return True
    return False

