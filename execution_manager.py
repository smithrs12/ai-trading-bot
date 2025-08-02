# execution_manager.py

import time
from datetime import datetime

from config import config
from trading_state import trading_state
from globals import market_status, redis_cache
from model_training import ensemble_model
from reinforcement import PyTorchQLearningAgent
from technical_indicators import passes_vwap, passes_volume_spike, extract_features
from meta_approval_system import meta_approval_system
from risk_management import risk_manager
from sentiment_analysis import get_sentiment
from filter_logic import passes_all_filters
import api_manager
import logger
import os
import redis
import streamlit as st
from alpaca_trade_api.rest import REST

api = st.session_state.get('alpaca_api')
if not api:
    # Don't stop the app, just disable trading
    print("⚠️ No Alpaca API credentials provided. Trading will be disabled.")
    trading_state.trading_disabled = True
    trading_state.disabled_reason = "No API credentials provided"

REDIS_URL = os.getenv("REDIS_URL")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

def check_subscription(user_id):
    active = redis_client.get(f"user:{user_id}:active")
    return active == "true"

def check_alpaca_subscription_status():
    """Check if the current Alpaca account has the required subscription for SIP data"""
    try:
        if not api:
            return False, "No API credentials"
        
        # Try to get account info
        account = api.get_account()
        
        # Try to access SIP data
        try:
            api.get_latest_quote('AAPL')
            return True, "SIP subscription active"
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "permission" in error_msg.lower() or "subscription" in error_msg.lower():
                return False, "SIP subscription required"
            else:
                return True, "API access available"
                
    except Exception as e:
        return False, f"API connection error: {str(e)}"

# Check both user subscription and Alpaca subscription
user_subscribed = check_subscription(config.USER_ID)
alpaca_subscribed, alpaca_msg = check_alpaca_subscription_status()

if not user_subscribed:
    print(f"🚫 User {config.USER_ID} is not subscribed. Trading disabled but bot remains active.")
    trading_state.trading_disabled = True
    trading_state.disabled_reason = "No active subscription"
elif not alpaca_subscribed:
    print(f"🚫 Alpaca subscription issue: {alpaca_msg}. Trading disabled but bot remains active.")
    trading_state.trading_disabled = True
    trading_state.disabled_reason = f"Alpaca subscription: {alpaca_msg}"
else:
    trading_state.trading_disabled = False
    trading_state.disabled_reason = None
    print(f"✅ All subscription checks passed for user {config.USER_ID}")

# === Reinforcement Learning Agent ===
q_agent = PyTorchQLearningAgent()

# === Decision Mapping ===
ACTION_MAP = {0: 'buy', 1: 'sell', 2: 'hold'}

# === Ultra Advanced Trading Logic ===
def ultra_advanced_trading_logic(ticker: str) -> bool:
    """Performs signal checks, model prediction, and executes a trade if approved."""
    try:
        # ⬅️ PATCHED: Refresh user-specific config before each trade logic
        config.reload_user_settings()
        
        # Check if trading is disabled
        if trading_state.trading_disabled:
            logger.deduped_log("info", f"🚫 Trading disabled: {trading_state.disabled_reason}")
            return False
            
        # Check if API is available
        if not api:
            logger.deduped_log("warning", f"🚫 No API available for {ticker}")
            return False
            
        if not passes_all_filters(ticker):
            return False

        # Get features  confidence from ensemble
        features = extract_features(ticker)
        confidence = ensemble_model.predict_weighted_proba(ticker)
        action_index = q_agent.act(features)
        action = ACTION_MAP[action_index]

        logger.deduped_log("debug", f"📊 {ticker} - RL Decision: {action} | Confidence: {confidence:.2f}")

        if not passes_sector_allocation(ticker):
            return False

        if risk_manager.block_trades_if_risky():
            logger.warning(f"🚫 Trade blocked due to risk controls for {ticker}")
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
    key = redis_key("COOLDOWN", ticker, user_id=config.USER_ID)
    timestamp = redis_cache.get(key)
    if timestamp:
        try:
            last_time = datetime.fromisoformat(timestamp)
            if (datetime.now() - last_time).total_seconds() < config.TRADE_COOLDOWN_MINUTES * 60:
                return True
        except:
            return False
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
        cache_key = redis_key("LAST_PRICE", ticker, user_id=config.USER_ID)
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
        logger.warning(f"⚠️ get_price failed for {ticker}: {e}")
    return 100.0  # fallback

def update_cooldown(ticker: str):
    key = redis_key("COOLDOWN", ticker, user_id=config.USER_ID)
    redis_cache.set(key, datetime.now().isoformat(), ttl_seconds=config.TRADE_COOLDOWN_MINUTES * 60)

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
    Position sizing with per-trade dollar cap.
    - In 'fixed' mode, interpret FIXED_TRADE_AMOUNT as a *dollar budget*.
    - In Kelly mode, cap the computed dollars by MAX_TRADE_AMOUNT.
    Returns an integer share quantity (>= 1 when possible).
    """
    try:
        price = float(get_price(ticker) or 100.0)
        # Safety for zero/negative prices
        if price <= 0:
            price = 100.0

        # Guaranteed cap from settings (fallback to a sane default)
        max_dollars = float(getattr(config, "MAX_TRADE_AMOUNT", 10000))

        mode = str(getattr(config, "POSITION_SIZING_MODE", "Kelly Criterion")).lower()
        if mode == "fixed":
            # Interpret FIXED_TRADE_AMOUNT as dollars; enforce MAX_TRADE_AMOUNT; convert to shares
            base_dollars = float(getattr(config, "FIXED_TRADE_AMOUNT", 1000))
            dollars = min(base_dollars, max_dollars)
            qty = max(1, int(dollars // price))
            return qty

        # Kelly sizing
        kelly_fraction = ((2.0 * float(confidence)) - 1.0) * float(getattr(config, "KELLY_MULTIPLIER", 0.5))
        kelly_fraction = max(0.01, min(kelly_fraction, float(getattr(config, "MAX_PORTFOLIO_RISK", 0.25))))

        account = api_manager.safe_api_call(api_manager.api.get_account)
        if account:
            equity_cash = float(account.cash)
            desired_dollars = equity_cash * kelly_fraction
            dollars = min(desired_dollars, max_dollars)
            qty = max(1, int(dollars // price))
            return qty

    except Exception as e:
        logger.error(f"❌ Kelly sizing failed for {ticker}: {e}")

    # Fallback: respect cap as best we can
    try:
        price = float(get_price(ticker) or 100.0)
        max_dollars = float(getattr(config, "MAX_TRADE_AMOUNT", 10000))
        return max(1, int(max_dollars // price))
    except Exception:
        return 1

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

    # Optionally update equity curve
    try:
        account = api_manager.safe_api_call(api_manager.api.get_account)
        if account:
            trading_state.equity_curve.append({
                "time": datetime.now(),
                "equity": float(account.equity)
            })
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

def update_subscription_status():
    """Periodically check and update subscription status"""
    try:
        user_subscribed = check_subscription(config.USER_ID)
        alpaca_subscribed, alpaca_msg = check_alpaca_subscription_status()
        
        if not user_subscribed:
            trading_state.trading_disabled = True
            trading_state.disabled_reason = "No active subscription"
            return False
        elif not alpaca_subscribed:
            trading_state.trading_disabled = True
            trading_state.disabled_reason = f"Alpaca subscription: {alpaca_msg}"
            return False
        else:
            trading_state.trading_disabled = False
            trading_state.disabled_reason = None
            return True
    except Exception as e:
        logger.error(f"Error updating subscription status: {e}")
        trading_state.trading_disabled = True
        trading_state.disabled_reason = f"Subscription check error: {e}"
        return False
