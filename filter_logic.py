from sentiment_analysis import get_sentiment
from technical_indicators import (
    passes_vwap,
    passes_volume_spike,
    calculate_price_momentum,
    get_indicator_snapshot
)
from config import config
import logger

def passes_all_filters(ticker: str, data=None, regime: str = "neutral") -> bool:
    """
    Combines sentiment, VWAP, volume, momentum, RSI, ADX, and regime checks.
    Returns True if all filters pass.
    """
    logger.debug(f"🔍 Filtering {ticker} for USER={config.USER_ID}")

    # === Sentiment filter ===
    sentiment_score = get_sentiment(ticker)
    if sentiment_score is None or sentiment_score < config.SENTIMENT_THRESHOLD:
        logger.debug(f"⛔ {ticker} rejected by sentiment: {sentiment_score}")
        return False

    # === VWAP filter ===
    if not passes_vwap(ticker):
        logger.debug(f"⛔ {ticker} rejected by VWAP filter")
        return False

    # === Volume spike filter ===
    if not passes_volume_spike(ticker):
        logger.debug(f"⛔ {ticker} rejected by volume spike filter")
        return False

    # === Load technical indicators ===
    if data is None:
        data = get_indicator_snapshot(ticker)
        if data is None or data.empty:
            logger.debug(f"⛔ {ticker} rejected: no data snapshot")
            return False

    latest = data.iloc[-1]

    # === Price momentum filter ===
    price_momentum = calculate_price_momentum(ticker)
    if abs(price_momentum) < config.PRICE_MOMENTUM_MIN:
        logger.debug(f"⛔ {ticker} rejected by momentum: {price_momentum}")
        return False

    # === RSI filter ===
    rsi = latest.get("rsi", None)
    rsi_min = getattr(config, "RSI_MIN", 30)
    rsi_max = getattr(config, "RSI_MAX", 70)
    if rsi is None or rsi < rsi_min or rsi > rsi_max:
        logger.debug(f"⛔ {ticker} rejected by RSI: {rsi}")
        return False

    # === ADX (trend strength) filter ===
    adx = latest.get("adx", None)
    if adx is not None and adx < config.ADX_MIN_THRESHOLD:
        logger.debug(f"⛔ {ticker} rejected by ADX: {adx}")
        return False

    # === Optional Bollinger Band filter ===
    bb_pos = latest.get("bb_position", 0.5)
    if bb_pos > 0.95 or bb_pos < 0.05:
        logger.debug(f"⛔ {ticker} rejected by Bollinger position: {bb_pos}")
        return False

    # === Market regime filter ===
    if config.ENFORCE_REGIME_FILTER and regime == "bearish":
        logger.debug(f"⛔ {ticker} rejected by regime: {regime}")
        return False

    return True
from sentiment_analysis import get_sentiment
from technical_indicators import (
    passes_vwap,
    passes_volume_spike,
    get_indicator_snapshot
)
from config import config
from logger import logger

def calculate_price_momentum(ticker: str) -> float:
    """Calculate price momentum for a ticker"""
    try:
        data = get_indicator_snapshot(ticker)
        if data is None or data.empty or len(data) < 5:
            return 0.0
        
        # Calculate momentum as percentage change over last 5 periods
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-5]
        momentum = (current_price - past_price) / past_price
        return momentum
    except Exception as e:
        logger.warning(f"Failed to calculate momentum for {ticker}: {e}")
        return 0.0

def passes_all_filters(ticker: str, data=None, regime: str = "neutral") -> bool:
    """
    Combines sentiment, VWAP, volume, momentum, RSI, ADX, and regime checks.
    Returns True if all filters pass.
    """
    logger.debug(f"🔍 Filtering {ticker} for USER={config.USER_ID}")

    # === Sentiment filter ===
    sentiment_score = get_sentiment(ticker)
    sentiment_threshold = getattr(config, "SENTIMENT_THRESHOLD", 0.1)
    if sentiment_score is None or sentiment_score < sentiment_threshold:
        logger.debug(f"⛔ {ticker} rejected by sentiment: {sentiment_score}")
        return False

    # === VWAP filter ===
    if not passes_vwap(ticker):
        logger.debug(f"⛔ {ticker} rejected by VWAP filter")
        return False

    # === Volume spike filter ===
    if not passes_volume_spike(ticker):
        logger.debug(f"⛔ {ticker} rejected by volume spike filter")
        return False

    # === Load technical indicators ===
    if data is None:
        data = get_indicator_snapshot(ticker)
        if data is None or data.empty:
            logger.debug(f"⛔ {ticker} rejected: no data snapshot")
            return False

    latest = data.iloc[-1]

    # === Price momentum filter ===
    price_momentum = calculate_price_momentum(ticker)
    momentum_min = getattr(config, "PRICE_MOMENTUM_MIN", 0.01)
    if abs(price_momentum) < momentum_min:
        logger.debug(f"⛔ {ticker} rejected by momentum: {price_momentum}")
        return False

    # === RSI filter ===
    rsi = latest.get("rsi", None)
    rsi_min = getattr(config, "RSI_MIN", 30)
    rsi_max = getattr(config, "RSI_MAX", 70)
    if rsi is None or rsi < rsi_min or rsi > rsi_max:
        logger.debug(f"⛔ {ticker} rejected by RSI: {rsi}")
        return False

    # === ADX (trend strength) filter ===
    adx = latest.get("adx", None)
    adx_min = getattr(config, "ADX_MIN_THRESHOLD", 25)
    if adx is not None and adx < adx_min:
        logger.debug(f"⛔ {ticker} rejected by ADX: {adx}")
        return False

    # === Optional Bollinger Band filter ===
    bb_pos = latest.get("bb_position", 0.5)
    if bb_pos > 0.95 or bb_pos < 0.05:
        logger.debug(f"⛔ {ticker} rejected by Bollinger position: {bb_pos}")
        return False

    # === Market regime filter ===
    enforce_regime = getattr(config, "ENFORCE_REGIME_FILTER", False)
    if enforce_regime and regime == "bearish":
        logger.debug(f"⛔ {ticker} rejected by regime: {regime}")
        return False

    return True
