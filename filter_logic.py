from sentiment_analysis import get_sentiment
from technical_indicators import (
    passes_vwap,
    passes_volume_spike,
    calculate_price_momentum,
    get_indicator_snapshot
)
from config import config

def passes_all_filters(ticker: str, data=None, regime: str = "neutral") -> bool:
    """
    Combines all filter checks — sentiment, volume, price action, regime, technicals.
    Optionally accepts `data` (pd.DataFrame) and `regime` string to prevent duplicate calculations.
    """
    from sentiment_analysis import get_sentiment
    from technical_indicators import (
        passes_vwap,
        passes_volume_spike,
        calculate_price_momentum,
        get_indicator_snapshot
    )
    from config import config

    # === Sentiment filter ===
    sentiment_score = get_sentiment(ticker)
    if sentiment_score < 0.1:
        return False

    # === VWAP & volume ===
    if not passes_vwap(ticker):
        return False
    if not passes_volume_spike(ticker):
        return False

    # === Load indicators if not passed
    if data is None:
        data = get_indicator_snapshot(ticker)
        if data is None or data.empty:
            return False
        latest = data.iloc[-1]
    else:
        latest = data.iloc[-1]

    # === Price momentum ===
    price_momentum = calculate_price_momentum(ticker)
    if abs(price_momentum) < config.PRICE_MOMENTUM_MIN:
        return False

    # === RSI filter ===
    rsi = latest.get("rsi", None)
    rsi_min = getattr(config, "RSI_MIN", 30)
    rsi_max = getattr(config, "RSI_MAX", 70)

    if rsi is None or rsi < rsi_min or rsi > rsi_max:
        logger.logger.debug(f"⛔ {ticker} filtered out by RSI: {rsi:.2f} not in [{rsi_min}, {rsi_max}]")
        return False

    # === ADX (trend strength) — optional, if you're calculating it
    adx = latest.get("adx", None)
    if adx is not None and adx < 20:
        return False

    # === Market regime check ===
    if config.ENFORCE_REGIME_FILTER and regime == "bearish":
        return False

    # === Optional filters: Bollinger, MACD divergence, etc.
    # bb_pos = latest.get("bb_position", 0.5)
    # if bb_pos > 0.9 or bb_pos < 0.1:
    #     return False

    return True
