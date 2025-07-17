from api_manager import api, safe_api_call
import numpy as np
import pandas as pd
import ta
from datetime import datetime, timedelta

from main_user_isolated import redis_cache, redis_key
from logger import logger

from main_user_isolated import redis_cache, redis_key  # 👈 Required

def get_indicator_snapshot(ticker: str, minutes: int = 120) -> pd.DataFrame:
    """
    Fetches 1-minute bars from Alpaca and computes technical indicators.
    Uses Redis to cache results for performance.
    """
    cache_key = redis_key("INDICATOR_SNAPSHOT", ticker)
    cached = redis_cache.get(cache_key)
    if cached:
        try:
            return pd.DataFrame(cached)
        except Exception:
            pass  # Fallback to fresh fetch if cache is corrupted

    try:
        end = datetime.utcnow()
        start = end - timedelta(minutes=minutes)

        bars = safe_api_call(lambda: api.get_bars(
            symbol=ticker,
            timeframe="1Min",
            start=start.isoformat() + "Z",
            end=end.isoformat() + "Z",
            adjustment='raw'
        ))

        if not bars or len(bars) < 20:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "timestamp": bar.t,
            "open": bar.o,
            "high": bar.h,
            "low": bar.l,
            "close": bar.c,
            "volume": bar.v
        } for bar in bars])

        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # === Technical Indicators ===
        df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
        ).vwap()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["macd"] = ta.trend.MACD(df["close"]).macd_diff()
        df["roc"] = ta.momentum.ROCIndicator(df["close"]).roc()
        df["volatility"] = df["close"].pct_change().rolling(window=10).std()

        df.dropna(inplace=True)
        if not df.empty:
            redis_cache.set(cache_key, df.tail(50).to_dict(orient="records"), ttl_seconds=300)

        return df

    except Exception as e:
        logger.warning(f"⚠️ Alpaca snapshot failed for {ticker}: {e}")

        return pd.DataFrame()
        return pd.DataFrame()

def extract_features(ticker: str) -> list:
    """
    Return normalized features for the RL model input.
    """
    df = get_indicator_snapshot(ticker)
    if df.empty or len(df) < 1:
        return [0.0] * 10

    latest = df.iloc[-1]

    features = [
        scale(latest["close"]),
        scale(latest["rsi"], 0, 100),
        scale(latest["macd"]),
        scale(latest["vwap"]),
        scale(latest["roc"]),
        scale(latest["volatility"]),
        scale(df["volume"].iloc[-1]),
        scale(latest["high"] - latest["low"]),
        scale(latest["close"] - df["close"].iloc[-2]),
        scale(df["close"].pct_change().mean())
    ]

    return features

def passes_vwap(ticker: str) -> bool:
    df = get_indicator_snapshot(ticker)
    if df.empty:
        return False
    latest = df.iloc[-1]
    return latest["close"] >= latest["vwap"]

def passes_volume_spike(ticker: str) -> bool:
    df = get_indicator_snapshot(ticker)
    if df.empty:
        return False
    recent_vol = df["volume"].iloc[-1]
    avg_vol = df["volume"].rolling(window=20).mean().iloc[-1]
    return recent_vol > 1.5 * avg_vol

def scale(x, min_val=-1, max_val=1):
    try:
        if pd.isna(x):
            return 0.0
        return float(np.clip(x, min_val, max_val))
    except Exception:
        return 0.0
