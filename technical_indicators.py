# technical_indicators.py

import yfinance as yf
import numpy as np
import pandas as pd
import ta

from datetime import datetime, timedelta

def get_price_data(ticker: str, days: int = 14, interval: str = "1h"):
    """
    Fetch OHLCV data using yfinance.
    """
    try:
        df = yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

def extract_features(ticker: str) -> list:
    """
    Return normalized features for the RL model input.
    Shape must match q_agent.state_size (default: 10).
    """
    df = get_price_data(ticker, days=5, interval="1h")
    if df.empty or len(df) < 20:
        return [0.0] * 10

    # === Add Indicators ===
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd_diff()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    ).vwap()
    df["roc"] = ta.momentum.ROCIndicator(df["Close"]).roc()
    df["volatility"] = df["Close"].pct_change().rolling(window=10).std()

    # Clean final row
    df.dropna(inplace=True)
    if df.empty:
        return [0.0] * 10

    latest = df.iloc[-1]

    # === Normalize to fixed range (optional) ===
    features = [
        scale(latest["Close"]),
        scale(latest["rsi"], 0, 100),
        scale(latest["macd"]),
        scale(latest["vwap"]),
        scale(latest["roc"]),
        scale(latest["volatility"]),
        scale(df["Volume"].iloc[-1]),
        scale(latest["High"] - latest["Low"]),
        scale(latest["Close"] - df["Close"].iloc[-2]),
        scale(df["Close"].pct_change().mean())
    ]

    return features

def scale(x, min_val=-1, max_val=1):
    """
    Clamp or normalize values. This version just returns x.
    You can replace with min-max scaling, z-score, etc.
    """
    try:
        if pd.isna(x):
            return 0.0
        return float(np.clip(x, min_val, max_val))
    except Exception:
        return 0.0

def passes_vwap(ticker: str) -> bool:
    df = get_price_data(ticker, days=1, interval="15m")
    if df.empty:
        return False
    latest = df.iloc[-1]
    return latest["Close"] >= latest["vwap"]

def passes_volume_spike(ticker: str) -> bool:
    df = get_price_data(ticker, days=2, interval="15m")
    if df.empty or "Volume" not in df.columns:
        return False
    last_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
    return last_vol > 1.5 * avg_vol
