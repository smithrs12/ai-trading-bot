from datetime import datetime
import traceback
from config import config
from trading_state import trading_state
from main_user_isolated import redis_cache, redis_key
import logger
import api_manager

try:
    from transformers import pipeline
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download("vader_lexicon", quiet=True)

    finbert_pipeline = pipeline("sentiment-analysis", model=config.FINBERT_MODEL_NAME)
    vader = SentimentIntensityAnalyzer()
except Exception as e:
    finbert_pipeline = None
    vader = None
    logger.logger.warning(f"⚠️ Sentiment analyzers not fully loaded: {e}")

def get_finbert_sentiment(text: str) -> float:
    try:
        if not finbert_pipeline or not text.strip():
            return 0.0
        result = finbert_pipeline(text)[0]
        label = result["label"]
        score = 1.0 if label == "positive" else -1.0 if label == "negative" else 0.0
        logger.deduped_log("debug", f"🧠 FinBERT: {label} → {score}")
        return score
    except Exception as e:
        logger.logger.warning(f"⚠️ FinBERT failed: {e}")
        return 0.0

def get_vader_sentiment(text: str) -> float:
    try:
        if not vader or not text.strip():
            return 0.0
        score = vader.polarity_scores(text)["compound"]
        logger.deduped_log("debug", f"🧠 VADER: {score:.2f}")
        return score
    except Exception as e:
        logger.logger.warning(f"⚠️ VADER failed: {e}")
        return 0.0

def get_combined_sentiment(text: str) -> float:
    try:
        finbert_score = get_finbert_sentiment(text)
        vader_score = get_vader_sentiment(text)
        combined = (finbert_score + vader_score) / 2
        return max(-1.0, min(1.0, combined))
    except Exception as e:
        logger.logger.warning(f"⚠️ Combined sentiment failed: {e}")
        return 0.0

def get_sentiment(ticker: str) -> float:
    """
    Retrieves cached sentiment or performs a new analysis.
    Uses Redis to cache sentiment per ticker.
    """
    try:
        cache_key = redis_key("SENTIMENT_SCORE", ticker)
        cached = redis_cache.get(cache_key)
        if cached:
            return float(cached)

        text = fetch_reddit_and_news(ticker)
        score = get_combined_sentiment(text)

        redis_cache.set(cache_key, score, ttl_seconds=1800)  # 30 minutes cache
        trading_state.sentiment_cache[ticker] = {
            "score": score,
            "timestamp": datetime.now(),
            "headline_summary": text[:250]
        }
        logger.deduped_log("info", f"📰 {ticker} Sentiment: {score:.2f}")
        return score

    except Exception:
        logger.logger.error(f"❌ get_sentiment failed for {ticker}: {traceback.format_exc()}")
        return 0.0

def fetch_reddit_and_news(ticker: str) -> str:
    """
    Pulls headlines from NewsAPI (centralized) and includes Reddit stubs.
    """
    try:
        headlines = []
        if hasattr(api_manager, "news_api") and api_manager.news_api:
            response = api_manager.safe_api_call(
                lambda: api_manager.news_api.get_everything(
                    q=ticker,
                    language="en",
                    sort_by="publishedAt",
                    page_size=10
                )
            )
            if response and response.get("articles"):
                headlines = [a["title"].strip() for a in response["articles"] if a.get("title")]

        # Simulated Reddit sentiment (placeholder)
        reddit_stub = f"{ticker} is trending on Reddit. Some bullish chatter."

        combined_text = ". ".join(headlines) + ". " + reddit_stub
        return combined_text if combined_text.strip() else f"{ticker} is active in financial news."
    except Exception as e:
        logger.logger.warning(f"⚠️ Failed to fetch news for {ticker}: {e}")
        return f"{ticker} is up today. Positive outlook."
