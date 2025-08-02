from datetime import datetime
import traceback
from config import config
from trading_state import trading_state
from redis_cache import redis_cache
from key_utils import redis_key
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
    logger.warning(f"⚠️ Sentiment analyzers not fully loaded: {e}")

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
        logger.warning(f"⚠️ FinBERT failed: {e}")
        return 0.0

def get_vader_sentiment(text: str) -> float:
    try:
        if not vader or not text.strip():
            return 0.0
        score = vader.polarity_scores(text)["compound"]
        logger.deduped_log("debug", f"🧠 VADER: {score:.2f}")
        return score
    except Exception as e:
        logger.warning(f"⚠️ VADER failed: {e}")
        return 0.0

def get_combined_sentiment(text: str) -> float:
    try:
        finbert_score = get_finbert_sentiment(text)
        vader_score = get_vader_sentiment(text)
        combined = (finbert_score + vader_score) / 2
        return max(-1.0, min(1.0, combined))
    except Exception as e:
        logger.warning(f"⚠️ Combined sentiment failed: {e}")
        return 0.0

def get_sentiment(ticker: str) -> float:
    """
    Retrieves cached sentiment or performs a new analysis.
    Uses Redis to cache sentiment per ticker.
    """
    try:
        cache_key = redis_key("SENTIMENT_SCORE", ticker, user=config.USER_ID)
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
        logger.error(f"❌ get_sentiment failed for {ticker}: {traceback.format_exc()}")
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
        logger.warning(f"⚠️ Failed to fetch news for {ticker}: {e}")
        return f"{ticker} is up today. Positive outlook."
from datetime import datetime
import traceback
import redis
import os
from urllib.parse import urlparse
from config import config
from trading_state import trading_state
from logger import logger

# Redis setup for caching
def get_redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    try:
        parsed_url = urlparse(redis_url)
        client = redis.Redis(
            host=parsed_url.hostname,
            port=parsed_url.port,
            password=parsed_url.password,
            ssl=parsed_url.scheme == 'rediss',
            decode_responses=True
        )
        client.ping()
        return client
    except Exception:
        return None

redis_client = get_redis_client()

class SimpleRedisCache:
    def __init__(self, client):
        self.client = client
        self.enabled = client is not None

    def get(self, key):
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            return float(value) if value else None
        except Exception:
            return None

    def set(self, key, value, ttl_seconds=1800):
        if not self.enabled:
            return
        try:
            self.client.setex(key, ttl_seconds, str(value))
        except Exception:
            pass

redis_cache = SimpleRedisCache(redis_client)

def redis_key(prefix, ticker, user=None):
    user_id = user or config.USER_ID
    return f"{user_id}:{prefix}:{ticker}"

def deduped_log(level, message, cooldown_sec=60):
    """Simple deduped logging"""
    getattr(logger, level, logger.info)(message)

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
    logger.warning(f"⚠️ Sentiment analyzers not fully loaded: {e}")

def get_finbert_sentiment(text: str) -> float:
    try:
        if not finbert_pipeline or not text.strip():
            return 0.0
        result = finbert_pipeline(text)[0]
        label = result["label"]
        score = 1.0 if label == "positive" else -1.0 if label == "negative" else 0.0
        deduped_log("debug", f"🧠 FinBERT: {label} → {score}")
        return score
    except Exception as e:
        logger.warning(f"⚠️ FinBERT failed: {e}")
        return 0.0

def get_vader_sentiment(text: str) -> float:
    try:
        if not vader or not text.strip():
            return 0.0
        score = vader.polarity_scores(text)["compound"]
        deduped_log("debug", f"🧠 VADER: {score:.2f}")
        return score
    except Exception as e:
        logger.warning(f"⚠️ VADER failed: {e}")
        return 0.0

def get_combined_sentiment(text: str) -> float:
    try:
        finbert_score = get_finbert_sentiment(text)
        vader_score = get_vader_sentiment(text)
        combined = (finbert_score + vader_score) / 2
        return max(-1.0, min(1.0, combined))
    except Exception as e:
        logger.warning(f"⚠️ Combined sentiment failed: {e}")
        return 0.0

def get_sentiment(ticker: str) -> float:
    """
    Retrieves cached sentiment or performs a new analysis.
    Uses Redis to cache sentiment per ticker.
    """
    try:
        cache_key = redis_key("SENTIMENT_SCORE", ticker, user=config.USER_ID)
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
        deduped_log("info", f"📰 {ticker} Sentiment: {score:.2f}")
        return score

    except Exception:
        logger.error(f"❌ get_sentiment failed for {ticker}: {traceback.format_exc()}")
        return 0.0

def fetch_reddit_and_news(ticker: str) -> str:
    """
    Pulls headlines from NewsAPI (centralized) and includes Reddit stubs.
    """
    try:
        headlines = []
        try:
            import api_manager
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
        except ImportError:
            pass

        # Simulated Reddit sentiment (placeholder)
        reddit_stub = f"{ticker} is trending on Reddit. Some bullish chatter."

        combined_text = ". ".join(headlines) + ". " + reddit_stub
        return combined_text if combined_text.strip() else f"{ticker} is active in financial news."
    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch news for {ticker}: {e}")
        return f"{ticker} is up today. Positive outlook."
