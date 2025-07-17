# sentiment_analysis.py

from datetime import datetime
import traceback
from config import config
from trading_state import trading_state
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
        if not finbert_pipeline:
            return 0.0
        result = finbert_pipeline(text)[0]
        label = result["label"]
        return 1.0 if label == "positive" else -1.0 if label == "negative" else 0.0
    except Exception as e:
        logger.logger.warning(f"⚠️ FinBERT failed: {e}")
        return 0.0

def get_vader_sentiment(text: str) -> float:
    try:
        if not vader:
            return 0.0
        return vader.polarity_scores(text)["compound"]
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
    Cached combined sentiment score for a ticker.
    """
    try:
        cache = trading_state.sentiment_cache.get(ticker)
        now = datetime.now()

        if cache and (now - cache["timestamp"]).total_seconds() < 1800:
            return cache["score"]

        text = fetch_reddit_and_news(ticker)
        score = get_combined_sentiment(text)

        trading_state.sentiment_cache[ticker] = {
            "score": score,
            "timestamp": now
        }

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
                headlines = [a["title"] for a in response["articles"] if a.get("title")]

        # Extend this later with Reddit fetching
        reddit_stub = f"{ticker} is trending on Reddit! Bulls are excited."

        combined_text = ". ".join(headlines) + ". " + reddit_stub
        return combined_text if combined_text.strip() else f"{ticker} is in the news."

    except Exception as e:
        logger.logger.warning(f"⚠️ Failed to fetch news for {ticker}: {e}")
        return f"{ticker} is up today. Positive outlook."
