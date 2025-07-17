# meta_approval_system.py

from config import config
from trading_state import trading_state
from main_user_isolated import redis_cache, redis_key
import logger

class MetaApprovalSystem:
    def __init__(self):
        self.performance_log = {}

    def evaluate_model_performance(self):
        """
        Evaluate all tickers in current state and log if below threshold.
        """
        for ticker, accuracy in trading_state.model_accuracy.items():
            sample_count = trading_state.model_sample_count.get(ticker, 0)
            if accuracy < config.META_MODEL_MIN_ACCURACY or sample_count < config.META_MODEL_MIN_TRADES:
                logger.logger.warning(f"⚠️ {ticker} flagged — accuracy: {accuracy:.2f}, samples: {sample_count}")
            else:
                logger.deduped_log("info", f"✅ {ticker} model approved — accuracy: {accuracy:.2f}, samples: {sample_count}")

    def is_model_approved(self, ticker: str, proba: float) -> bool:
        """
        Approves or rejects a model's prediction based on performance.
        Uses Redis to cache approval per ticker.
        """
        try:
            cache_key = redis_key("META_APPROVAL", ticker)
            cached = redis_cache.get(cache_key)
            if cached is not None:
                return cached

            accuracy = trading_state.model_accuracy.get(ticker, 0)
            sample_count = trading_state.model_sample_count.get(ticker, 0)

            approved = (accuracy >= config.META_MODEL_MIN_ACCURACY and
                        sample_count >= config.META_MODEL_MIN_TRADES and
                        proba >= config.ENSEMBLE_CONFIDENCE_THRESHOLD)

            redis_cache.set(cache_key, approved, ttl_seconds=300)
            return approved
        except Exception as e:
            logger.logger.warning(f"⚠️ Meta approval check failed for {ticker}: {e}")
            return False

meta_approval_system = MetaApprovalSystem()
